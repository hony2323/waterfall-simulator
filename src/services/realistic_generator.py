import asyncio
import time
from datetime import datetime, timezone

import numpy as np

from ..base.metrics import MetricsCollector
from ..base.models import BandFrame
from ..base.state import StateManager
from ..config.settings import Settings
from ..core.interfaces import IDataService

# Signal definitions as fractions of band width so they scale with any band config.
# Each entry: (center_frac, sigma_frac, peak_dB)
# sigma_frac × band_size = sigma in bins; 1 bin = step Hz (default 2500 Hz)
#   ~1  bin  → CW beacon / unmodulated carrier   (~2.5 kHz)
#   ~3  bins → NFM voice / narrowband digital    (~7.5 kHz)
#   ~8  bins → wideband digital / TETRA          (~20 kHz)
#   ~20 bins → WFM broadcast                     (~50 kHz)
_STATIC_CARRIERS = [
    (0.12, 0.00004, 58),   # CW beacon — ~1 bin wide
    (0.31, 0.00012, 50),   # NFM voice — ~3 bins wide
    (0.58, 0.00003, 62),   # strong CW — ~1 bin wide
    (0.79, 0.00032, 46),   # wideband digital — ~8 bins wide
    (0.91, 0.00080, 44),   # WFM-like broadcast — ~20 bins wide
]

# Frequency-hop channel positions (fractions of band width)
_HOP_CHANNELS = [0.22, 0.35, 0.47, 0.61, 0.74, 0.84]
_HOP_DWELL_S  = 0.4   # seconds per channel
_HOP_ON_FRAC  = 0.75  # fraction of dwell time the signal is active


def _gaussian(size: int, center: int, sigma: float, peak: float) -> np.ndarray:
    """Return a Gaussian-shaped signal added to a zero array of `size` samples."""
    x = np.arange(size, dtype=np.float32)
    return peak * np.exp(-0.5 * ((x - center) / max(1.0, sigma)) ** 2)


class RealisticGeneratorService(IDataService):
    """
    Generates realistic-looking RF spectrogram frames.

    Signals present:
    - Noise floor: chi-squared-distributed thermal noise (~20 dB, ±3 dB variation)
    - Static narrow carriers with Gaussian IF-filter shape
    - One AM-modulated carrier (amplitude varies at ~1 Hz)
    - One slowly drifting carrier (sinusoidal drift across ~30% of band)
    - Frequency-hopping spread-spectrum signal (cycles through fixed channels)
    - Periodic burst signal with smooth rise/fall envelope (~3 s on, ~8 s cycle)
    - Intermittent wideband noise source (~2 s on, ~13 s cycle)
    """

    def __init__(
        self, state: StateManager, settings: Settings, metrics: MetricsCollector
    ) -> None:
        self._state    = state
        self._settings = settings
        self._metrics  = metrics
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        interval = self._settings.generation_interval_ms / 1000.0
        while True:
            ts = datetime.now(timezone.utc)
            t  = time.time()

            for band in self._settings.bands:
                step = band.step if band.step is not None else self._settings.step
                size = max(1, round((band.end - band.start) / step))
                t0   = time.perf_counter()
                data = _generate(size, t)
                frame = BandFrame(
                    band_id=band.id,
                    band_start=band.start,
                    band_end=band.end,
                    timestamp=ts,
                    data=data,
                )
                self._state.push(band.id, frame)
                duration_ms = (time.perf_counter() - t0) * 1000
                depth       = len(self._state.get_band_history(band.id))
                self._metrics.record_frame_generated(band.id, duration_ms, duration_ms, depth)

            await asyncio.sleep(interval)


def _generate(size: int, t: float) -> np.ndarray:
    """
    Build one frame of `size` float32 samples in the 0–100 range.

    All frequency positions use fractional indices so the output is
    independent of band width / step configuration.
    """
    # ── Noise floor ────────────────────────────────────────────────────────
    # Chi-squared(2) = sum of two squared Gaussians → heavier tail than normal,
    # closer to real thermal + receiver noise.
    noise_floor = (
        np.random.normal(0, 1, size) ** 2 +
        np.random.normal(0, 1, size) ** 2
    ).astype(np.float32)
    # Scale: chi-sq(2) has mean=2, std≈1.4 → shift/scale to mean≈20, std≈3
    frame = noise_floor * 2.1 + 15.8

    # Slow drift on the noise floor baseline (±4 dB, period ~30 s)
    frame += float(np.sin(t * 0.21) * 4.0)

    # ── Static carriers ────────────────────────────────────────────────────
    for frac, sigma_frac, peak in _STATIC_CARRIERS:
        center = int(frac * size)
        sigma  = max(1.0, sigma_frac * size)
        frame += _gaussian(size, center, sigma, peak)

    # ── AM-modulated carrier — NFM-width, amplitude varies ~±8 dB at 0.8 Hz ──
    am_center = int(0.44 * size)
    am_sigma  = max(1.0, 0.00010 * size)   # ~2.5 bins (NFM)
    am_depth  = 1.0 + 0.35 * np.sin(t * 0.8 * 2 * np.pi)
    frame    += _gaussian(size, am_center, am_sigma, 42.0 * float(am_depth))

    # ── Slowly drifting carrier — CW-width, sinusoidal path, period ~20 s ──
    drift_frac   = 0.15 + 0.50 * (np.sin(t * (2 * np.pi / 20.0)) * 0.5 + 0.5)
    drift_center = int(drift_frac * size)
    drift_sigma  = max(1.0, 0.00004 * size)   # ~1 bin (CW)
    frame       += _gaussian(size, drift_center, drift_sigma, 38.0)

    # ── Frequency-hopping signal — narrowband digital width ────────────────
    hop_cycle     = t % (_HOP_DWELL_S * len(_HOP_CHANNELS))
    hop_slot      = int(hop_cycle / _HOP_DWELL_S)
    hop_phase     = (hop_cycle % _HOP_DWELL_S) / _HOP_DWELL_S
    if hop_phase < _HOP_ON_FRAC:
        ramp       = min(1.0, hop_phase / 0.10, ((_HOP_ON_FRAC - hop_phase) / 0.10))
        hop_center = int(_HOP_CHANNELS[hop_slot % len(_HOP_CHANNELS)] * size)
        hop_sigma  = max(1.0, 0.00006 * size)   # ~1.5 bins
        frame     += _gaussian(size, hop_center, hop_sigma, 36.0 * ramp)

    # ── Burst signal — wideband digital width, 3 s on / 5 s off ──────────
    burst_cycle = t % 8.0
    if burst_cycle < 3.0:
        ramp        = min(1.0, burst_cycle / 0.15, (3.0 - burst_cycle) / 0.15)
        burst_c     = int(0.66 * size)
        burst_sigma = max(1.0, 0.00030 * size)   # ~7.5 bins (digital burst)
        frame      += _gaussian(size, burst_c, burst_sigma, 58.0 * ramp)

    # ── Intermittent wideband noise source ─────────────────────────────────
    # Simulates a nearby broadband emitter: ~2 s on every ~13 s.
    wb_cycle = t % 13.0
    if 5.5 < wb_cycle < 7.5:
        ramp   = min(1.0, (wb_cycle - 5.5) / 0.2, (7.5 - wb_cycle) / 0.2)
        lo, hi = int(0.25 * size), int(0.45 * size)
        frame[lo:hi] += (
            np.random.normal(0, 1, hi - lo) ** 2 * 3.0 * ramp + 18.0 * ramp
        ).astype(np.float32)

    return np.clip(frame, 0.0, 100.0).astype(np.float32)
