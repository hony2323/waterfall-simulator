"""Microphone input service.

Captures live audio via sounddevice, runs a windowed FFT on each chunk,
and pushes per-band magnitude spectra (scaled 0-100) into the shared state.

Band boundaries in Settings.bands are interpreted as Hz.  The FFT frequency
resolution is  sample_rate / fft_size  Hz per bin, so effective array_size
per band  =  round((band.end - band.start) * fft_size / sample_rate).

Install dependency:  pip install sounddevice
"""

import asyncio
import time
from datetime import datetime, timezone

import numpy as np

from ..base.metrics import MetricsCollector
from ..base.models import BandFrame
from ..base.state import StateManager
from ..config.settings import Settings
from ..core.interfaces import IDataService

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover
    sd = None  # type: ignore


class MicrophoneService(IDataService):
    """Streams microphone audio -> FFT -> BandFrames into StateManager."""

    def __init__(
        self, state: StateManager, settings: Settings, metrics: MetricsCollector
    ) -> None:
        self._state = state
        self._settings = settings
        self._metrics = metrics
        self._task: asyncio.Task | None = None
        self._stream = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=128)
        self._loop: asyncio.AbstractEventLoop

        fft_size = settings.mic_fft_size
        sample_rate = settings.mic_sample_rate

        # Pre-allocated rolling buffer (shifted in-place; no reallocation per chunk)
        self._buffer = np.zeros(fft_size, dtype=np.float32)
        # Pre-computed constants — calculated once, reused every frame
        self._freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        self._window = np.hanning(fft_size).astype(np.float32)
        self._window_sum = float(np.sum(self._window))
        # Band slice indices into the FFT output (avoids boolean mask per frame)
        self._band_slices: list[tuple[int, int]] = [
            (
                int(np.searchsorted(self._freqs, band.start)),
                int(np.searchsorted(self._freqs, band.end)),
            )
            for band in settings.bands
        ]

    # ------------------------------------------------------------------
    # IDataService
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. Run: pip install sounddevice"
            )

        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self._settings.mic_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._settings.mic_chunk_size,
            device=self._settings.mic_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """Called from sounddevice's audio thread; must not block."""
        mono = indata[:, 0].copy()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, mono)
        except Exception:
            pass  # drop frame if queue is full or loop is gone

    async def _run(self) -> None:
        s = self._settings
        fft_size = len(self._buffer)
        db_range = s.mic_db_max - s.mic_db_min

        while True:
            chunk: np.ndarray = await self._queue.get()
            cycle_start = time.perf_counter()

            self._metrics.record_input_queue_depth(self._queue.qsize())

            # Shift buffer in-place (no reallocation) then write new samples at the end
            n = len(chunk)
            self._buffer[:-n] = self._buffer[n:]
            self._buffer[-n:] = chunk

            ts = datetime.now(timezone.utc)

            # Windowed FFT magnitude (normalised by window sum → dBFS-correct)
            spectrum = np.abs(np.fft.rfft(self._buffer * self._window)) / self._window_sum

            # Convert to dB and normalise to [0, 100]
            db = 20.0 * np.log10(spectrum + 1e-9)
            normalized = np.clip(
                (db - s.mic_db_min) / db_range * 100.0, 0.0, 100.0
            ).astype(np.float32)

            for band, (si, ei) in zip(s.bands, self._band_slices):
                if si >= ei:
                    continue
                # Contiguous slice copy — no boolean mask overhead
                band_data = normalized[si:ei].copy()

                t0 = time.perf_counter()
                frame = BandFrame(
                    band_id=band.id,
                    band_start=band.start,
                    band_end=band.end,
                    timestamp=ts,
                    data=band_data,
                )
                self._state.push(band.id, frame)
                push_ms = (time.perf_counter() - t0) * 1000
                processing_ms = (time.perf_counter() - cycle_start) * 1000
                depth = self._state.band_depth(band.id)
                self._metrics.record_frame_generated(band.id, push_ms, processing_ms, depth, bins=len(band_data))
