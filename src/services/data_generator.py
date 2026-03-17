import asyncio
import time
from datetime import datetime, timezone

import numpy as np

from ..base.metrics import MetricsCollector
from ..base.models import BandFrame
from ..base.state import StateManager
from ..config.settings import Settings
from ..core.interfaces import IDataService


class DataGeneratorService(IDataService):
    """Periodically generates a random float32 array (values 0-100) for every
    configured band and pushes it into the shared state."""

    def __init__(
        self, state: StateManager, settings: Settings, metrics: MetricsCollector
    ) -> None:
        self._state = state
        self._settings = settings
        self._metrics = metrics
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
            # All bands share the same timestamp per generation cycle
            ts = datetime.now(timezone.utc)

            for band in self._settings.bands:
                step = band.step if band.step is not None else self._settings.step
                size = max(1, round((band.end - band.start) / step))
                t0 = time.perf_counter()
                t = time.time()
                # Gradient baseline with slow brightness shift
                phase = np.sin(t * 0.8) * 15.0
                data = np.clip(
                    np.linspace(0.0, 100.0, size, dtype=np.float32) + phase,
                    0.0, 100.0,
                )
                # Fixed narrow spikes on top of the gradient
                for frac, width in [(0.15, 2), (0.42, 1), (0.71, 3)]:
                    idx = int(frac * size)
                    data[max(0, idx - width):idx + width + 1] = 100.0
                # Slowly drifting spike
                drift_idx = int((np.sin(t * 0.3) * 0.5 + 0.5) * (size - 10) + 5)
                data[max(0, drift_idx - 1):drift_idx + 2] = 100.0
                frame = BandFrame(
                    band_id=band.id,
                    band_start=band.start,
                    band_end=band.end,
                    timestamp=ts,
                    data=data,
                )
                self._state.push(band.id, frame)
                duration_ms = (time.perf_counter() - t0) * 1000
                depth = len(self._state.get_band_history(band.id))
                self._metrics.record_frame_generated(band.id, duration_ms, duration_ms, depth)

            await asyncio.sleep(interval)
