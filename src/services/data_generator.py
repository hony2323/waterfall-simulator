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
                data = np.random.uniform(0.0, 100.0, size).astype(np.float32)
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
                self._metrics.record_frame_generated(band.id, duration_ms, depth)

            await asyncio.sleep(interval)
