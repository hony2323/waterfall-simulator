import asyncio

from ..config.settings import Settings
from ..core.interfaces import IStateManager
from .models import BandFrame
from .ring_buffer import RingBuffer


class StateManager(IStateManager):
    def __init__(self, settings: Settings) -> None:
        self._default_capacity = settings.ring_buffer_size
        self._buffers: dict[str, RingBuffer[BandFrame]] = {
            band.id: RingBuffer(band.ring_buffer_size or self._default_capacity)
            for band in settings.bands
        }
        self.data_available: asyncio.Condition = asyncio.Condition()

    def push(self, band_id: str, frame: BandFrame) -> None:
        if band_id not in self._buffers:
            self._buffers[band_id] = RingBuffer(self._default_capacity)
        self._buffers[band_id].push(frame)
        # Notify any WS clients waiting in push mode
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._notify())
        except RuntimeError:
            pass  # not called from within a running event loop

    async def _notify(self) -> None:
        async with self.data_available:
            self.data_available.notify_all()

    def latest(self) -> dict[str, BandFrame | None]:
        """Returns the most recent frame for every known band."""
        return {band_id: buf.latest() for band_id, buf in self._buffers.items()}

    def get_band_history(self, band_id: str) -> list[BandFrame]:
        buf = self._buffers.get(band_id)
        return buf.all() if buf else []

    def band_depth(self, band_id: str) -> int:
        buf = self._buffers.get(band_id)
        return len(buf) if buf else 0
