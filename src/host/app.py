import asyncio
import json
import struct
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..base.metrics import MetricsCollector
from ..base.models import BandFrame
from ..base.state import StateManager
from ..config.settings import Settings
from ..services.data_generator import DataGeneratorService


def _build_binary_message(frames: list[BandFrame], precision: str) -> bytes:
    """
    Binary frame layout:
      [uint32 header_len][JSON header bytes][band data bytes ...]

    Header is a JSON array, one entry per band:
      { band_id, band_start, band_end, timestamp, length, precision }

    Data section is all bands' raw bytes concatenated in the same order.
    Client reconstructs each band by slicing `length` samples of the known dtype.
    """
    header_entries = []
    data_parts = []

    for frame in frames:
        raw = frame.to_bytes(precision)
        header_entries.append({
            "band_id": frame.band_id,
            "band_start": frame.band_start,
            "band_end": frame.band_end,
            "timestamp": frame.timestamp.isoformat(),
            "length": len(frame.data),
            "precision": precision,
        })
        data_parts.append(raw)

    header_bytes = json.dumps(header_entries).encode()
    return struct.pack(">I", len(header_bytes)) + header_bytes + b"".join(data_parts)


def create_app() -> FastAPI:
    settings = Settings()
    state = StateManager(settings)
    metrics = MetricsCollector()
    generator = DataGeneratorService(state, settings, metrics)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await generator.start()
        yield
        await generator.stop()

    app = FastAPI(title="Waterfall Backend", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def get_metrics():
        return metrics.snapshot()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        metrics.record_ws_connected()
        interval = settings.ws_send_interval_ms / 1000.0

        try:
            while True:
                snapshot = state.latest()
                frames = [f for f in snapshot.values() if f is not None]

                message = _build_binary_message(frames, settings.precision)
                await websocket.send_bytes(message)

                now = datetime.now(timezone.utc)
                frame_age_ms = (
                    max((now - f.timestamp).total_seconds() * 1000 for f in frames)
                    if frames else 0.0
                )
                metrics.record_ws_send(len(message), frame_age_ms)

                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            metrics.record_ws_disconnected()

    return app
