import asyncio
import json
import struct
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..base.metrics import MetricsCollector
from ..base.models import BandFrame
from ..base.state import StateManager
from ..config.settings import Settings
from ..core.interfaces import IDataService
from ..services.data_generator import DataGeneratorService
from ..services.microphone_service import MicrophoneService
from ..services.realistic_generator import RealisticGeneratorService


def _build_binary_message(frames: list[BandFrame], precision: str) -> bytes:
    """
    Binary frame layout:
      [uint32 header_len][JSON header bytes][band data bytes ...]

    Header is a JSON array, one entry per band:
      { band_id, band_start, band_end, timestamp, sent_at, length, precision }

    `sent_at` is the Unix epoch in milliseconds recorded just before the message
    is serialised; the client subtracts it from its receive time to measure
    end-to-end delivery latency.

    Data section is all bands' raw bytes concatenated in the same order.
    Client reconstructs each band by slicing `length` samples of the known dtype.
    """
    sent_at_ms = datetime.now(timezone.utc).timestamp() * 1000

    header_entries = []
    data_parts = []

    for frame in frames:
        raw = frame.to_bytes(precision)
        header_entries.append({
            "band_id": frame.band_id,
            "band_start": frame.band_start,
            "band_end": frame.band_end,
            "timestamp": frame.timestamp.isoformat(),
            "sent_at": sent_at_ms,
            "length": len(raw),
            "precision": precision,
        })
        data_parts.append(raw)

    header_bytes = json.dumps(header_entries).encode()
    return struct.pack(">I", len(header_bytes)) + header_bytes + b"".join(data_parts)


def create_app() -> FastAPI:
    settings = Settings()
    state = StateManager(settings)
    metrics = MetricsCollector()

    service: IDataService
    if settings.data_source == "microphone":
        service = MicrophoneService(state, settings, metrics)
    elif settings.data_source == "generator_simple":
        service = DataGeneratorService(state, settings, metrics)
    else:
        service = RealisticGeneratorService(state, settings, metrics)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="Waterfall Backend", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def get_metrics():
        return metrics.snapshot()

    async def _send_snapshot(websocket: WebSocket) -> None:
        snapshot = state.latest()
        frames = [f for f in snapshot.values() if f is not None]

        t_build = time.perf_counter()
        message = _build_binary_message(frames, settings.precision)
        build_ms = (time.perf_counter() - t_build) * 1000

        t_send = time.perf_counter()
        await websocket.send_bytes(message)
        send_ms = (time.perf_counter() - t_send) * 1000

        now = datetime.now(timezone.utc)
        frame_age_ms = (
            max((now - f.timestamp).total_seconds() * 1000 for f in frames)
            if frames else 0.0
        )
        metrics.record_ws_send(len(message), frame_age_ms, build_ms, send_ms)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        metrics.record_ws_connected()

        try:
            if settings.ws_mode == "push":
                while True:
                    async with state.data_available:
                        await state.data_available.wait()
                    await _send_snapshot(websocket)
            else:
                interval = settings.ws_send_interval_ms / 1000.0
                while True:
                    await _send_snapshot(websocket)
                    await asyncio.sleep(interval)
        except WebSocketDisconnect:
            metrics.record_ws_disconnected()

    return app
