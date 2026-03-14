import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..base.state import StateManager
from ..config.settings import Settings
from ..services.data_generator import DataGeneratorService


def create_app() -> FastAPI:
    settings = Settings()
    state = StateManager(settings)
    generator = DataGeneratorService(state, settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await generator.start()
        yield
        await generator.stop()

    app = FastAPI(title="Waterfall Backend", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        interval = settings.ws_send_interval_ms / 1000.0

        try:
            while True:
                snapshot = state.latest()

                # Build payload: one entry per band that has data.
                # Multiple bands per timestamp are fully supported.
                payload = {
                    band_id: frame.to_payload()
                    for band_id, frame in snapshot.items()
                    if frame is not None
                }

                await websocket.send_json(payload)
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            pass

    return app
