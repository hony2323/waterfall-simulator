from pydantic import BaseModel
from pydantic_settings import BaseSettings


class BandConfig(BaseModel):
    id: str
    start: float        # Hz or logical band start
    end: float          # Hz or logical band end
    ring_buffer_size: int | None = None  # overrides global if set


class Settings(BaseSettings):
    array_size: int = 65536          # number of samples per frame (64k default)
    ring_buffer_size: int = 100      # frames kept per band
    generation_interval_ms: int = 500
    ws_send_interval_ms: int = 100

    # Multiple bands supported; override via BANDS env var as JSON
    bands: list[BandConfig] = [
        BandConfig(id="band_0", start=0.0, end=1000.0),
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
