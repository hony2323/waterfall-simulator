from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class BandConfig(BaseModel):
    id: str
    start: float        # Hz or logical band start
    end: float          # Hz or logical band end
    step: float | None = None        # Hz per sample; overrides global if set
    ring_buffer_size: int | None = None  # overrides global if set


class Settings(BaseSettings):
    step: float = 1.0                # Hz per sample (array size = (end - start) / step)
    ring_buffer_size: int = 100      # frames kept per band
    generation_interval_ms: int = 500
    ws_send_interval_ms: int = 100
    # uint8 → 1 B/sample (integer 0-100)
    # uint16 → 2 B/sample (scaled ×100, 2 decimal places)
    # float32 → 4 B/sample (full precision)
    precision: Literal["uint8", "uint16", "float32"] = "uint8"

    # Multiple bands supported; override via BANDS env var as JSON
    bands: list[BandConfig] = [
        BandConfig(id="band_0", start=0.0, end=1000.0),
    ]

    # Data source: "generator" uses synthetic data; "microphone" uses live audio
    data_source: Literal["generator", "microphone"] = "generator"

    # Microphone settings (used when data_source="microphone")
    mic_sample_rate: int = 44100     # Hz
    mic_chunk_size: int = 2048       # samples per audio callback
    mic_fft_size: int = 4096         # FFT window size (power of 2)
    mic_device: int | str | None = None  # None = system default
    # dB range mapped to [0, 100]: signals below mic_db_min → 0, at mic_db_max → 100
    mic_db_min: float = -60.0
    mic_db_max: float = 0.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
