from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class BandFrame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    band_id: str
    band_start: float
    band_end: float
    timestamp: datetime
    # Stored as numpy array internally for efficiency; serialized on send
    data: Any  # np.ndarray[float32]

    def to_payload(self) -> dict:
        return {
            "band_id": self.band_id,
            "band_start": self.band_start,
            "band_end": self.band_end,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data.tolist(),
        }

    def to_bytes(self, precision: str) -> bytes:
        if precision == "uint8":
            # scale [0,100] → [0,255] to use full uint8 range
            return (self.data * 2.55).astype(np.uint8).tobytes()
        if precision == "uint16":
            # scale [0,100] → [0,65535] to use full uint16 range
            return (self.data * 655.35).astype(np.uint16).tobytes()
        return self.data.astype(np.float32).tobytes()  # float32
