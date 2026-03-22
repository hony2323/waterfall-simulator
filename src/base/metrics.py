from dataclasses import dataclass, field
from threading import Lock


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@dataclass
class BandMetrics:
    frames_generated: int = 0
    ring_buffer_depth: int = 0
    last_push_duration_ms: float = 0.0       # time for state.push() only
    last_processing_duration_ms: float = 0.0  # full FFT + normalize + push cycle
    last_frame_bins: int = 0                  # number of samples in the last frame


@dataclass
class WsMetrics:
    active_connections: int = 0
    total_sends: int = 0
    last_message_bytes: int = 0
    avg_message_bytes: float = 0.0
    last_frame_age_ms: float = 0.0
    last_build_duration_ms: float = 0.0   # time to build binary message
    last_send_duration_ms: float = 0.0    # time for websocket.send_bytes()


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._bands: dict[str, BandMetrics] = {}
        self._ws = WsMetrics()
        self._input_queue_depth: int = 0

    # --- generation side ---

    def record_frame_generated(
        self,
        band_id: str,
        push_duration_ms: float,
        processing_duration_ms: float,
        ring_depth: int,
        bins: int = 0,
    ) -> None:
        with self._lock:
            m = self._bands.setdefault(band_id, BandMetrics())
            m.frames_generated += 1
            m.last_push_duration_ms = push_duration_ms
            m.last_processing_duration_ms = processing_duration_ms
            m.ring_buffer_depth = ring_depth
            if bins:
                m.last_frame_bins = bins

    def record_input_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._input_queue_depth = depth

    # --- websocket side ---

    def record_ws_connected(self) -> None:
        with self._lock:
            self._ws.active_connections += 1

    def record_ws_disconnected(self) -> None:
        with self._lock:
            self._ws.active_connections = max(0, self._ws.active_connections - 1)

    def record_ws_send(
        self,
        byte_count: int,
        frame_age_ms: float,
        build_duration_ms: float,
        send_duration_ms: float,
    ) -> None:
        with self._lock:
            self._ws.total_sends += 1
            self._ws.last_message_bytes = byte_count
            n = self._ws.total_sends
            self._ws.avg_message_bytes += (byte_count - self._ws.avg_message_bytes) / n
            self._ws.last_frame_age_ms = frame_age_ms
            self._ws.last_build_duration_ms = build_duration_ms
            self._ws.last_send_duration_ms = send_duration_ms

    # --- snapshot ---

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "websocket": {
                    "active_connections": self._ws.active_connections,
                    "total_sends": self._ws.total_sends,
                    "last_message_bytes": self._ws.last_message_bytes,
                    "last_message_size": _human_bytes(self._ws.last_message_bytes),
                    "avg_message_bytes": round(self._ws.avg_message_bytes, 1),
                    "avg_message_size": _human_bytes(self._ws.avg_message_bytes),
                    "last_frame_age_ms": round(self._ws.last_frame_age_ms, 2),
                    "last_build_duration_ms": round(self._ws.last_build_duration_ms, 3),
                    "last_send_duration_ms": round(self._ws.last_send_duration_ms, 3),
                },
                "input_queue_depth": self._input_queue_depth,
                "bands": {
                    band_id: {
                        "frames_generated": m.frames_generated,
                        "ring_buffer_depth": m.ring_buffer_depth,
                        "last_push_duration_ms": round(m.last_push_duration_ms, 3),
                        "last_processing_duration_ms": round(
                            m.last_processing_duration_ms, 3
                        ),
                        "last_frame_bins": m.last_frame_bins,
                    }
                    for band_id, m in self._bands.items()
                },
            }
