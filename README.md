
## Packages:
core is for interfaces
base is for inside things that have some logic
services is for things that run in the background
host is for creating the app
config is for config

The app is exposed through src.__init__ and started in main

---

## `/metrics` response

```json
{
  "websocket": {
    "active_connections": 1,
    "total_sends": 4557,
    "last_message_bytes": 7612,
    "last_message_size": "7.4 KB",
    "avg_message_bytes": 7611.2,
    "avg_message_size": "7.4 KB",
    "last_frame_age_ms": 30.89,
    "last_build_duration_ms": 0.205,
    "last_send_duration_ms": 0.927
  },
  "input_queue_depth": 0,
  "bands": {
    "band_0": {
      "frames_generated": 10714,
      "ring_buffer_depth": 100,
      "last_push_duration_ms": 0.024,
      "last_processing_duration_ms": 0.224
    }
  }
}
```

| Field | Description |
|---|---|
| `last_frame_age_ms` | How old the latest frame was when the WS message was sent |
| `last_build_duration_ms` | Time to serialise the binary message |
| `last_send_duration_ms` | Time for `websocket.send_bytes()` |
| `input_queue_depth` | Microphone audio queue backlog — should stay at 0 |
| `last_push_duration_ms` | Time for `state.push()` only |
| `last_processing_duration_ms` | Full FFT + normalise + push cycle time |

---

## WebSocket binary frame — `length` field

The `length` field in each band header is the **byte length** of that band's data section, not the sample count. Clients must divide by the element size to get the sample count:

| `precision` | element size | sample count |
|---|---|---|
| `uint8` | 1 B | `length` |
| `uint16` | 2 B | `length / 2` |
| `float32` | 4 B | `length / 4` |

---

## Key `.env` defaults

```
PRECISION=float32        # uint8 | uint16 | float32
DATA_SOURCE=microphone   # generator | microphone
BANDS='[{"id":"band_0","start":20,"end":20000}]'
```
