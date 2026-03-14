from collections import deque
from threading import Lock
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    def __init__(self, capacity: int) -> None:
        self._buffer: deque[T] = deque(maxlen=capacity)
        self._lock = Lock()

    def push(self, item: T) -> None:
        with self._lock:
            self._buffer.append(item)

    def latest(self) -> Optional[T]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def all(self) -> list[T]:
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
