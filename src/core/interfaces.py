from abc import ABC, abstractmethod


class IStateManager(ABC):
    @abstractmethod
    def push(self, band_id: str, frame) -> None: ...

    @abstractmethod
    def latest(self) -> dict: ...

    @abstractmethod
    def get_band_history(self, band_id: str) -> list: ...


class IDataService(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...
