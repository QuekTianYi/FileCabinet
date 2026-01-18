from abc import ABC, abstractmethod
from pathlib import Path

class FileHandler(ABC):
    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        pass

    @abstractmethod
    def extract(self, path: Path) -> str:
        pass
    