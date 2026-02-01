from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

@dataclass
class ExtractResult:
    text: str
    sampled: list[str]

class FileHandler(ABC):
    def can_handle(self, path: Path) -> bool:
        raise NotImplementedError

    def extract(self, path: Path) -> ExtractResult:
        raise NotImplementedError
    