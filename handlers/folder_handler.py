from pathlib import Path
from handlers.base import FileHandler
from handlers.doc_handler import DocHandler
from handlers.utils import (
    MAX_FILES_PER_CONTAINER,
    MAX_TOTAL_TEXT_CHARS,
)

class FolderHandler(FileHandler):
    def __init__(self, file_handlers: list[FileHandler]):
        self.file_handlers = file_handlers

    def can_handle(self, path: Path) -> bool:
        return path.is_dir()

    def extract(self, path: Path) -> str:
        pieces = [f"folder_name: {path.name.replace('_',' ').replace('-',' ')}"]
        total_chars = 0
        taken = 0

        for fp in path.rglob("*"):
            if not fp.is_file():
                continue

            for handler in self.file_handlers:
                if handler.can_handle(fp):
                    text = handler.extract(fp) or ""
                    if not text.strip():
                        text = fp.name

                    chunk = f" file: {fp.name} content: {text}"
                    pieces.append(chunk)

                    total_chars += len(chunk)
                    taken += 1
                    break

            if taken >= MAX_FILES_PER_CONTAINER or total_chars >= MAX_TOTAL_TEXT_CHARS:
                break

        return " | ".join(pieces)[:MAX_TOTAL_TEXT_CHARS]