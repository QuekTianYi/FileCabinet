from pathlib import Path
from handlers.base import ExtractResult, FileHandler
from handlers.doc_handler import DocHandler
from handlers.utils import (
    MAX_FILES_PER_CONTAINER,
    MAX_TOTAL_TEXT_CHARS,
)

def filename_fallback(p: Path) -> str:
    return p.name.replace("_", " ").replace("-", " ")

class FolderHandler(FileHandler):
    def __init__(self, file_handlers: list[FileHandler]):
        self.file_handlers = file_handlers

    def can_handle(self, path: Path) -> bool:
        return path.is_dir()

    def _is_safe_path(self, p: Path) -> bool:
        # Skip hidden dirs/files like .git, .venv, etc.
        return not any(part.startswith(".") for part in p.parts)

    def _iter_files(self, folder: Path):
        for fp in folder.rglob("*"):
            if not self._is_safe_path(fp):
                continue
            if fp.is_file():
                yield fp

    def _extract_file_text(self, fp: Path) -> str:
        for h in self.file_handlers:
            if h.can_handle(fp):
                res = h.extract(fp)
                return res.text
        return ""

    def extract(self, folder: Path) -> ExtractResult:
        pieces = [f"folder_name: {folder.name.replace('_',' ').replace('-',' ')}"]
        sampled: list[str] = []
        total_chars = 0
        taken = 0

        for fp in self._iter_files(folder):
            if taken >= MAX_FILES_PER_CONTAINER or total_chars >= MAX_TOTAL_TEXT_CHARS:
                break

            text = self._extract_file_text(fp)
            if not text.strip():
                text = filename_fallback(fp)

            chunk = f" file: {fp.name} content: {text}"
            pieces.append(chunk)
            sampled.append(str(fp))

            total_chars += len(chunk)
            taken += 1

        agg = " | ".join(pieces)[:MAX_TOTAL_TEXT_CHARS]
        return ExtractResult(text=agg, sampled=sampled)