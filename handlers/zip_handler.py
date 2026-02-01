from __future__ import annotations

import zipfile
from pathlib import Path

from handlers.base import ExtractResult, FileHandler
from handlers.utils import MAX_FILES_PER_CONTAINER, MAX_TOTAL_TEXT_CHARS


class ZipHandler(FileHandler):
    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".zip"

    def extract(self, path: Path) -> ExtractResult:
        pieces = [f"zip_name: {path.stem.replace('_',' ').replace('-',' ')}"]
        sampled: list[str] = []
        total_chars = 0
        taken = 0

        try:
            with zipfile.ZipFile(path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue

                    inner_name = Path(name).name
                    chunk = f" inner_file: {inner_name}"
                    pieces.append(chunk)
                    sampled.append(name)

                    total_chars += len(chunk)
                    taken += 1

                    if taken >= MAX_FILES_PER_CONTAINER or total_chars >= MAX_TOTAL_TEXT_CHARS:
                        break

        except Exception:
            return ExtractResult(text=f"zip_name: {path.stem}", sampled=[])

        text = " | ".join(pieces)[:MAX_TOTAL_TEXT_CHARS]
        return ExtractResult(text=text, sampled=sampled)
