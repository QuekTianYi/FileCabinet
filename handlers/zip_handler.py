import zipfile
from pathlib import Path
from handlers.base import FileHandler
from handlers.utils import (
    MAX_FILES_PER_CONTAINER,
    MAX_TOTAL_TEXT_CHARS,
    MAX_TEXT_CHARS_PER_ITEM,
)

class ZipHandler(FileHandler):

    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".zip"

    def extract(self, path: Path) -> str:
        pieces = [f"zip_name: {path.stem.replace('_',' ').replace('-',' ')}"]
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

                    total_chars += len(chunk)
                    taken += 1

                    if taken >= MAX_FILES_PER_CONTAINER or total_chars >= MAX_TOTAL_TEXT_CHARS:
                        break
        except Exception:
            return f"zip_name: {path.stem}"

        return " | ".join(pieces)[:MAX_TOTAL_TEXT_CHARS]