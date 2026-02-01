from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable

from pypdf import PdfReader
import docx

from handlers.base import ExtractResult, FileHandler


# ============================================================
# Config
# ============================================================

MAX_CHARS = 1500  # cap to keep it fast + consistent


class DocHandler(FileHandler):
    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in {
            ".pdf", ".docx", ".txt", ".md", ".csv", ".log"
        }

    def extract(self, path: Path) -> ExtractResult:
        text = file_to_text(path)
        # sampled should represent what you actually read
        return ExtractResult(text=text, sampled=[str(path)])


# ============================================================
# Helper functions
# ============================================================

def safe_date(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


def extract_pdf_text(path: Path, max_chars: int = MAX_CHARS) -> str:
    try:
        reader = PdfReader(str(path))
        out: list[str] = []
        for page in reader.pages[:2]:  # first 2 pages only
            t = page.extract_text() or ""
            out.append(t)
            if sum(len(x) for x in out) >= max_chars:
                break
        return "\n".join(out)[:max_chars]
    except Exception:
        return ""


def extract_docx_text(path: Path, max_chars: int = MAX_CHARS) -> str:
    try:
        d = docx.Document(str(path))
        text = "\n".join(p.text for p in d.paragraphs if p.text)
        return text[:max_chars]
    except Exception:
        return ""


def extract_txt_text(path: Path, max_chars: int = MAX_CHARS) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def file_to_text(p: Path) -> str:
    # ---- metadata ----
    ext = p.suffix.lower().lstrip(".")
    stat = p.stat()
    size_kb = int(stat.st_size / 1024)
    mdate = safe_date(stat.st_mtime)

    # include a bit of folder context (last 2 folder names)
    parents = [x for x in p.parts[-3:-1]]
    parent_hint = " ".join(parents)

    name = p.stem.replace("_", " ").replace("-", " ")

    meta = (
        f"filename: {name} | ext: {ext} | size_kb: {size_kb} | "
        f"modified: {mdate} | parent: {parent_hint}"
    )

    # ---- content (type-based) ----
    content = ""
    if ext == "pdf":
        content = extract_pdf_text(p)
    elif ext == "docx":
        content = extract_docx_text(p)
    elif ext in {"txt", "md", "csv", "log"}:
        content = extract_txt_text(p)

    if content.strip():
        return meta + " | content: " + content

    return meta
