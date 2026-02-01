from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from pipeline.format import detect_file_format

@dataclass(frozen=True)
class WorkItem:
    item_type: str                      # "folder" | "zip" | "file"
    path: Path
    raw_text_fn: Callable[[], str]       # produces raw_text
    fallback_text_fn: Callable[[], str]  # produces fallback string
    extra_fields_fn: Callable[[], dict]  # produces extra record fields (sampled_files, etc.)
    covered_paths_fn: Callable[[], list[str]]  # paths covered by this item (folder children)
