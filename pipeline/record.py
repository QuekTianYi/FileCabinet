# pipeline/records.py
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, List, Optional

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@dataclass
class SortRecord:
    # --- identity ---
    type: str                  # "file" | "folder" | "zip"
    path: str

    # --- proposal ---
    propose_move_to_category: str
    decision: str              # "AUTO_SUGGEST" | "REVIEW"
    best_score: float
    margin: float

    # --- optional diagnostics ---
    top5: Optional[List[Any]] = None
    format: Optional[str] = None
    mime: Optional[str] = None
    format_confidence: Optional[float] = None
    format_source: Optional[str] = None

    # --- container-specific ---
    sampled_files: Optional[List[str]] = None
    sampled_inner_files: Optional[List[str]] = None

    def to_json(self) -> str:
        """
        Serialize safely to JSON string (for JSONL).
        """
        return json_dumps_clean(asdict(self))


def json_dumps_clean(obj: dict) -> str:
    """
    Removes None values before dumping.
    Keeps logs clean and compact.
    """
    import json
    return json.dumps(
        {k: v for k, v in obj.items() if v is not None},
        ensure_ascii=False
    )
