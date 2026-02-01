import json
import zipfile
from pathlib import Path
from time import time
import numpy as np
from sentence_transformers import SentenceTransformer
from handlers.base import ExtractResult
from handlers.doc_handler import DocHandler
from handlers.folder_handler import FolderHandler
from handlers.utils import MAX_FILES_PER_CONTAINER, MAX_TEXT_CHARS_PER_ITEM, MAX_TOTAL_TEXT_CHARS, ZIP_EXTS
from handlers.zip_handler import ZipHandler
from datetime import datetime, timezone
import uuid

from pipeline.file_format import FileFormat
from pipeline.format import detect_file_format
from pipeline.record import SortRecord
from pipeline.work_item import WorkItem

# -----------------------------
# Globals / setup
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

doc_handler = DocHandler()

handlers = [
    # TODO: add ImageHandler later (CLIP etc.)
    FolderHandler(file_handlers=[doc_handler]),
    ZipHandler(),
    doc_handler,
]

# -----------------------------
# Embedding + scoring
# -----------------------------
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def embed_texts(model, texts, batch_size=64):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

def make_folder_centroids(model, folder_to_prototypes: dict[str, list[str]]):
    folder_centroids = {}
    for folder, protos in folder_to_prototypes.items():
        vecs = embed_texts(model, protos)
        centroid = np.mean(vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        folder_centroids[folder] = centroid
    return folder_centroids

def propose_destination(model, file_text, folder_centroids):
    fvec = model.encode([file_text], normalize_embeddings=True)[0]
    scored = [(folder, cosine(fvec, cvec)) for folder, cvec in folder_centroids.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_folder, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else -1.0
    margin = best_score - second_score
    return best_folder, best_score, second_score, margin, scored[:5]

# -----------------------------
# Helpers
# -----------------------------

def is_safe_path(p: Path) -> bool:
    # Skip hidden dirs/files like .git, .venv, etc.
    return not any(part.startswith(".") for part in p.parts)

def file_to_text(p: Path) -> str:
    # fallback: filename only
    return p.name.replace("_", " ").replace("-", " ")

def process_file_content(path: Path) -> str:
    """
    Try handlers to extract file content.
    Falls back to filename if no handler matches.
    """
    for handler in handlers:
        if handler.can_handle(path):
            text = handler.extract(path) or ""
            return text[:MAX_TEXT_CHARS_PER_ITEM]
    return ""

def iter_files_in_folder(folder: Path):
    """
    Yield files inside folder (recursive), safely.
    """
    for fp in folder.rglob("*"):
        if not is_safe_path(fp):
            continue
        if fp.is_file():
            yield fp
            
def extract_text_result(path: Path) -> ExtractResult:
    for h in handlers:
        if h.can_handle(path):
            return h.extract(path)  # MUST return ExtractResult(text, sampled)
    return ExtractResult(text="", sampled=[])

def detect_format_for_path(path: Path):
    """
    Always returns a dict with keys:
    format, mime, confidence, source
    Never raises even if detector fails.
    """
    if path.is_dir():
        return {
            "format": "FOLDER",
            "mime": None,
            "confidence": 1.0,
            "source": "container",
        }

    try:
        fr = detect_file_format(path)
    except Exception as e:
        fr = None

    # If detector returns None (or failed), fall back to extension-based guess
    if fr is None:
        ext = path.suffix.lower()

        # super simple fallback buckets (adjust as your enum grows)
        if ext in {".zip", ".rar", ".7z", ".tar", ".gz"}:
            fmt = "ARCHIVE"
        elif ext in {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".txt", "._toggle"}:
            fmt = "DOCUMENT"
        elif ext in {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".html", ".css", ".json", ".yml", ".yaml"}:
            fmt = "CODE"
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            fmt = "IMAGE"
        elif ext in {".exe", ".msi"}:
            fmt = "INSTALLER"
        else:
            fmt = "UNKNOWN"

        return {
            "format": fmt,
            "mime": None,
            "confidence": 0.3,   # low confidence fallback
            "source": "extension_fallback",
        }

    # Normal case (FormatDetector succeeded)
    return {
        "format": fr.format.name,
        "mime": getattr(fr, "mime", None),
        "confidence": getattr(fr, "confidence", 0.7),
        "source": getattr(fr, "source", "detector"),
    }


def inject_format_hint(text: str, fmt_name: str) -> str:
    """
    Simple but effective: prepend a stable hint so embeddings learn the context.
    This alone often improves accuracy (e.g., documents stop being mistaken as software).
    """
    return f"format: {fmt_name}. {text}"

def decide(best_score: float, margin: float, auto_th: float, margin_th: float) -> str:
    return "AUTO_SUGGEST" if (best_score >= auto_th and margin >= margin_th) else "REVIEW"

def prepare_text(raw_text: str, fallback: str, fmt: dict) -> str:
    """
    - ensure non-empty text
    - inject format hint
    """
    if not (raw_text or "").strip():
        raw_text = fallback
    return inject_format_hint(raw_text, fmt.get("format", "UNKNOWN"))

def propose_item(
    *,
    item_type: str,
    path: Path,
    raw_text: str,
    fallback_text: str,
    folder_centroids: dict,
    auto_th: float,
    margin_th: float,
    extra_record_fields: dict | None = None,
):
    """
    Returns: record_dict, best_folder (for printing)
    """
    fmt = detect_format_for_path(path)
    text = prepare_text(raw_text, fallback_text, fmt)

    best_folder, best_score, second_score, margin, top5 = propose_destination(
        model, text, folder_centroids
    )

    decision_str = decide(best_score, margin, auto_th, margin_th)

    record = {
        "type": item_type,
        "path": str(path),
        "propose_move_to_category": best_folder,
        "best_score": best_score,
        "margin": margin,
        "decision": decision_str,
        "top5": top5,
        # keep format info consistent everywhere (optional but recommended)
        "format": fmt,
    }

    if extra_record_fields:
        record.update(extra_record_fields)

    return record, best_folder, decision_str

def process_work_item(
    wi: WorkItem,
    *,
    folder_centroids: dict,
    auto_th: float,
    margin_th: float,
):
    raw_text = wi.raw_text_fn()
    fallback = wi.fallback_text_fn()
    extra = wi.extra_fields_fn()

    record, best_folder, decision = propose_item(
        item_type=wi.item_type,
        path=wi.path,
        raw_text=raw_text,
        fallback_text=fallback,
        folder_centroids=folder_centroids,
        auto_th=auto_th,
        margin_th=margin_th,
        extra_record_fields=extra,
    )
    covered = wi.covered_paths_fn()
    return record, best_folder, covered

def build_work_items(
    WATCH_DIR: Path,
    top_level_folders: list[Path],
    zips: list[Path],
    loose_files: list[Path],
):
    items: list[WorkItem] = []

    # folders
    for folder in sorted(top_level_folders):
        res = extract_text_result(folder)
        items.append(
            WorkItem(
                item_type="folder",
                path=folder,
                raw_text_fn=lambda r=res: r.text,
                fallback_text_fn=lambda f=folder: f"folder_name: {f.name}",
                extra_fields_fn=lambda r=res: {"sampled_files": r.sampled},
                covered_paths_fn=lambda f=folder: [str(fp) for fp in iter_files_in_folder(f)],
            )
        )

    # zips
    for zp in sorted(zips):
        res = extract_text_result(zp)
        items.append(
            WorkItem(
                item_type="zip",
                path=zp,
                raw_text_fn=lambda r=res: r.text,
                fallback_text_fn=lambda z=zp: f"zip_name: {z.stem}",
                extra_fields_fn=lambda r=res: {"sampled_inner_files": r.sampled},
                covered_paths_fn=lambda: [],
            )
        )

    # files
    for fp in sorted(loose_files):
        res = extract_text_result(fp)
        items.append(
            WorkItem(
                item_type="file",
                path=fp,
                raw_text_fn=lambda r=res: r.text,
                fallback_text_fn=lambda p=fp: file_to_text(p),
                extra_fields_fn=lambda: {},
                covered_paths_fn=lambda: [],
            )
        )

    return items

# -----------------------------
# Main runner
# -----------------------------
def run(
    WATCH_DIR: Path,
    LOG_PATH: Path,
    AUTO_THRESHOLD: float,
    MARGIN_THRESHOLD: float,
    FOLDER_TO_PROTOTYPE: dict,
):
    start_time = time()
    count = 0

    print("Building folder centroids...")
    folder_centroids = make_folder_centroids(model, FOLDER_TO_PROTOTYPE)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Starting scan (top-level folders + loose files + zip)...")

    covered_files = set()

    # Only top-level items directly inside WATCH_DIR
    top_level_folders = []
    zips = []
    loose_files = []

    for p in WATCH_DIR.iterdir():
        if not is_safe_path(p):
            continue

        if p.is_dir():
            top_level_folders.append(p)
        elif p.is_file() and p.suffix.lower() in ZIP_EXTS:
            zips.append(p)
        elif p.is_file():
            loose_files.append(p)

    run_id = uuid.uuid4().hex[:10] # short run id like "a1b2c3d4e5"
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    items = build_work_items(WATCH_DIR, top_level_folders, zips, loose_files)

    covered_files = set()

    with LOG_PATH.open("w", encoding="utf-8") as logf:
        logf.write(json.dumps({
            "type": "run_header",
            "run_id": run_id,
            "started_at_utc": started_at,
            "watch_dir": str(WATCH_DIR),
        }, ensure_ascii=False) + "\n")

        for wi in items:
            # skip file if already covered by a folder item
            if wi.item_type == "file" and str(wi.path) in covered_files:
                continue

            record, best_folder, newly_covered = process_work_item(
                wi,
                folder_centroids=folder_centroids,
                auto_th=AUTO_THRESHOLD,
                margin_th=MARGIN_THRESHOLD,
            )
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")

            for c in newly_covered:
                covered_files.add(c)

            count += 1
            if count % 25 == 0:
                elapsed = time() - start_time
                print(f"[{count}] items processed ({elapsed:.1f}s) latest: {wi.path.name} -> {record['propose_move_to_category']}")

    print(f"Done. {count} item(s) processed. run_id={run_id}")