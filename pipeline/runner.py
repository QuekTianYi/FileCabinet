import json
import zipfile
from pathlib import Path
from time import time
import numpy as np
from sentence_transformers import SentenceTransformer
from handlers.doc_handler import DocHandler
from handlers.folder_handler import FolderHandler
from handlers.utils import MAX_FILES_PER_CONTAINER, MAX_TEXT_CHARS_PER_ITEM, MAX_TOTAL_TEXT_CHARS, ZIP_EXTS
from handlers.zip_handler import ZipHandler
from datetime import datetime, timezone
import uuid

from pipeline.file_format import FileFormat
from pipeline.format import detect_file_format
from pipeline.record import SortRecord

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

def aggregate_folder_text(folder: Path) -> tuple[str, list[str]]:
    """
    Build a single text blob representing the folder based on:
    - folder name
    - content of files inside (sampled)
    Returns: (text, sampled_file_paths)
    """
    pieces = [f"folder_name: {folder.name.replace('_',' ').replace('-',' ')}"]
    sampled = []
    total_chars = 0
    taken = 0

    for fp in iter_files_in_folder(folder):
        # Stop if we already sampled enough
        if taken >= MAX_FILES_PER_CONTAINER or total_chars >= MAX_TOTAL_TEXT_CHARS:
            break

        text = process_file_content(fp)
        if not text.strip():
            # even if we can't extract content, filename is still helpful
            text = file_to_text(fp)

        # add a small header so names don't blur together
        chunk = f" file: {fp.name} content: {text}"
        pieces.append(chunk)
        sampled.append(str(fp))

        total_chars += len(chunk)
        taken += 1

    agg = " | ".join(pieces)
    return agg[:MAX_TOTAL_TEXT_CHARS], sampled

def aggregate_zip_text(zip_path: Path) -> tuple[str, list[str]]:
    """
    Build a single text blob representing the zip based on:
    - zip filename
    - names of files inside
    - (optional) extracted text from text-like files inside zip (sampled)
    Returns: (text, sampled_inner_names)
    """
    pieces = [f"zip_name: {zip_path.stem.replace('_',' ').replace('-',' ')}"]
    sampled = []
    total_chars = 0
    taken = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # list members (skip directories)
            members = [m for m in zf.namelist() if not m.endswith("/")]
            # light ordering heuristic: smaller names first (often more relevant),
            # but keep it simple
            for m in members:
                if taken >= MAX_FILES_PER_CONTAINER or total_chars >= MAX_TOTAL_TEXT_CHARS:
                    break

                # always include inner filename
                inner_name = Path(m).name
                chunk_parts = [f" inner_file: {inner_name}"]

                # Optionally extract content from obviously text-like files inside zip
                ext = Path(m).suffix.lower()
                if ext in {".txt", ".md", ".csv", ".log"}:
                    try:
                        raw = zf.read(m)
                        txt = raw.decode("utf-8", errors="ignore")[:MAX_TEXT_CHARS_PER_ITEM]
                        if txt.strip():
                            chunk_parts.append(f" content: {txt}")
                    except Exception:
                        pass

                chunk = " ".join(chunk_parts)
                pieces.append(chunk)
                sampled.append(m)

                total_chars += len(chunk)
                taken += 1

    except Exception:
        # If zip is invalid or unreadable, fallback to name only
        return f"zip_name: {zip_path.stem}", []

    agg = " | ".join(pieces)
    return agg[:MAX_TOTAL_TEXT_CHARS], sampled

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
    with LOG_PATH.open("w", encoding="utf-8") as logf:

        header = {
            "type": "run_header",
            "run_id": run_id,
            "started_at_utc": started_at,
            "watch_dir": str(WATCH_DIR),
        }
        logf.write(json.dumps(header, ensure_ascii=False) + "\n")

        # ---- Folder proposals (TOP-LEVEL only; move whole folder)
        for folder in sorted(top_level_folders):
            raw_text, sampled = aggregate_folder_text(folder)
            if not raw_text.strip():
                raw_text = f"folder_name: {folder.name}"

            fmt = detect_format_for_path(folder)  # FOLDER
            text = inject_format_hint(raw_text, fmt["format"])

            best_folder, best_score, second_score, margin, top5 = propose_destination(
                model, text, folder_centroids
            )

            decision = "REVIEW"
            if best_score >= AUTO_THRESHOLD and margin >= MARGIN_THRESHOLD:
                decision = "AUTO_SUGGEST"
            
            # TODO: Change to use SortRecord
            record = {
                "type": "folder",
                "path": str(folder),
                "propose_move_to_category": best_folder,
                "best_score": best_score,
                "margin": margin,
                "decision": decision,
                "sampled_files": sampled,
                "top5": top5,
            }

            # mark contained files as covered so we don't also propose them individually
            for fp in iter_files_in_folder(folder):
                covered_files.add(str(fp))

            count += 1
            if count % 10 == 0:
                elapsed = time() - start_time
                print(f"[{count}] items processed ({elapsed:.1f}s) latest folder: {folder.name} -> {best_folder}")

        # ---- Zip proposals (move whole zip) - top-level zips
        for zp in sorted(zips):
            text, sampled_inner = aggregate_zip_text(zp)
            if not text.strip():
                text = f"zip_name: {zp.stem}"

            best_folder, best_score, second_score, margin, top5 = propose_destination(
                model, text, folder_centroids
            )

            decision = "REVIEW"
            if best_score >= AUTO_THRESHOLD and margin >= MARGIN_THRESHOLD:
                decision = "AUTO_SUGGEST"

            # TODO: Change to use SortRecord
            record = {
                "type": "zip",
                "path": str(zp),
                "propose_move_to_category": best_folder,
                "best_score": best_score,
                "margin": margin,
                "decision": decision,
                "sampled_inner_files": sampled_inner,
                "top5": top5,
            }
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1
            if count % 25 == 0:
                elapsed = time() - start_time
                print(f"[{count}] items processed ({elapsed:.1f}s) latest zip: {zp.name} -> {best_folder}")

        # ---- File proposals (move file) - only for top-level loose files (not covered)
        for fp in sorted(loose_files):
            if str(fp) in covered_files:
                continue

            raw_text = process_file_content(fp)
            if not raw_text.strip():
                raw_text = file_to_text(fp)

            fmt = detect_format_for_path(fp)  # NEW
            text = inject_format_hint(raw_text, fmt["format"])

            best_folder, best_score, second_score, margin, top5 = propose_destination(
                model, text, folder_centroids
            )

            decision = "REVIEW"
            if best_score >= AUTO_THRESHOLD and margin >= MARGIN_THRESHOLD:
                decision = "AUTO_SUGGEST"

            # TODO: Change to use SortRecord
            record = {
                "type": "file",
                "path": str(fp),
                "propose_move_to_category": best_folder,
                "best_score": best_score,
                "margin": margin,
                "decision": decision,
                "format": fmt, 
                "top5": top5,
            }
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1
            if count % 50 == 0:
                elapsed = time() - start_time
                print(f"[{count}] items processed ({elapsed:.1f}s) latest file: {fp.name} -> {best_folder}")

    print(f"Done. {count} item(s) processed.")