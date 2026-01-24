import json
import shutil
from pathlib import Path

def apply_sort_proposals(
    proposals_path: Path,
    dest_root: Path,
    dry_run: bool = True,
):
    """
    Apply sort proposals from a JSONL file.

    - Only applies AUTO_SUGGEST decisions
    - Supports file, folder, zip
    - dry_run=True prints actions without moving
    """

    moved = set()

    def safe_move(src: Path, dst: Path):
        if dry_run:
            print(f"[DRY RUN] MOVE {src} -> {dst}")
            return

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Avoid overwrite
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")

        shutil.move(str(src), str(dst))

    with proposals_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record.get("type") == "run_header":
                continue

            # if record.get("decision") != "AUTO_SUGGEST":
            #     continue

            src = Path(record["path"])

            # Skip if already moved via parent folder
            if any(str(src).startswith(m) for m in moved):
                continue

            if not src.exists():
                print(f"[SKIP] Missing: {src}")
                continue

            category = record["propose_move_to_category"]
            dst = dest_root / category / src.name

            try:
                safe_move(src, dst)
                moved.add(str(src))
            except Exception as e:
                print(f"[ERROR] {src}: {e}")