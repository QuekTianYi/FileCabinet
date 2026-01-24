from pathlib import Path
from apply_sort import apply_sort_proposals

PROPOSALS = Path("sorter_proposals.jsonl")
DEST_ROOT = Path.home() / "FileCabinet"

# First run: dry-run (HIGHLY recommended)
apply_sort_proposals(
    proposals_path=PROPOSALS,
    dest_root=DEST_ROOT,
    dry_run=True,
)

# When confident:
# apply_sort_proposals(PROPOSALS, DEST_ROOT, dry_run=False)