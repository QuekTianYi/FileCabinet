import json
from pathlib import Path
from pipeline import runner
import numpy as np

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Basic validation + normalization
    if "folder_to_prototypes" not in cfg or not isinstance(cfg["folder_to_prototypes"], dict):
        raise ValueError("config.json must contain folder_to_prototypes as an object/dict")

    cfg["watch_dir"] = Path(cfg["watch_dir"]).expanduser()
    cfg["log_path"] = Path(cfg["log_path"]).expanduser()

    return cfg

'''
Iterate through each file and subfolders
'''
def main():
    print("Loading configuration...")
    cfg = load_config(Path("config.json"))

    WATCH_DIR: Path = cfg["watch_dir"]
    LOG_PATH: Path = cfg["log_path"]
    AUTO_THRESHOLD: float = float(cfg.get("auto_threshold", 0.45))
    MARGIN_THRESHOLD: float = float(cfg.get("margin_threshold", 0.05))
    FOLDER_TO_PROTOTYPES: dict = cfg["folder_to_prototypes"]
    
    runner.run(WATCH_DIR, LOG_PATH, AUTO_THRESHOLD, MARGIN_THRESHOLD, FOLDER_TO_PROTOTYPES)

if __name__ == "__main__":
    main()