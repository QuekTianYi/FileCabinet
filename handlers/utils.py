# Tuning knobs for folder/zip content sampling (speed vs accuracy)
MAX_FILES_PER_CONTAINER = 30         # max files to read per folder/zip
MAX_TEXT_CHARS_PER_ITEM = 2000       # max chars extracted per file (handler should respect too)
MAX_TOTAL_TEXT_CHARS = 12000         # cap for aggregated folder/zip text
ZIP_EXTS = {".zip"}