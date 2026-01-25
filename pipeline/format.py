import magic
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from pipeline.file_format import FileExt, FileFormat

file_ext = FileExt()

@dataclass(frozen=True)
class FormatResult:
    format: FileFormat
    mime: Optional[str]
    confidence: float       # 0.0 – 1.0
    source: str             # magic | extension | heuristic
    
# -----------------------------
# Main function
# -----------------------------
def detect_file_format(path: Path) -> FormatResult:
    mime = None

    try:
        mime = magic.from_file(path, mime=True)
    except Exception:
        pass

    suffix = path.suffix.lower()

    # ---------- HARD PRIORITY (magic bytes) ----------
    if mime:
        if mime.startswith("image/"):
             FormatResult(FileFormat.IMAGE, mime, 0.95, "magic")

        if mime.startswith("audio/"):
            return FormatResult(FileFormat.AUDIO, mime, 0.95, "magic")

        if mime.startswith("video/"):
            return FormatResult(FileFormat.VIDEO, mime, 0.95, "magic")

        if mime in {
            "application/x-msdownload",
            "application/x-dosexec"
        }:
            return FormatResult(FileFormat.EXECUTABLE, mime, 0.95, "magic")

        if mime in {
            "application/zip",
            "application/x-tar",
            "application/x-7z-compressed",
            "application/x-rar"
        }:
            return FormatResult(FileFormat.ARCHIVE, mime, 0.9, "magic")

        if mime == "application/pdf":
            return FormatResult(FileFormat.DOCUMENT, mime, 0.9, "magic")

        # ---------- EXTENSION FALLBACK ----------
        if suffix in file_ext.INSTALLER_EXTS:
            return FormatResult(FileFormat.INSTALLER, mime, 0.85, "extension")

        if suffix in file_ext.ARCHIVE_EXTS:
            return FormatResult(FileFormat.ARCHIVE, mime, 0.8, "extension")

        if suffix in file_ext.EXECUTABLE_EXTS:
            return FormatResult(FileFormat.EXECUTABLE, mime, 0.8, "extension")

        if suffix in file_ext.CODE_EXTS:
            return FormatResult(FileFormat.CODE, mime, 0.75, "extension")

        if suffix in file_ext.DOCUMENT_EXTS:
            return FormatResult(FileFormat.DOCUMENT, mime, 0.75, "extension")

        if suffix in file_ext.DATA_EXTS:
            return FormatResult(FileFormat.DATA, mime, 0.75, "extension")

        if suffix in file_ext.FONT_EXTS:
            return FormatResult(FileFormat.FONT, mime, 0.75, "extension")

        if suffix in file_ext.DISK_IMAGE_EXTS:
            return FormatResult(FileFormat.DISK_IMAGE, mime, 0.8, "extension")

        if suffix in file_ext.DATABASE_EXTS:
            return FormatResult(FileFormat.DATABASE, mime, 0.75, "extension")

        if suffix in file_ext.EMAIL_EXTS:
            return FormatResult(FileFormat.EMAIL, mime, 0.75, "extension")

        if suffix in file_ext.CALENDAR_EXTS:
            return FormatResult(FileFormat.CALENDAR, mime, 0.75, "extension")

        if suffix in file_ext.SHORTCUT_EXTS:
            return FormatResult(FileFormat.SHORTCUT, mime, 0.7, "extension")

        if suffix in file_ext.CERT_EXTS:
            return FormatResult(FileFormat.CERTIFICATE, mime, 0.8, "extension")

        if suffix in file_ext.VM_EXTS:
            return FormatResult(FileFormat.VIRTUAL_MACHINE, mime, 0.8, "extension")
        
        return FormatResult(FileFormat.UNKNOWN, mime, 0.3, "heuristic")