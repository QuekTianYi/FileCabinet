from enum import Enum, auto

class FileFormat(Enum):
    DOCUMENT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    ARCHIVE = auto()
    INSTALLER = auto()
    EXECUTABLE = auto()
    CODE = auto()
    DATA = auto()
    FONT = auto()
    DISK_IMAGE = auto()
    CONFIG = auto()
    DATABASE = auto()
    EMAIL = auto()
    CALENDAR = auto()
    SHORTCUT = auto()
    BACKUP = auto()
    CERTIFICATE = auto()
    FIRMWARE = auto()
    VIRTUAL_MACHINE = auto()
    GAME_ASSET = auto()
    UNKNOWN = auto()

class FileExt():
    DOCUMENT_EXTS = {
    ".txt", ".md", ".rst", ".pdf", ".doc", ".docx", ".odt", ".rtf", ".tex"
    }

    IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic", ".svg"
    }

    AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".mid", ".midi"
    }

    VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv"
    }

    ARCHIVE_EXTS = {
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".tgz"
    }

    INSTALLER_EXTS = {
    ".msi", ".pkg", ".dmg", ".deb", ".rpm", ".apk"
    }

    EXECUTABLE_EXTS = {
    ".exe", ".app", ".bin", ".run", ".sh", ".bat", ".cmd", ".ps1"
    }

    CODE_EXTS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs",
    ".php", ".rb", ".html", ".css", ".scss", ".xml", ".yaml", ".yml", ".json", ".toml"
    }

    DATA_EXTS = {
    ".csv", ".tsv", ".xls", ".xlsx", ".ods", ".parquet", ".feather", ".jsonl"
    }

    FONT_EXTS = {".ttf", ".otf", ".woff", ".woff2"}
    DISK_IMAGE_EXTS = {".iso", ".img", ".vhd", ".vhdx", ".vmdk"}
    DATABASE_EXTS = {".db", ".sqlite", ".sqlite3", ".mdb"}
    EMAIL_EXTS = {".eml", ".msg", ".pst", ".mbox"}
    CALENDAR_EXTS = {".ics"}
    SHORTCUT_EXTS = {".lnk", ".url", ".desktop", ".webloc"}
    CERT_EXTS = {".pem", ".crt", ".cer", ".key", ".p12", ".pfx"}
    VM_EXTS = {".ova", ".ovf", ".vdi"}
    GAME_ASSET_EXTS = {".pak", ".wad", ".unity3d", ".uasset"}