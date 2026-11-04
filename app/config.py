"""Flask configuration."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    EMAIL_DB_PATH = Path(os.getenv("EMAIL_DB_PATH", ROOT / "data" / "processed" / "emails.json"))
    MAILBOX_PATH = Path(os.getenv("MAILBOX_PATH", ROOT / "instance" / "mailbox.json"))
    MAILBOX_OWNER = os.getenv("MAILBOX_OWNER", "student@spamshield.dev")
    MAX_GENERATE = 50
    PRELOAD_MODEL = True
    SEED = None
    JSON_SORT_KEYS = False
