"""Centralised project configuration (paths and model settings)."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "SMSSpamCollection"
PROCESSED_DIR = DATA_DIR / "processed"
EMAIL_DB_PATH = PROCESSED_DIR / "emails.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"

MODEL_NAME = os.getenv("SPAM_MODEL_NAME", "Goodmotion/spam-mail-classifier")
MODEL_CACHE_DIR = Path(os.getenv("SPAM_MODEL_CACHE", PROJECT_ROOT / "models"))
MAX_LENGTH = 128
BATCH_SIZE = 32
ID2LABEL = {0: "ham", 1: "spam"}  # model card: ["NOSPAM", "SPAM"]
SPAM_THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.5"))
RANDOM_STATE = 42

# Fine-tuned version of MODEL_NAME (transfer learning on the SMS corpus).
FINETUNED_MODEL_DIR = PROJECT_ROOT / "models" / "spam-mail-classifier-sms"
SPLITS = {"train": PROCESSED_DIR / "train.csv", "val": PROCESSED_DIR / "val.csv",
          "test": PROCESSED_DIR / "test.csv"}


def resolve_model_path() -> str:
    """Use the fine-tuned checkpoint when present, else the original Hugging Face model."""
    if os.getenv("SPAM_MODEL_NAME"):
        return MODEL_NAME
    if (FINETUNED_MODEL_DIR / "config.json").exists():
        return str(FINETUNED_MODEL_DIR)
    return MODEL_NAME
