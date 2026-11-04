"""Fine-tune the Hugging Face spam classifier on the SMS corpus.

Usage:
    python scripts/train_model.py [--epochs 3] [--batch-size 16] [--lr 3e-5]
"""
import argparse

import _bootstrap  # noqa: F401
from spam_detection.data_split import make_splits
from spam_detection.train import fine_tune

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    a = ap.parse_args()
    make_splits()
    fine_tune(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr)
