"""Build ``data/processed/emails.json``: the pool the web app's dummy e-mail generator draws from.

By default the pool is built from the *test* split, i.e. messages the fine-tuned model never saw
during training, so the inbox demo is an honest picture of real-world behaviour.

Usage:
    python scripts/build_email_db.py [--source test|all]
"""
import argparse
import json
import random

import _bootstrap  # noqa: F401
from spam_detection.config import EMAIL_DB_PATH, RANDOM_STATE
from spam_detection.data_loader import load_dataset
from spam_detection.data_split import load_splits
from spam_detection.email_factory import make_email


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["test", "all"], default="test")
    args = ap.parse_args()
    df = load_splits()["test"] if args.source == "test" else load_dataset(drop_duplicates=True)
    rng = random.Random(RANDOM_STATE)
    emails = [make_email(i, row.text, row.label, rng) for i, row in enumerate(df.itertuples())]
    EMAIL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMAIL_DB_PATH.write_text(json.dumps(emails, ensure_ascii=False, indent=2), encoding="utf-8")
    n_spam = sum(e["true_label"] == "spam" for e in emails)
    print(f"Wrote {len(emails)} e-mails ({n_spam} spam) to {EMAIL_DB_PATH}")


if __name__ == "__main__":
    main()
