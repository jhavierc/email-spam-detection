"""Loading utilities for the SMS Spam Collection v.1 dataset."""
from pathlib import Path

import pandas as pd

from .config import RAW_DATA_PATH


def load_dataset(path: Path | str = RAW_DATA_PATH, drop_duplicates: bool = False) -> pd.DataFrame:
    """Read the tab-separated corpus into a DataFrame with columns ``label``, ``text``, ``target``.

    ``target`` is 1 for spam and 0 for ham.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            label, text = line.split("\t", 1)
            rows.append((label.strip(), text.strip()))
    df = pd.DataFrame(rows, columns=["label", "text"])
    df["target"] = (df["label"] == "spam").astype(int)
    if drop_duplicates:
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df
