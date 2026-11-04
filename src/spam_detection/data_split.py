"""Deterministic, stratified train/validation/test split (after de-duplication)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DIR, RANDOM_STATE, SPLITS
from .data_loader import load_dataset


def make_splits(val_size: float = 0.15, test_size: float = 0.15, save: bool = True) -> dict[str, pd.DataFrame]:
    # Duplicated messages are removed first so the same text can't leak across splits.
    df = load_dataset(drop_duplicates=True)
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["target"], random_state=RANDOM_STATE)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["target"],
                                  random_state=RANDOM_STATE)
    splits = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part.to_csv(SPLITS[name], index=False)
    return splits


def load_splits() -> dict[str, pd.DataFrame]:
    if not all(p.exists() for p in SPLITS.values()):
        return make_splits()
    return {name: pd.read_csv(path, keep_default_na=False) for name, path in SPLITS.items()}
