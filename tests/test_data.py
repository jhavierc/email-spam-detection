from spam_detection.data_loader import load_dataset
from spam_detection.data_split import make_splits
from spam_detection.email_factory import make_subject
from spam_detection.features import extract_features


def test_dataset_matches_readme():
    df = load_dataset()
    assert len(df) == 5574
    assert df["label"].value_counts().to_dict() == {"ham": 4827, "spam": 747}


def test_splits_are_disjoint_and_stratified():
    s = make_splits(save=False)
    train, val, test = (set(s[k]["text"]) for k in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    for part in s.values():
        assert 0.10 < part["target"].mean() < 0.15


def test_features_and_subject():
    df = extract_features(load_dataset().head(50))
    assert {"n_chars", "has_url", "uppercase_ratio"} <= set(df.columns)
    assert make_subject("Hello there. How are you?") == "Hello there."
