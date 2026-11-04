"""Evaluate the classifiers on the held-out test split and write ``reports/metrics.json``.

Models compared:
  * zero_shot  - Goodmotion/spam-mail-classifier exactly as published on Hugging Face
  * fine_tuned - the same network fine-tuned on the SMS training split (models/spam-mail-classifier-sms)
  * baseline   - TF-IDF (word + char n-grams) + Logistic Regression

Usage:
    python scripts/evaluate_model.py
"""
import json
import time

import pandas as pd

import _bootstrap  # noqa: F401
from spam_detection.baseline import build_baseline
from spam_detection.config import FINETUNED_MODEL_DIR, METRICS_PATH, MODEL_NAME, PREDICTIONS_PATH, REPORTS_DIR
from spam_detection.data_split import load_splits
from spam_detection.evaluation import compute_metrics
from spam_detection.model import SpamClassifier


def main() -> None:
    splits = load_splits()
    train, test = splits["train"], splits["test"]
    texts = test["text"].tolist()
    preds = test[["label", "text", "target"]].copy()
    results = {}

    candidates = {"zero_shot": MODEL_NAME}
    if (FINETUNED_MODEL_DIR / "config.json").exists():
        candidates["fine_tuned"] = str(FINETUNED_MODEL_DIR)
    for key, path in candidates.items():
        t0 = time.perf_counter()
        preds[f"prob_{key}"] = SpamClassifier(model_name=path).predict_proba(texts)
        results[key] = compute_metrics(test["target"], preds[f"prob_{key}"])
        results[key]["model"] = path
        results[key]["inference_seconds"] = round(time.perf_counter() - t0, 2)

    baseline = build_baseline().fit(train["text"], train["target"])
    preds["prob_baseline"] = baseline.predict_proba(texts)[:, 1]
    results["baseline"] = compute_metrics(test["target"], preds["prob_baseline"])
    results["baseline"]["model"] = "TF-IDF + LogisticRegression"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps({"split": "test", "n_test": len(test), "models": results}, indent=2))

    table = pd.DataFrame({k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                          for k, v in results.items()}).T
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
