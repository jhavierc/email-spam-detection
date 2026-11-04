"""Wrapper around the Hugging Face model ``Goodmotion/spam-mail-classifier``.

Base model: microsoft/Multilingual-MiniLM-L12-H384 fine-tuned for binary SPAM / NOSPAM
classification. Index 0 -> NOSPAM (ham), index 1 -> SPAM.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from typing import Iterable

from .config import BATCH_SIZE, ID2LABEL, MAX_LENGTH, MODEL_CACHE_DIR, SPAM_THRESHOLD, resolve_model_path


@dataclass
class Prediction:
    label: str           # "spam" | "ham"
    spam_probability: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


class SpamClassifier:
    """Lazy-loading, thread-safe classifier."""

    def __init__(self, model_name: str | None = None, threshold: float = SPAM_THRESHOLD,
                 device: str | None = None):
        self.model_name = model_name or resolve_model_path()
        self.threshold = threshold
        self._device = device
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> "SpamClassifier":
        with self._lock:
            if self._model is None:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                if self._device is None:
                    self._device = ("cuda" if torch.cuda.is_available()
                                    else "mps" if torch.backends.mps.is_available() else "cpu")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=MODEL_CACHE_DIR)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, cache_dir=MODEL_CACHE_DIR
                ).to(self._device)
                self._model.eval()
        return self

    @property
    def is_finetuned(self) -> bool:
        return not self.model_name.startswith("Goodmotion/")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict_proba(self, texts: Iterable[str], batch_size: int = BATCH_SIZE) -> list[float]:
        """Return P(spam) for each text."""
        import torch

        self.load()
        texts = [t if t else " " for t in texts]
        probs: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                                  return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        return probs

    def predict(self, texts: Iterable[str], batch_size: int = BATCH_SIZE) -> list[Prediction]:
        results = []
        for p in self.predict_proba(texts, batch_size):
            label = ID2LABEL[int(p >= self.threshold)]
            results.append(Prediction(label=label, spam_probability=round(p, 6),
                                      confidence=round(p if label == "spam" else 1 - p, 6)))
        return results

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]
