"""Dummy e-mail generator: draws random messages from the JSON pool without repetition."""
from __future__ import annotations

import json
import random
from pathlib import Path


class EmailGenerator:
    def __init__(self, db_path: Path, seed: int | None = None):
        self.db_path = Path(db_path)
        self._rng = random.Random(seed)
        self._pool: list[dict] = []
        self._queue: list[dict] = []
        if self.db_path.exists():
            self._pool = json.loads(self.db_path.read_text(encoding="utf-8"))

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def sample(self, n: int = 1, spam_ratio: float | None = None) -> list[dict]:
        """Return ``n`` e-mails. With ``spam_ratio`` the share of spam is forced (demo-friendly)."""
        if not self._pool:
            return []
        if spam_ratio is not None:
            spam = [e for e in self._pool if e["true_label"] == "spam"]
            ham = [e for e in self._pool if e["true_label"] == "ham"]
            k_spam = sum(self._rng.random() < spam_ratio for _ in range(n))
            out = self._rng.sample(spam, min(k_spam, len(spam))) + self._rng.sample(ham, min(n - k_spam, len(ham)))
            self._rng.shuffle(out)
            return [dict(e) for e in out]
        out = []
        for _ in range(n):
            if not self._queue:
                self._queue = self._pool[:]
                self._rng.shuffle(self._queue)
            out.append(dict(self._queue.pop()))
        return out
