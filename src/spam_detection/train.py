"""Fine-tune ``Goodmotion/spam-mail-classifier`` on the SMS Spam Collection.

The original checkpoint was trained on e-mail subjects; SMS messages are short, informal and full
of abbreviations, so the zero-shot model suffers from domain shift. A light fine-tuning (transfer
learning) of the same network adapts it to the SMS domain.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import FINETUNED_MODEL_DIR, MAX_LENGTH, MODEL_CACHE_DIR, MODEL_NAME, RANDOM_STATE
from .data_split import load_splits


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _predict(model, tokenizer, texts, device, batch_size=64) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=MAX_LENGTH,
                        return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**enc).logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def fine_tune(epochs: int = 3, batch_size: int = 16, lr: float = 3e-5, weight_decay: float = 0.01,
              output_dir=FINETUNED_MODEL_DIR) -> dict:
    _set_seed(RANDOM_STATE)
    device = _device()
    splits = load_splits()
    train, val = splits["train"], splits["val"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR).to(device)

    # Class weights compensate the 87/13 ham/spam imbalance.
    counts = np.bincount(train["target"], minlength=2)
    weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    data = list(zip(train["text"].tolist(), train["target"].tolist()))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(RANDOM_STATE),
                        collate_fn=lambda b: ([t for t, _ in b], torch.tensor([y for _, y in b])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)

    history, best_f1 = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for texts, labels in loader:
            enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt").to(device)
            loss = loss_fn(model(**enc).logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
        val_prob = _predict(model, tokenizer, val["text"].tolist(), device)
        val_f1 = f1_score(val["target"], (val_prob >= 0.5).astype(int))
        record = {"epoch": epoch, "train_loss": running / len(loader), "val_f1": float(val_f1),
                  "seconds": round(time.perf_counter() - t0, 1)}
        history.append(record)
        print(json.dumps(record))
        if val_f1 > best_f1:
            best_f1 = val_f1
            output_dir.mkdir(parents=True, exist_ok=True)
            model.config.id2label = {0: "NOSPAM", 1: "SPAM"}
            model.config.label2id = {"NOSPAM": 0, "SPAM": 1}
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    summary = {"base_model": MODEL_NAME, "device": device, "epochs": epochs, "batch_size": batch_size,
               "learning_rate": lr, "best_val_f1": best_f1, "history": history,
               "train_size": len(train), "val_size": len(val)}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
