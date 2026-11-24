"""Fine-tuned transformer token classifier (RoBERTa family) for toxic spans.

Labels: a token is 1 if any non-whitespace character in its offset range is in the gold
spans, else 0. Special tokens get -100. Inference turns token probabilities into
characters with a threshold tuned on val, plus an optional whitespace-filling post-process.

Layout of the saved model folder (models/transformer/):
    config.json   our settings: base model, threshold, fill_whitespace, max_length, val scores
    hf/           Hugging Face weights + tokenizer (save_pretrained)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .metrics import evaluate
from .utils import MODELS_DIR, SEED, get_device, set_seed, token_offsets_to_char_indices

DEFAULT_DIR = MODELS_DIR / "transformer"
MAX_LENGTH = 256
THRESHOLDS = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]


def default_model_name(device: str) -> str:
    return "roberta-base" if device == "cuda" else "distilroberta-base"


# ----------------------------------------------------------------------------- encoding
def label_tokens(text: str, offsets, gold: set[int]) -> list[int]:
    labels = []
    for start, end in offsets:
        if end <= start:
            labels.append(-100)
            continue
        toxic = any((i in gold) and not text[i].isspace() for i in range(start, min(end, len(text))))
        labels.append(1 if toxic else 0)
    return labels


def encode(tokenizer, rows: list[dict], max_length: int = MAX_LENGTH) -> tuple[list[dict], int]:
    """Tokenize rows; returns (features, n_truncated)."""
    feats, n_trunc = [], 0
    for r in rows:
        enc = tokenizer(r["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        if len(tokenizer(r["text"], add_special_tokens=True)["input_ids"]) > max_length:
            n_trunc += 1
        feats.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": label_tokens(r["text"], enc["offset_mapping"], set(r["spans"])),
            }
        )
    return feats, n_trunc


def collate(batch: list[dict], pad_id: int):
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        mask[i, :n] = torch.tensor(b["attention_mask"])
        labels[i, :n] = torch.tensor(b["labels"])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ----------------------------------------------------------------------------- inference helpers
def token_probs(model, tokenizer, texts: list[str], device: str, batch_size: int = 32, max_length: int = MAX_LENGTH):
    """Return list of (offsets, probs) per text: P(toxic) for each token."""
    import torch

    model.eval()
    # Process texts sorted by length (less padding), then restore the original order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list = [None] * len(texts)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            chunk = [texts[k] for k in idx]
            enc = tokenizer(
                chunk, truncation=True, max_length=max_length, padding=True,
                return_offsets_mapping=True, return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping").tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1)[..., 1].cpu().numpy()
            lengths = enc["attention_mask"].sum(dim=1).cpu().tolist()
            for j, k in enumerate(idx):
                n = int(lengths[j])
                out[k] = (offsets[j][:n], probs[j][:n].tolist())
    return out


def length_grouped_batches(lengths: list[int], batch_size: int, seed: int) -> list[list[int]]:
    """Shuffle, sort within chunks of 50 batches by length, then shuffle the batches (less padding)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lengths)).tolist()
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: lengths[i])
        batches += [part[j : j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def probs_to_chars(text: str, offsets, probs, threshold: float, fill_whitespace: bool) -> list[int]:
    return token_offsets_to_char_indices(text, offsets, [p >= threshold for p in probs], fill_whitespace)


def sweep(rows: list[dict], cached) -> tuple[dict, list[dict]]:
    """Grid over thresholds x fill_whitespace on precomputed token probs; returns best + full grid."""
    golds = [r["spans"] for r in rows]
    grid = []
    for fill in (False, True):
        for th in THRESHOLDS:
            preds = [probs_to_chars(r["text"], o, p, th, fill) for r, (o, p) in zip(rows, cached)]
            grid.append({"threshold": th, "fill_whitespace": fill, "val_f1": evaluate(preds, golds)["f1"]})
    best = max(grid, key=lambda g: (g["val_f1"], -abs(g["threshold"] - 0.5)))
    return best, grid


# ----------------------------------------------------------------------------- training
def train(
    train_rows: list[dict],
    val_rows: list[dict],
    model_name: str | None = None,
    epochs: int = 3,
    lr: float = 3e-5,
    batch_size: int | None = None,
    max_train_samples: int | None = None,
    pos_weight: float = 1.0,
    device: str | None = None,
    out_dir: Path = DEFAULT_DIR,
    max_length: int = MAX_LENGTH,
) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(SEED)
    device = get_device(device)
    model_name = model_name or default_model_name(device)
    batch_size = batch_size or (16 if device == "cuda" else 8)
    if max_train_samples and max_train_samples < len(train_rows):
        rng = np.random.default_rng(SEED)
        idx = sorted(rng.choice(len(train_rows), size=max_train_samples, replace=False).tolist())
        train_rows = [train_rows[i] for i in idx]
    print(f"device={device} model={model_name} train={len(train_rows)} val={len(val_rows)} "
          f"epochs={epochs} lr={lr} batch={batch_size} pos_weight={pos_weight}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "O", 1: "TOXIC"}, label2id={"O": 0, "TOXIC": 1}
    ).to(device)

    feats, n_trunc = encode(tokenizer, train_rows, max_length)
    _, n_trunc_val = encode(tokenizer, val_rows, max_length)
    print(f"truncated to {max_length} tokens: train {n_trunc}/{len(train_rows)}, val {n_trunc_val}/{len(val_rows)}")

    lengths = [len(f["input_ids"]) for f in feats]
    n_batches = len(length_grouped_batches(lengths, batch_size, SEED))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * n_batches
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device), ignore_index=-100)

    best_f1, best_state, history = -1.0, None, []
    val_texts = [r["text"] for r in val_rows]
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        loader = [collate([feats[i] for i in b], tokenizer.pad_token_id)
                  for b in length_grouped_batches(lengths, batch_size, SEED + epoch)]
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            if step % 50 == 0 or step == len(loader):
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        cached = token_probs(model, tokenizer, val_texts, device, max_length=max_length)
        preds = [probs_to_chars(r["text"], o, p, 0.5, False) for r, (o, p) in zip(val_rows, cached)]
        f1 = evaluate(preds, [r["spans"] for r in val_rows])["f1"]
        history.append({"epoch": epoch, "train_loss": running / len(loader), "val_f1@0.5": f1})
        print(f"epoch {epoch}: train loss {running / len(loader):.4f}  val F1@0.5 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    cached = token_probs(model, tokenizer, val_texts, device, max_length=max_length)
    best, grid = sweep(val_rows, cached)
    print(f"best val: threshold={best['threshold']} fill_whitespace={best['fill_whitespace']} F1={best['val_f1']:.4f}")

    out_dir = Path(out_dir)
    (out_dir / "hf").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "hf")
    tokenizer.save_pretrained(out_dir / "hf")
    config = {
        "base_model": model_name,
        "threshold": best["threshold"],
        "fill_whitespace": best["fill_whitespace"],
        "max_length": max_length,
        "val_f1": best["val_f1"],
        "val_f1_no_fill": max(x["val_f1"] for x in grid if not x["fill_whitespace"]),
        "val_f1_fill": max(x["val_f1"] for x in grid if x["fill_whitespace"]),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "pos_weight": pos_weight,
        "train_samples": len(train_rows),
        "truncated_train_posts": n_trunc,
        "truncated_val_posts": n_trunc_val,
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "seed": SEED,
        "history": history,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "threshold_sweep.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    return config


# ----------------------------------------------------------------------------- predictor
class TransformerModel:
    name = "Transformer"

    def __init__(self, model_dir: Path = DEFAULT_DIR, device: str | None = None):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.dir = Path(model_dir)
        self.config = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir / "hf", use_fast=True, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.dir / "hf").to(self.device).eval()
        self.threshold = float(self.config["threshold"])
        self.fill = bool(self.config["fill_whitespace"])
        self.max_length = int(self.config.get("max_length", MAX_LENGTH))
        self.name = f"Transformer ({self.config['base_model']})"

    @staticmethod
    def available(model_dir: Path = DEFAULT_DIR) -> bool:
        return (Path(model_dir) / "config.json").exists() and (Path(model_dir) / "hf").exists()

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[list[int]]:
        cached = token_probs(self.model, self.tokenizer, texts, self.device, batch_size, self.max_length)
        return [probs_to_chars(t, o, p, self.threshold, self.fill) for t, (o, p) in zip(texts, cached)]

    def predict(self, text: str) -> list[int]:
        return self.predict_batch([text])[0]

    def score(self, rows: list[dict]) -> dict:
        return evaluate(self.predict_batch([r["text"] for r in rows]), [r["spans"] for r in rows])


def steps_estimate(n: int, batch_size: int, epochs: int) -> int:
    return epochs * math.ceil(n / batch_size)
