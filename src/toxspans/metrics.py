"""Official SemEval-2021 Task 5 metric: per-post character F1, averaged over posts.

Per post:  P = |pred ∩ gold| / |pred|,  R = |pred ∩ gold| / |gold|,  F1 = 2PR / (P + R).
Both empty -> F1 = 1.  Exactly one empty -> F1 = 0.
Headline = mean per-post F1. Micro precision/recall are computed over all characters.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def f1_post(pred: Iterable[int], gold: Iterable[int]) -> float:
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def precision_recall_post(pred: Iterable[int], gold: Iterable[int]) -> tuple[float, float]:
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0, 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    return prec, rec


def evaluate(preds: Sequence[Iterable[int]], golds: Sequence[Iterable[int]]) -> dict:
    """Return mean per-post F1 (headline) plus micro P/R/F1 over characters."""
    if len(preds) != len(golds):
        raise ValueError(f"{len(preds)} predictions vs {len(golds)} gold posts")
    n = len(golds)
    f1s, tp, n_pred, n_gold = [], 0, 0, 0
    for pred, gold in zip(preds, golds):
        p, g = set(pred), set(gold)
        f1s.append(f1_post(p, g))
        tp += len(p & g)
        n_pred += len(p)
        n_gold += len(g)
    micro_p = tp / n_pred if n_pred else 0.0
    micro_r = tp / n_gold if n_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {
        "f1": sum(f1s) / n if n else 0.0,
        "precision": micro_p,
        "recall": micro_r,
        "micro_f1": micro_f1,
        "n": n,
    }
