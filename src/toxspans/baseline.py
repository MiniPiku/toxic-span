"""Lexicon baseline: a data-driven word list built from the training spans.

For each lowercase word w in the training set:
    count(w)        = number of occurrences
    toxic_count(w)  = occurrences whose characters fall inside gold spans (majority of chars)
    toxic_ratio(w)  = toxic_count / count
Keep words with toxic_ratio >= t and count >= c (t, c tuned on val).
Prediction: mark the characters of every lexicon word found in the text.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from .metrics import evaluate
from .utils import MODELS_DIR, WORD_RE

DEFAULT_DIR = MODELS_DIR / "baseline"
T_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
C_GRID = [1, 2, 3, 5, 10]


def word_stats(rows: list[dict]) -> tuple[Counter, Counter]:
    counts, toxic = Counter(), Counter()
    for r in rows:
        gold = set(r["spans"])
        for m in WORD_RE.finditer(r["text"]):
            w = m.group(0).lower()
            counts[w] += 1
            inside = sum(1 for i in range(m.start(), m.end()) if i in gold)
            if inside * 2 >= (m.end() - m.start()):
                toxic[w] += 1
    return counts, toxic


class LexiconModel:
    name = "Lexicon baseline"

    def __init__(self, lexicon: dict[str, float] | None = None, t: float = 0.5, c: int = 2):
        self.lexicon = lexicon or {}
        self.t, self.c = t, c

    # --------------------------------------------------------------- training
    @classmethod
    def from_stats(cls, counts: Counter, toxic: Counter, t: float, c: int) -> "LexiconModel":
        lex = {w: toxic[w] / n for w, n in counts.items() if n >= c and toxic[w] / n >= t}
        return cls(lex, t, c)

    @classmethod
    def fit(cls, train: list[dict], val: list[dict] | None = None, verbose: bool = True) -> tuple["LexiconModel", list]:
        counts, toxic = word_stats(train)
        if not val:
            return cls.from_stats(counts, toxic, 0.5, 2), []
        grid = []
        best, best_f1 = None, -1.0
        for t, c in itertools.product(T_GRID, C_GRID):
            m = cls.from_stats(counts, toxic, t, c)
            f1 = m.score(val)["f1"]
            grid.append({"t": t, "c": c, "size": len(m.lexicon), "val_f1": f1})
            if f1 > best_f1:
                best, best_f1 = m, f1
        if verbose:
            print(f"Best lexicon: t={best.t} c={best.c} size={len(best.lexicon)} val F1={best_f1:.4f}")
        return best, grid

    # --------------------------------------------------------------- inference
    def predict(self, text: str) -> list[int]:
        out: list[int] = []
        for m in WORD_RE.finditer(text):
            if m.group(0).lower() in self.lexicon:
                out.extend(range(m.start(), m.end()))
        return out

    def predict_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.predict(t) for t in texts]

    def score(self, rows: list[dict]) -> dict:
        return evaluate(self.predict_batch([r["text"] for r in rows]), [r["spans"] for r in rows])

    # --------------------------------------------------------------- io
    def save(self, out_dir: Path = DEFAULT_DIR) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "lexicon.json"
        payload = {"t": self.t, "c": self.c, "size": len(self.lexicon), "lexicon": dict(sorted(self.lexicon.items()))}
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, model_dir: Path = DEFAULT_DIR) -> "LexiconModel":
        payload = json.loads((Path(model_dir) / "lexicon.json").read_text(encoding="utf-8"))
        return cls(payload["lexicon"], payload["t"], payload["c"])

    @staticmethod
    def available(model_dir: Path = DEFAULT_DIR) -> bool:
        return (Path(model_dir) / "lexicon.json").exists()
