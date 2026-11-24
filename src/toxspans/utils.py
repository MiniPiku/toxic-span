"""Seeding, device detection, span <-> offset helpers and masking for display."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

WORD_RE = re.compile(r"\w+(?:['’]\w+)*")
VOWELS = set("aeiouAEIOU")


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_device(preference: str | None = None) -> str:
    """Return 'cuda', 'mps' or 'cpu'. `preference` ('auto' / None means detect)."""
    if preference and preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def indices_to_ranges(indices: Iterable[int]) -> list[tuple[int, int]]:
    """[1,2,3,7,8] -> [(1,4),(7,9)]  (end exclusive)."""
    idx = sorted(set(int(i) for i in indices))
    ranges: list[tuple[int, int]] = []
    for i in idx:
        if ranges and ranges[-1][1] == i:
            ranges[-1] = (ranges[-1][0], i + 1)
        else:
            ranges.append((i, i + 1))
    return ranges


def ranges_to_indices(ranges: Iterable[Sequence[int]]) -> list[int]:
    """[(1,4),(7,9)] -> [1,2,3,7,8]  (end exclusive)."""
    out: set[int] = set()
    for start, end in ranges:
        out.update(range(int(start), int(end)))
    return sorted(out)


def spans_to_strings(text: str, indices: Iterable[int]) -> list[str]:
    return [text[s:e] for s, e in indices_to_ranges(indices)]


def token_offsets_to_char_indices(
    text: str,
    offsets: Sequence[Sequence[int]],
    token_is_toxic: Sequence[bool],
    fill_whitespace: bool = False,
) -> list[int]:
    """Map token-level decisions to character indices.

    Characters of a toxic token are marked except whitespace (RoBERTa offsets
    can include a leading space). Special tokens have offset (0, 0) and are ignored.
    With `fill_whitespace`, the whitespace gap between two consecutive toxic
    tokens is also marked (gold spans like "stupid idiot" include the space).
    """
    chars: set[int] = set()
    prev_end: int | None = None  # end of previous toxic token (in token order)
    for (start, end), toxic in zip(offsets, token_is_toxic):
        if end <= start:  # special / padding token
            continue
        if not toxic:
            prev_end = None
            continue
        for c in range(start, min(end, len(text))):
            if not text[c].isspace():
                chars.add(c)
        if fill_whitespace and prev_end is not None:
            # first non-space char of this token
            first = start
            while first < end and first < len(text) and text[first].isspace():
                first += 1
            gap = text[prev_end:first]
            if gap and gap.isspace():
                chars.update(range(prev_end, first))
        prev_end = end
    return sorted(chars)


def mask_word(word: str) -> str:
    """Mild masking for docs/UI: keep the first letter, star the vowels. 'stupid' -> 'st*p*d'."""
    if len(word) <= 2:
        return word[0] + "*" * (len(word) - 1)
    masked = word[0] + "".join("*" if ch in VOWELS else ch for ch in word[1:])
    if "*" not in masked:  # no vowels: star the middle
        masked = word[0] + "*" * (len(word) - 2) + word[-1]
    return masked


def mask_text(text: str, indices: Iterable[int] = (), extra_words: set[str] | None = None) -> str:
    """Mask every word that overlaps `indices` or whose lowercase form is in `extra_words`."""
    idx = set(indices)
    extra = extra_words or set()
    out, last = [], 0
    for m in WORD_RE.finditer(text):
        out.append(text[last : m.start()])
        w = m.group(0)
        if w.lower() in extra or any(i in idx for i in range(m.start(), m.end())):
            out.append(mask_word(w))
        else:
            out.append(w)
        last = m.end()
    out.append(text[last:])
    return "".join(out)
