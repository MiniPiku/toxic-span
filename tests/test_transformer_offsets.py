"""Offset-to-character mapping with a real fast RoBERTa tokenizer (skipped if it cannot be loaded)."""
import pytest

from toxspans.transformer import label_tokens, length_grouped_batches, probs_to_chars
from toxspans.utils import spans_to_strings


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, add_prefix_space=True)
    except Exception as e:  # offline and not cached
        pytest.skip(f"tokenizer unavailable: {e}")


TEXT = "   Well, you're a total idiot... and dumb!!"


def _gold(text, words):
    out = []
    for w in words:
        s = text.index(w)
        out += range(s, s + len(w))
    return out


def test_labels_and_roundtrip_with_leading_spaces_and_punctuation(tokenizer):
    gold = _gold(TEXT, ["idiot", "dumb"])
    enc = tokenizer(TEXT, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    labels = label_tokens(TEXT, offsets, set(gold))
    # special tokens ignored
    assert labels[0] == -100 and labels[-1] == -100
    # every labelled-toxic token's text, stripped, lies inside a gold word
    toxic_pieces = {TEXT[s:e].strip() for (s, e), l in zip(offsets, labels) if l == 1}
    assert toxic_pieces and all(p in ("idiot", "dumb") or p in "idiot dumb" for p in toxic_pieces)
    # perfect token probabilities -> exact gold characters back (no whitespace, no punctuation)
    probs = [1.0 if l == 1 else 0.0 for l in labels]
    chars = probs_to_chars(TEXT, offsets, probs, threshold=0.5, fill_whitespace=False)
    assert chars == sorted(gold)
    assert spans_to_strings(TEXT, chars) == ["idiot", "dumb"]


def test_fill_whitespace_joins_multiword_span(tokenizer):
    text = "  you total idiot."
    gold = _gold(text, ["total idiot"])
    enc = tokenizer(text, return_offsets_mapping=True)
    labels = label_tokens(text, enc["offset_mapping"], set(gold))
    probs = [1.0 if l == 1 else 0.0 for l in labels]
    no_fill = probs_to_chars(text, enc["offset_mapping"], probs, 0.5, False)
    fill = probs_to_chars(text, enc["offset_mapping"], probs, 0.5, True)
    assert spans_to_strings(text, no_fill) == ["total", "idiot"]
    assert spans_to_strings(text, fill) == ["total idiot"]
    assert fill == sorted(gold)


def test_length_grouped_batches_cover_all_indices_once():
    lengths = [5, 50, 7, 33, 2, 90, 11, 4, 60, 8]
    batches = length_grouped_batches(lengths, 3, seed=42)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))
    assert batches == length_grouped_batches(lengths, 3, seed=42)  # deterministic
