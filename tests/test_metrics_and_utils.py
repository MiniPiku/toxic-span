import pytest

from toxspans.data import parse_spans
from toxspans.metrics import evaluate, f1_post
from toxspans.utils import (
    indices_to_ranges,
    mask_text,
    mask_word,
    ranges_to_indices,
    spans_to_strings,
    token_offsets_to_char_indices,
)


# ------------------------------------------------------------------ metric edge cases
def test_both_empty_is_one():
    assert f1_post([], []) == 1.0


def test_one_empty_is_zero():
    assert f1_post([1, 2], []) == 0.0
    assert f1_post([], [1, 2]) == 0.0


def test_perfect_and_disjoint():
    assert f1_post([1, 2, 3], [3, 2, 1]) == 1.0
    assert f1_post([1, 2], [5, 6]) == 0.0


def test_partial_overlap():
    # pred {1,2,3,4}, gold {3,4,5,6}: P = 0.5, R = 0.5, F1 = 0.5
    assert f1_post([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)
    # pred {1,2}, gold {1,2,3,4}: P=1, R=0.5 -> F1 = 2/3
    assert f1_post([1, 2], [1, 2, 3, 4]) == pytest.approx(2 / 3)


def test_duplicates_ignored():
    assert f1_post([1, 1, 2], [1, 2]) == 1.0


def test_evaluate_mean_and_micro():
    res = evaluate([[1, 2], [], [7]], [[1, 2], [], [8]])
    assert res["f1"] == pytest.approx((1 + 1 + 0) / 3)
    assert res["precision"] == pytest.approx(2 / 3)  # 2 tp out of 3 predicted chars
    assert res["recall"] == pytest.approx(2 / 3)
    assert res["n"] == 3


def test_evaluate_length_mismatch():
    with pytest.raises(ValueError):
        evaluate([[1]], [[1], [2]])


# ------------------------------------------------------------------ span parsing
def test_parse_spans_basic_and_dedupe_sort():
    assert parse_spans("[5, 3, 3, 4]") == [3, 4, 5]


def test_parse_spans_out_of_range_and_negative():
    assert parse_spans("[-1, 0, 2, 10]", text_len=5) == [0, 2]


def test_parse_spans_bad_input():
    assert parse_spans("") == []
    assert parse_spans("not a list") == []
    assert parse_spans(None) == []
    assert parse_spans("[]") == []
    assert parse_spans("__import__('os')") == []  # literal_eval refuses code


# ------------------------------------------------------------------ ranges
def test_indices_to_ranges_and_back():
    idx = [1, 2, 3, 7, 8, 12]
    ranges = indices_to_ranges(idx)
    assert ranges == [(1, 4), (7, 9), (12, 13)]
    assert ranges_to_indices(ranges) == idx


def test_ranges_empty():
    assert indices_to_ranges([]) == []
    assert ranges_to_indices([]) == []


def test_spans_to_strings():
    text = "you are a dumb clown"
    assert spans_to_strings(text, list(range(10, 14)) + list(range(15, 20))) == ["dumb", "clown"]


# ------------------------------------------------------------------ token offsets -> characters
def test_offsets_strip_leading_space_and_ignore_special():
    text = "  Hello, dumb world!"
    # RoBERTa-style offsets: special tokens (0,0); a token can include its leading space.
    offsets = [(0, 0), (0, 7), (7, 8), (8, 13), (13, 19), (19, 20), (0, 0)]
    toxic = [True, False, False, True, False, False, True]
    assert token_offsets_to_char_indices(text, offsets, toxic) == [9, 10, 11, 12]


def test_offsets_fill_whitespace_between_toxic_tokens():
    text = "a dumb clown here"
    offsets = [(0, 1), (1, 6), (6, 12), (12, 17)]
    toxic = [False, True, True, False]
    no_fill = token_offsets_to_char_indices(text, offsets, toxic, fill_whitespace=False)
    fill = token_offsets_to_char_indices(text, offsets, toxic, fill_whitespace=True)
    assert spans_to_strings(text, no_fill) == ["dumb", "clown"]
    assert spans_to_strings(text, fill) == ["dumb clown"]


def test_offsets_no_fill_across_non_toxic_token():
    text = "dumb and clown"
    offsets = [(0, 4), (4, 8), (8, 14)]
    toxic = [True, False, True]
    out = token_offsets_to_char_indices(text, offsets, toxic, fill_whitespace=True)
    assert spans_to_strings(text, out) == ["dumb", "clown"]


# ------------------------------------------------------------------ masking
def test_mask_word():
    assert mask_word("stupid") == "st*p*d"
    assert "*" in mask_word("dmb")


def test_mask_text_only_masks_span_words():
    text = "what a stupid plan"
    out = mask_text(text, indices=range(7, 13))
    assert out == "what a st*p*d plan"
