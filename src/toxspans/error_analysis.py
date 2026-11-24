"""Error analysis of the best model -> results/error_analysis.md (offensive words masked)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .baseline import word_stats
from .data import load_split
from .metrics import evaluate, f1_post
from .utils import RESULTS_DIR, WORD_RE, indices_to_ranges, mask_word

LENGTH_BUCKETS = [(0, 100, "< 100 chars"), (100, 200, "100-199"), (200, 400, "200-399"), (400, 10**9, ">= 400")]
OBFUSCATED_RE = re.compile(r"\w[*@#$!%.]+\w|\w\*+|\*+\w")
STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being am it its it's "
    "this that these those i you he she we they me him her us them my your his our their what who which "
    "so not no do does did have has had will would can could should just than then there here all any".split()
)


def word_errors(rows: list[dict], preds: list[list[int]]) -> tuple[Counter, Counter, int, int]:
    """Word-level false positives / negatives (a word is 'marked' if >= half its chars are)."""
    fp, fn, n_fp, n_fn = Counter(), Counter(), 0, 0
    for r, p in zip(rows, preds):
        gold, pred = set(r["spans"]), set(p)
        for m in WORD_RE.finditer(r["text"]):
            rng = range(m.start(), m.end())
            g = sum(i in gold for i in rng) * 2 >= len(rng)
            q = sum(i in pred for i in rng) * 2 >= len(rng)
            w = m.group(0).lower()
            if q and not g:
                fp[w] += 1
                n_fp += 1
            elif g and not q:
                fn[w] += 1
                n_fn += 1
    return fp, fn, n_fp, n_fn


def bucket_scores(rows, preds_by_model: dict[str, list]) -> list[dict]:
    out = []
    groups = [(lo <= len(r["text"]) < hi) for lo, hi, _ in LENGTH_BUCKETS for r in rows]
    for bi, (_, _, label) in enumerate(LENGTH_BUCKETS):
        idx = [i for i, r in enumerate(rows) if groups[bi * len(rows) + i]]
        out.append({"group": f"Length {label}", "n": len(idx),
                    **{k: evaluate([p[i] for i in idx], [rows[i]["spans"] for i in idx])["f1"] if idx else float("nan")
                       for k, p in preds_by_model.items()}})
    for label, cond in (("Posts with toxic spans", True), ("Posts with no toxic spans", False)):
        idx = [i for i, r in enumerate(rows) if bool(r["spans"]) == cond]
        out.append({"group": label, "n": len(idx),
                    **{k: evaluate([p[i] for i in idx], [rows[i]["spans"] for i in idx])["f1"] if idx else float("nan")
                       for k, p in preds_by_model.items()}})
    return out


def run(results_dir: Path = RESULTS_DIR) -> str:
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    preds_all = json.loads((results_dir / "predictions_test.json").read_text(encoding="utf-8"))
    test = load_split("test")
    full_models = {k: v for k, v in metrics["models"].items() if "full_test" in v}
    if not full_models:
        raise SystemExit("No model evaluated on the full test set - run `make evaluate` first.")
    best = max(full_models, key=lambda k: full_models[k]["full_test"]["f1"])
    best_name = full_models[best]["name"]
    preds = preds_all[best]

    fp, fn, n_fp, n_fn = word_errors(test, preds)
    train_counts, train_toxic = word_stats(load_split("train"))
    per_post = [f1_post(p, r["spans"]) for p, r in zip(preds, test)]
    groups = bucket_scores(test, {k: preds_all[k] for k in full_models})

    # ---------------- data-driven observations
    empty = [i for i, r in enumerate(test) if not r["spans"]]
    empty_fp = sum(1 for i in empty if preds[i])
    fp_often_toxic = sum(c for w, c in fp.items() if train_counts[w] >= 5 and train_toxic[w] / train_counts[w] >= 0.3)
    fn_rare = sum(c for w, c in fn.items() if train_counts[w] < 3)
    fn_stop = sum(c for w, c in fn.items() if w in STOPWORDS)
    fn_neutral = [w for w, _ in fn.most_common(40) if w not in STOPWORDS and train_counts[w]
                  and train_toxic[w] / train_counts[w] < 0.1][:4]
    gold_spans = [r["text"][s:e] for r in test for s, e in indices_to_ranges(r["spans"])]
    obfus = sum(1 for s in gold_spans if OBFUSCATED_RE.search(s))
    long_spans = sum(1 for s in gold_spans if len(WORD_RE.findall(s)) >= 4)
    partial = sum(1 for f in per_post if 0 < f < 1)
    len_scores = [g[best] for g in groups[:4]]

    def pct(a, b):
        return 100 * a / b if b else 0.0

    obs = [
        f"**The metric punishes any prediction on clean posts.** {len(empty)} of {len(test)} test posts "
        f"({pct(len(empty), len(test)):.1f}%) have no gold span; the model flagged something in {empty_fp} of them "
        f"({pct(empty_fp, len(empty)):.1f}%), and each of those scores F1 = 0. Many such posts are offensive overall "
        f"but annotators did not tie the toxicity to specific words (context-dependent toxicity).",
        f"**False positives are mostly genuinely rude words that were not annotated in that post.** "
        f"{pct(fp_often_toxic, n_fp):.1f}% of false-positive word occurrences are words that are marked toxic in at "
        f"least 30% of their training occurrences (e.g. {', '.join(mask_word(w) for w, _ in fp.most_common(3))}). "
        f"Whether such a word is 'the toxic span' depends on context and annotator judgement (e.g. self-directed "
        f"or quoted use, sarcasm), which a word list cannot see and a model trained on ~8k posts learns only partly.",
        f"**Many missed words are harmless out of context; fewer are rare or obfuscated.** {pct(fn_rare, n_fn):.1f}% of missed "
        f"word occurrences appear fewer than 3 times in training and {obfus} of {len(gold_spans)} gold spans "
        f"({pct(obfus, len(gold_spans)):.1f}%) contain obfuscation (asterisks or symbols inside a word). Several of the "
        f"most-missed words are neutral on their own ({', '.join(fn_neutral) or 'none'}): they are toxic only as part of "
        f"a threat or accusation, which needs context rather than a word list.",
        f"**Boundary disagreement is common.** {partial} posts ({pct(partial, len(test)):.1f}%) get partial credit "
        f"(0 < F1 < 1); {long_spans} gold spans ({pct(long_spans, len(gold_spans)):.1f}%) are phrases of 4+ words where "
        f"annotators marked a whole insult or threat, while models mark only the strongest word.",
        (f"**Longer posts are harder.** " if len_scores[0] - len_scores[-1] > 0.05 else f"**Length has a small effect.** ")
        + "F1 by length bucket for the best model: "
        + ", ".join(f"{b[2]}: {s:.3f}" for b, s in zip(LENGTH_BUCKETS, len_scores))
        + ". Long posts give more chances for both extra and missed words; posts over 256 tokens are also truncated "
        "for the transformer.",
    ]

    # ---------------- markdown
    model_keys = list(full_models)
    lines = [
        "# Error analysis", "",
        f"Best model on the full test set: **{best_name}** "
        f"(F1 = {full_models[best]['full_test']['f1']:.4f}, n = {len(test)}).", "",
        "Offensive words are masked (first letter kept, vowels replaced by `*`). "
        "A word counts as *marked* when at least half of its characters are marked.", "",
        f"## Top 10 false positives (word marked by the model, not in gold) - {n_fp} occurrences in total", "",
        "Function words are excluded from both top-10 tables.", "",
        "| Word (masked) | Count | Toxic ratio in train |", "|---|---|---|",
    ]
    def disp(w: str) -> str:  # mask words with real toxic usage (or unseen words); show benign words as-is
        ratio = train_toxic[w] / train_counts[w] if train_counts[w] else 1.0
        return mask_word(w) if ratio >= 0.1 else w

    for w, c in [x for x in fp.most_common() if x[0] not in STOPWORDS][:10]:
        ratio = train_toxic[w] / train_counts[w] if train_counts[w] else 0.0
        lines.append(f"| {disp(w)} | {c} | {ratio:.2f} |")
    lines += ["", f"## Top 10 false negatives (gold word the model missed) - {n_fn} occurrences in total", "",
              "| Word (masked) | Count | Occurrences in train |", "|---|---|---|"]
    for w, c in [x for x in fn.most_common() if x[0] not in STOPWORDS][:10]:
        lines.append(f"| {disp(w)} | {c} | {train_counts[w]} |")
    lines += ["", f"Function words (e.g. *and*, *the*, *are*) are left out of both tables: they account for "
              f"{fn_stop} missed occurrences ({pct(fn_stop, n_fn):.1f}% of false negatives), almost all inside "
              f"multi-word gold phrases."]
    names = {k: full_models[k]["name"] for k in model_keys}
    lines += ["", "## F1 by text length and by presence of toxic spans (full test set)", "",
              "| Group | Posts | " + " | ".join(names[k] for k in model_keys) + " |",
              "|---|---|" + "---|" * len(model_keys)]
    for g in groups:
        lines.append(f"| {g['group']} | {g['n']} | " + " | ".join(f"{g[k]:.3f}" for k in model_keys) + " |")
    lines += ["", "## Observations", ""] + [f"{i}. {o}" for i, o in enumerate(obs, 1)] + [""]
    md = "\n".join(lines)
    (results_dir / "error_analysis.md").write_text(md, encoding="utf-8")
    return md
