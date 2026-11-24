"""Download, load, split and describe the SemEval-2021 Task 5 (Toxic Spans) data.

Source of truth: github.com/ipavlopoulos/toxic_spans, folder SemEval2021/data/
(tsd_train.csv, tsd_trial.csv, tsd_test.csv; columns: spans, text).
Splits used: train = tsd_train, val = tsd_trial, test = tsd_test (official labelled test).
"""
from __future__ import annotations

import ast
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

from .utils import DATA_DIR, RESULTS_DIR, SEED, WORD_RE

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GITHUB_BASE = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/SemEval2021/data/"
RAW_FILES = {"train": "tsd_train.csv", "trial": "tsd_trial.csv", "test": "tsd_test.csv"}
# Candidate Hugging Face mirrors (tried in order if GitHub fails). Columns must include text + spans.
HF_CANDIDATES = ["heegyu/toxic-spans", "nmiko/toxic_spans"]
SYNTHETIC_FLAG = RAW_DIR / "SYNTHETIC_DATA_WARNING.txt"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ----------------------------------------------------------------------------- parsing
def parse_spans(raw, text_len: int | None = None) -> list[int]:
    """Safely parse a stringified list of char indices; dedupe, sort, drop out-of-range."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    else:
        value = raw
    if not isinstance(value, (list, tuple, set)):
        return []
    out = set()
    for v in value:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i < 0 or (text_len is not None and i >= text_len):
            continue
        out.add(i)
    return sorted(out)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"text": r["text"], "spans": parse_spans(r.get("spans"), len(r["text"]))} for r in rows]


# ----------------------------------------------------------------------------- download
def _download_github(raw_dir: Path) -> bool:
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        for fname in RAW_FILES.values():
            dest = raw_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            r = requests.get(GITHUB_BASE + fname, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"  downloaded {fname} ({len(r.content):,} bytes)")
        return True
    except Exception as e:  # network / HTTP errors
        print(f"  GitHub download failed: {e}")
        return False


def _download_hf(raw_dir: Path) -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        return False
    split_map = {"train": "train", "trial": "validation", "test": "test"}
    for name in HF_CANDIDATES:
        try:
            ds = load_dataset(name)
        except Exception as e:
            print(f"  HF mirror {name} unavailable: {type(e).__name__}")
            continue
        if "train" not in ds or not {"text", "spans"} <= set(ds["train"].column_names):
            continue
        for ours, theirs in split_map.items():
            if theirs not in ds:
                continue
            with open(raw_dir / RAW_FILES[ours], "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["spans", "text"])
                for ex in ds[theirs]:
                    w.writerow([str(list(ex["spans"])), ex["text"]])
        print(f"  used Hugging Face mirror {name}")
        return True
    return False


def _write_synthetic(raw_dir: Path) -> None:
    """Tiny, clearly-labelled synthetic sample (mild insults only) so the pipeline still runs."""
    rng = random.Random(SEED)
    insults = ["stupid", "idiot", "dumb", "moron", "clown", "pathetic", "loser", "foolish"]
    neutral = [
        "I think the council made a reasonable decision on the budget.",
        "Thanks for sharing, this was an interesting read.",
        "The weather has been lovely this week.",
        "Could you post the source for that statistic?",
        "Traffic on the bridge was slow again today.",
    ]
    templates = [
        "What a {w} idea, honestly.",
        "Only a {w} would believe this.",
        "You are being {w} about the whole thing.",
        "This {w} article ignores the facts.",
        "Stop acting like a {w} and read the report.",
    ]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", 200), ("trial", 40), ("test", 60)):
        with open(raw_dir / RAW_FILES[split], "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["spans", "text"])
            for _ in range(n):
                if rng.random() < 0.25:
                    w.writerow(["[]", rng.choice(neutral)])
                    continue
                word = rng.choice(insults)
                text = rng.choice(templates).format(w=word)
                start = text.index(word)
                w.writerow([str(list(range(start, start + len(word)))), text])
    SYNTHETIC_FLAG.write_text("SYNTHETIC DATA - place the real tsd_*.csv files in data/raw/\n")


def download(raw_dir: Path = RAW_DIR) -> str:
    """Return the data source used: 'github', 'huggingface' or 'synthetic'."""
    flag = raw_dir / SYNTHETIC_FLAG.name
    if all((raw_dir / f).exists() for f in RAW_FILES.values()) and not flag.exists():
        print("Raw files already present in", raw_dir)
        return "github"
    print("Downloading SemEval-2021 Task 5 data from GitHub ...")
    if _download_github(raw_dir):
        flag.unlink(missing_ok=True)
        return "github"
    print("Trying Hugging Face mirrors ...")
    if _download_hf(raw_dir):
        flag.unlink(missing_ok=True)
        return "huggingface"
    banner = "!" * 78
    print(
        f"\n{banner}\nWARNING: could not download the real dataset. Writing a TINY SYNTHETIC sample.\n"
        f"Results will be meaningless. Place tsd_train.csv / tsd_trial.csv / tsd_test.csv in {raw_dir}\n{banner}\n"
    )
    _write_synthetic(raw_dir)
    return "synthetic"


# ----------------------------------------------------------------------------- splits
def _with_ids(rows: list[dict], prefix: str) -> list[dict]:
    return [{"id": f"{prefix}-{i:05d}", "text": r["text"], "spans": r["spans"]} for i, r in enumerate(rows)]


def build_splits(raw_dir: Path = RAW_DIR) -> dict[str, list[dict]]:
    train = read_csv(raw_dir / RAW_FILES["train"])
    trial = read_csv(raw_dir / RAW_FILES["trial"]) if (raw_dir / RAW_FILES["trial"]).exists() else []
    test = read_csv(raw_dir / RAW_FILES["test"]) if (raw_dir / RAW_FILES["test"]).exists() else []
    test_labelled = bool(test) and any(r["spans"] for r in test)
    if trial and test_labelled:
        return {"train": _with_ids(train, "train"), "val": _with_ids(trial, "val"), "test": _with_ids(test, "test")}
    # Fallback 2: seeded 80/10/10 split of all labelled data.
    print("Labelled test/trial not available -> seeded 80/10/10 split of the labelled data.")
    pool = train + trial
    random.Random(SEED).shuffle(pool)
    n = len(pool)
    a, b = int(0.8 * n), int(0.9 * n)
    return {"train": _with_ids(pool[:a], "train"), "val": _with_ids(pool[a:b], "val"), "test": _with_ids(pool[b:], "test")}


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(name: str, processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    path = processed_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make data` first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"posts": 0}
    lens = [len(r["text"]) for r in rows]
    toxic = [len(r["spans"]) for r in rows]
    return {
        "posts": n,
        "mean_chars": sum(lens) / n,
        "mean_words": sum(len(WORD_RE.findall(r["text"])) for r in rows) / n,
        "pct_empty_spans": 100 * sum(1 for t in toxic if t == 0) / n,
        "mean_toxic_chars": sum(toxic) / n,
        "pct_toxic_chars": 100 * sum(toxic) / max(1, sum(lens)),
    }


def stats_markdown(stats: dict[str, dict], source: str) -> str:
    lines = [
        "# Dataset statistics",
        "",
        f"Source: **{source}** (SemEval-2021 Task 5, Toxic Spans Detection). "
        "Splits: train = `tsd_train.csv`, val = `tsd_trial.csv`, test = `tsd_test.csv`.",
        "",
        "| Split | Posts | Mean length (chars) | Mean length (words) | % posts with empty spans | Mean toxic chars / post | % of chars toxic |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, s in stats.items():
        lines.append(
            f"| {split} | {s['posts']} | {s['mean_chars']:.1f} | {s['mean_words']:.1f} | "
            f"{s['pct_empty_spans']:.1f} | {s['mean_toxic_chars']:.1f} | {s['pct_toxic_chars']:.1f} |"
        )
    if source == "synthetic":
        lines += ["", "> **WARNING: synthetic placeholder data. Put the real CSVs in `data/raw/` and rerun `make data`.**"]
    return "\n".join(lines) + "\n"


def prepare(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, results_dir: Path = RESULTS_DIR) -> dict:
    source = download(raw_dir)
    splits = build_splits(raw_dir)
    stats = {}
    for name, rows in splits.items():
        write_jsonl(rows, processed_dir / f"{name}.jsonl")
        stats[name] = split_stats(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    md = stats_markdown(stats, source)
    (results_dir / "data_stats.md").write_text(md, encoding="utf-8")
    (results_dir / "data_stats.json").write_text(json.dumps({"source": source, **stats}, indent=2), encoding="utf-8")
    return {"source": source, "stats": stats, "markdown": md}
