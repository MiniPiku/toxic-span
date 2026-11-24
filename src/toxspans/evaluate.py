"""Run every available model on the same test posts and write results/.

Two evaluation sets:
  * full_test  - all test posts (models that are cheap to run: lexicon, transformer)
  * subset     - the first `llm_n` test posts, on which EVERY available model is scored
                 (incl. Claude) for a fair side-by-side.
Writes results/metrics.json, results/results.md, results/f1_comparison.png and
results/predictions_test.json (character indices only, no text).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .baseline import LexiconModel
from .data import load_split
from .llm import ClaudeModel, api_key_available
from .metrics import evaluate
from .transformer import TransformerModel
from .utils import RESULTS_DIR, set_seed

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated reference palette)


def load_models(test_subset: list[dict], include_llm: bool = True) -> tuple[dict, dict]:
    """Return ({key: model}, {key: reason-unavailable})."""
    models, missing = {}, {}
    if LexiconModel.available():
        models["baseline"] = LexiconModel.load()
    else:
        missing["baseline"] = "not trained (run `make baseline`)"
    if TransformerModel.available():
        models["transformer"] = TransformerModel()
    else:
        missing["transformer"] = "not trained (run `make train`)"
    if include_llm:
        train = None
        for shots in (0, 6):
            key = f"claude_{shots}shot"
            if shots and train is None:
                train = load_split("train")
            m = ClaudeModel(shots=shots, train_rows=train, offline=not api_key_available())
            if m.offline and not all(m.cached(r["text"]) for r in test_subset):
                missing[key] = "no ANTHROPIC_API_KEY and no complete cache for the subset (run `make llm`)"
                continue
            models[key] = m
    return models, missing


def run_model(model, rows: list[dict]) -> tuple[list[list[int]], float]:
    """Return (predictions, average latency per post in seconds)."""
    texts = [r["text"] for r in rows]
    if isinstance(model, ClaudeModel):
        metas = [model.predict_with_meta(t) for t in texts]
        return [m["spans"] for m in metas], sum(m["latency"] for m in metas) / max(1, len(metas))
    t0 = time.perf_counter()
    preds = model.predict_batch(texts)
    return preds, (time.perf_counter() - t0) / max(1, len(texts))


def run_all(llm_n: int = 300, results_dir: Path = RESULTS_DIR) -> dict:
    set_seed()
    test = load_split("test")
    subset = test[:llm_n]
    models, missing = load_models(subset)
    out = {"test_posts": len(test), "subset_posts": len(subset), "models": {}, "unavailable": missing}
    predictions = {}
    for key, model in models.items():
        entry = {"name": model.name}
        is_llm = isinstance(model, ClaudeModel)
        if not is_llm:
            preds, lat = run_model(model, test)
            predictions[key] = preds
            entry["full_test"] = {**evaluate(preds, [r["spans"] for r in test]), "avg_latency_s": lat}
            sub_preds = preds[: len(subset)]
            entry["subset"] = {**evaluate(sub_preds, [r["spans"] for r in subset]), "avg_latency_s": lat}
        else:
            preds, lat = run_model(model, subset)
            predictions[key] = preds
            entry["subset"] = {**evaluate(preds, [r["spans"] for r in subset]), "avg_latency_s": lat,
                               "json_failures": model.failures}
        if key == "transformer":
            entry["config"] = {k: model.config[k] for k in ("base_model", "threshold", "fill_whitespace",
                                                             "train_samples", "epochs", "device")}
        elif key == "baseline":
            entry["config"] = {"t": model.t, "c": model.c, "lexicon_size": len(model.lexicon)}
        out["models"][key] = entry
        print(f"{model.name:45s} " + "  ".join(
            f"{s}: F1={entry[s]['f1']:.4f} (n={entry[s]['n']})" for s in ("full_test", "subset") if s in entry))

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (results_dir / "predictions_test.json").write_text(json.dumps(predictions), encoding="utf-8")
    (results_dir / "results.md").write_text(results_markdown(out), encoding="utf-8")
    plot(out, results_dir / "f1_comparison.png")
    return out


def _fmt_latency(s: float) -> str:
    return f"{s * 1000:.2f} ms" if s < 1 else f"{s:.2f} s"


def results_markdown(out: dict) -> str:
    lines = ["# Results", "",
             "F1 = official SemEval-2021 Task 5 metric (mean per-post character F1). "
             "Precision / Recall = micro-averaged over all characters. "
             "Latency for local models is amortised batch inference on this machine's CPU; "
             "for Claude it is the measured API round-trip of the original (uncached) call.", ""]
    for split, title in (("full_test", f"Full test set ({out['test_posts']} posts)"),
                         ("subset", f"Same-subset comparison: first {out['subset_posts']} test posts")):
        rows = [(e["name"], e[split]) for e in out["models"].values() if split in e]
        if not rows:
            continue
        lines += [f"## {title}", "", "| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |",
                  "|---|---|---|---|---|---|"]
        for name, m in sorted(rows, key=lambda x: -x[1]["f1"]):
            lines.append(f"| {name} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                         f"{_fmt_latency(m['avg_latency_s'])} | {m['n']} |")
        lines.append("")
    if out["unavailable"]:
        lines += ["## Not evaluated", ""]
        lines += [f"- **{k}**: {v}" for k, v in out["unavailable"].items()]
        lines.append("")
    return "\n".join(lines)


def plot(out: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [e["name"] for e in out["models"].values()]
    full = [e.get("full_test", {}).get("f1") for e in out["models"].values()]
    sub = [e["subset"]["f1"] for e in out["models"].values()]
    y = list(range(len(names)))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.8 * len(names)))
    full_y = [yy + h / 2 for yy, f in zip(y, full) if f is not None]
    full_v = [f for f in full if f is not None]
    ax.barh(full_y, full_v, height=h - 0.04, color=BLUE, label=f"Full test (n={out['test_posts']})")
    ax.barh([yy - h / 2 for yy in y], sub, height=h - 0.04, color=ORANGE, label=f"First {out['subset_posts']} test posts")
    for yy, v in zip(full_y, full_v):
        ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
    for yy, v in zip(y, sub):
        ax.text(v + 0.01, yy - h / 2, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(y, names, fontsize=9, color="#0b0b0b")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (mean per-post character F1)", color="#52514e")
    ax.set_title("Toxic span detection - F1 by model", loc="left", fontsize=11, color="#0b0b0b", pad=26)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(axis="x", colors="#52514e")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
