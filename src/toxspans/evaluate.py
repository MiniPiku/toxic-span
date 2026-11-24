"""Run every available model on the same test posts and write results/.

Two evaluation sets:
  * full_test  - all test posts (models that are cheap to run: lexicon, transformer)
  * subset     - the first `llm_n` test posts, on which EVERY available model is scored
                 (incl. the LLM) for a fair side-by-side.
Writes results/metrics.json, results/results.md, results/f1_comparison.png and
results/predictions_test.json (character indices only, no text).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .baseline import LexiconModel
from .data import load_split
from .llm import LLMModel, api_key_available, provider
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
            key = f"llm_{shots}shot"
            if shots and train is None:
                train = load_split("train")
            # Evaluation never calls the API: it scores the cached responses of `make llm`.
            m = LLMModel(shots=shots, train_rows=train, offline=True)
            covered = 0
            while covered < len(test_subset) and m.cached(test_subset[covered]["text"]):
                covered += 1
            if covered == 0:
                var = "GEMINI_API_KEY" if provider() == "gemini" else "ANTHROPIC_API_KEY"
                missing[key] = f"no cached responses for {m.model} (set {var} and run `make llm`)"
                continue
            m.covered = covered
            models[key] = m
    return models, missing


def run_model(model, rows: list[dict]) -> tuple[list[list[int]], float]:
    """Return (predictions, average latency per post in seconds)."""
    texts = [r["text"] for r in rows]
    if isinstance(model, LLMModel):
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
        is_llm = isinstance(model, LLMModel)
        if not is_llm:
            preds, lat = run_model(model, test)
            predictions[key] = preds
            entry["full_test"] = {**evaluate(preds, [r["spans"] for r in test]), "avg_latency_s": lat}
            sub_preds = preds[: len(subset)]
            entry["subset"] = {**evaluate(sub_preds, [r["spans"] for r in subset]), "avg_latency_s": lat}
        else:
            rows = subset[: model.covered]
            preds, lat = run_model(model, rows)
            predictions[key] = preds
            entry["subset"] = {**evaluate(preds, [r["spans"] for r in rows]), "avg_latency_s": lat,
                               "json_failures": model.failures}
            entry["provider"], entry["model"] = model.provider, model.model
            if model.covered < len(subset):
                entry["partial"] = True
                entry["name"] += f" [partial: {model.covered}/{len(subset)} posts]"
        if key == "transformer":
            entry["config"] = {k: model.config[k] for k in ("base_model", "threshold", "fill_whitespace",
                                                             "train_samples", "epochs", "device")}
        elif key == "baseline":
            entry["config"] = {"t": model.t, "c": model.c, "lexicon_size": len(model.lexicon)}
        out["models"][key] = entry
        print(f"{model.name:45s} " + "  ".join(
            f"{s}: F1={entry[s]['f1']:.4f} (n={entry[s]['n']})" for s in ("full_test", "subset") if s in entry))

    # Extra fair comparisons on the first n posts covered by every partial LLM run.
    extra = sorted({len(predictions[k]) for k, e in out["models"].items() if e.get("partial")})
    out["extra_comparisons"] = extra
    for n in extra:
        gold = [r["spans"] for r in test[:n]]
        for k, e in out["models"].items():
            if len(predictions[k]) >= n:
                base = e["full_test"] if "full_test" in e else e["subset"]
                e[f"first_{n}"] = {**evaluate(predictions[k][:n], gold), "avg_latency_s": base["avg_latency_s"]}

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
             "for the LLM it is the measured API round-trip of the original (uncached) call.", ""]
    tables = [("full_test", f"Full test set ({out['test_posts']} posts)"),
              ("subset", f"Same-subset comparison: first {out['subset_posts']} test posts")]
    tables += [(f"first_{n}", f"All models on the first {n} test posts (covered by every LLM run)")
               for n in out.get("extra_comparisons", [])]
    for split, title in tables:
        rows = [(e["name"], e[split]) for e in out["models"].values() if split in e]
        if not rows:
            continue
        lines += [f"## {title}", "", "| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |",
                  "|---|---|---|---|---|---|"]
        for name, m in sorted(rows, key=lambda x: -x[1]["f1"]):
            lines.append(f"| {name} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                         f"{_fmt_latency(m['avg_latency_s'])} | {m['n']} |")
        if split == "subset" and out.get("extra_comparisons"):
            lines += ["", "Rows marked *partial* did not finish (free-tier daily request quota); their F1 is on fewer "
                      "posts, so compare them only in the next table, where every model is scored on the same posts."]
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

    import textwrap

    names = [textwrap.fill(e["name"].replace("Gemini gemini-", "Gemini ").replace(" posts]", "]"), 30, break_on_hyphens=False)
             for e in out["models"].values()]
    full = [e.get("full_test", {}).get("f1") for e in out["models"].values()]
    common = f"first_{out['extra_comparisons'][0]}" if out.get("extra_comparisons") else "subset"
    common_n = out["extra_comparisons"][0] if out.get("extra_comparisons") else out["subset_posts"]
    sub = [e[common]["f1"] for e in out["models"].values()]
    y = list(range(len(names)))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.85 * len(names)))
    full_y = [yy + h / 2 for yy, f in zip(y, full) if f is not None]
    full_v = [f for f in full if f is not None]
    ax.barh(full_y, full_v, height=h - 0.04, color=BLUE, label=f"Full test (n={out['test_posts']})")
    ax.barh([yy - h / 2 for yy in y], sub, height=h - 0.04, color=ORANGE, label=f"First {common_n} test posts (all models)")
    for yy, v in zip(full_y, full_v):
        ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
    for yy, v in zip(y, sub):
        ax.text(v + 0.01, yy - h / 2, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(y, names, fontsize=9, color="#0b0b0b")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (mean per-post character F1)", color="#52514e")
    fig.suptitle("Toxic span detection - F1 by model", x=0.01, ha="left", fontsize=11, color="#0b0b0b")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(axis="x", colors="#52514e")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=2, frameon=False, fontsize=9)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)
