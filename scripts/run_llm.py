"""Run the LLM span predictor (Gemini or Claude) on the first --n test posts (cost-bounded). Skips cleanly without a key."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxspans.data import load_split  # noqa: E402
from toxspans.llm import LLMModel, QuotaExhausted, api_key_available, display_name, model_name, provider  # noqa: E402
from toxspans.metrics import evaluate  # noqa: E402
from toxspans.utils import RESULTS_DIR, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, default=0, help="0 = zero-shot; e.g. 6 = few-shot from train (seed 42)")
    ap.add_argument("--n", type=int, default=300, help="evaluate on the first n test posts")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    set_seed()

    rows = load_split(args.split)[: args.n]
    probe = LLMModel(shots=args.shots, train_rows=load_split("train") if args.shots else None, offline=True)
    all_cached = all(probe.cached(r["text"]) for r in rows)
    if not api_key_available() and not all_cached:
        var = "GEMINI_API_KEY" if provider() == "gemini" else "ANTHROPIC_API_KEY"
        print(f"{var} is not set (env or .env) -> skipping the LLM step. "
              "Copy .env.example to .env, add a key and rerun `make llm`.")
        return

    model = LLMModel(shots=args.shots, train_rows=load_split("train") if args.shots else None)
    print(f"Provider {display_name()} | model {model_name()} | shots={args.shots} | {args.split} posts={len(rows)}")
    preds, lat, t0 = [], [], time.time()
    for i, r in enumerate(rows, 1):
        try:
            out = model.predict_with_meta(r["text"])
        except QuotaExhausted as e:
            print(f"Stopping: {e}. {i - 1}/{len(rows)} posts are cached in data/cache/llm_cache.jsonl; "
                  "rerun `make llm` after the quota resets to finish (cached posts are not re-sent). "
                  "`make evaluate` scores the finished posts and marks the run as partial.")
            return
        preds.append(out["spans"])
        lat.append(out["latency"])
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  api_calls={model.api_calls}  json_failures={model.failures}  ({time.time() - t0:.0f}s)")
    res = evaluate(preds, [r["spans"] for r in rows])
    res.update({"json_failures": model.failures, "api_calls": model.api_calls, "model": model.model, "provider": model.provider,
                "shots": args.shots, "avg_latency_s": sum(lat) / len(lat)})
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"llm_{args.split}_shots{args.shots}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{args.split.upper()} F1={res['f1']:.4f}  P={res['precision']:.4f}  R={res['recall']:.4f}  "
          f"(n={res['n']}, json failures={model.failures})")


if __name__ == "__main__":
    main()
