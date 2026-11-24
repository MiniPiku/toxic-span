"""Evaluate every available model on the same test posts -> results/metrics.json, results.md, f1_comparison.png."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxspans.evaluate import run_all  # noqa: E402
from toxspans.utils import RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-n", type=int, default=300, help="size of the same-subset comparison (first n test posts)")
    args = ap.parse_args()
    out = run_all(llm_n=args.llm_n)
    for k, why in out["unavailable"].items():
        print(f"skipped {k}: {why}")
    print((RESULTS_DIR / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
