"""Write results/error_analysis.md for the best model (needs `make evaluate` outputs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxspans.error_analysis import run  # noqa: E402

if __name__ == "__main__":
    md = run()
    print(md.split("## Observations")[0][:1500])
    print("Wrote results/error_analysis.md")
