"""Download SemEval-2021 Task 5 data -> data/processed/{train,val,test}.jsonl + results/data_stats.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxspans.data import prepare  # noqa: E402
from toxspans.utils import set_seed  # noqa: E402


def main() -> None:
    set_seed()
    out = prepare()
    print(out["markdown"])
    if out["source"] == "synthetic":
        print("!!! SYNTHETIC DATA IN USE - put the real tsd_*.csv files in data/raw/ and rerun `make data` !!!")


if __name__ == "__main__":
    main()
