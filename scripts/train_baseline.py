"""Build the lexicon from train, tune (t, c) on val, report val/test F1, save models/baseline/lexicon.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxspans.baseline import DEFAULT_DIR, LexiconModel  # noqa: E402
from toxspans.data import load_split  # noqa: E402
from toxspans.utils import set_seed  # noqa: E402


def main() -> None:
    set_seed()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    model, grid = LexiconModel.fit(train, val)
    path = model.save()
    (DEFAULT_DIR / "tuning_grid.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    v, t = model.score(val), model.score(test)
    print(f"Saved lexicon ({len(model.lexicon)} words, t={model.t}, c={model.c}) -> {path}")
    print(f"VAL  F1={v['f1']:.4f}  P={v['precision']:.4f}  R={v['recall']:.4f}  (n={v['n']})")
    print(f"TEST F1={t['f1']:.4f}  P={t['precision']:.4f}  R={t['recall']:.4f}  (n={t['n']})")


if __name__ == "__main__":
    main()
