"""Fine-tune a RoBERTa-family token classifier for toxic spans; prints val and test F1."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxspans.data import load_split  # noqa: E402
from toxspans.transformer import DEFAULT_DIR, TransformerModel, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default=None, help="default: roberta-base on GPU, distilroberta-base on CPU")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=None, help="default: 16 on GPU, 8 on CPU")
    ap.add_argument("--max-train-samples", type=int, default=None, help="subsample train (CPU-friendly)")
    ap.add_argument("--pos-weight", type=float, default=1.0, help="class weight for the toxic label")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out-dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()

    train_rows, val_rows, test_rows = load_split("train"), load_split("val"), load_split("test")
    cfg = train(
        train_rows, val_rows, model_name=args.model_name, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, max_train_samples=args.max_train_samples, pos_weight=args.pos_weight,
        device=None if args.device == "auto" else args.device, out_dir=Path(args.out_dir), max_length=args.max_length,
    )
    model = TransformerModel(Path(args.out_dir), device=cfg["device"])
    v, t = model.score(val_rows), model.score(test_rows)
    cfg["test_f1"] = t["f1"]
    Path(args.out_dir, "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved model -> {args.out_dir}  (threshold={cfg['threshold']}, fill_whitespace={cfg['fill_whitespace']})")
    print(f"VAL  F1={v['f1']:.4f}  P={v['precision']:.4f}  R={v['recall']:.4f}  (n={v['n']})")
    print(f"TEST F1={t['f1']:.4f}  P={t['precision']:.4f}  R={t['recall']:.4f}  (n={t['n']})")


if __name__ == "__main__":
    main()
