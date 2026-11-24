"""Generate notebooks/colab_train.ipynb (same training as `make train`, on a free Colab GPU)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

cells = [
    nbf.v4.new_markdown_cell(
        "# Toxic Span Detector - train the transformer on a Colab GPU\n\n"
        "Runs exactly the same code as `make train`, but on a GPU, so the default model becomes `roberta-base` "
        "and the full training set is used.\n\n"
        "1. *Runtime -> Change runtime type -> T4 GPU*.\n"
        "2. Set `REPO_URL` below **or** leave it empty and upload a zip of the project folder (`toxic-spans.zip`).\n"
        "3. *Runtime -> Run all*. At the end the trained model folder is downloaded as `transformer_model.zip`; "
        "unzip it into `models/` on your machine (so you get `models/transformer/config.json` and `models/transformer/hf/`)."
    ),
    nbf.v4.new_code_cell("!nvidia-smi -L || echo 'No GPU - switch the runtime to GPU for fast training'"),
    nbf.v4.new_code_cell(
        "REPO_URL = ''  # e.g. 'https://github.com/<you>/toxic-spans.git'\n"
        "import os\n"
        "if REPO_URL:\n"
        "    !git clone -q $REPO_URL toxic-spans\n"
        "else:\n"
        "    from google.colab import files\n"
        "    uploaded = files.upload()  # choose toxic-spans.zip\n"
        "    !unzip -q -o toxic-spans.zip\n"
        "%cd toxic-spans"
    ),
    nbf.v4.new_code_cell("!pip install -q -r requirements.txt && pip install -q -e ."),
    nbf.v4.new_code_cell("!python scripts/download_data.py"),
    nbf.v4.new_code_cell("!python scripts/train_baseline.py"),
    nbf.v4.new_code_cell(
        "# Full training set, 3 epochs; device auto-detected (roberta-base on GPU).\n"
        "!python scripts/train_transformer.py --epochs 3"
    ),
    nbf.v4.new_code_cell("!python scripts/evaluate.py --llm-n 300 && python scripts/error_analysis.py"),
    nbf.v4.new_code_cell(
        "!cd models && zip -q -r ../transformer_model.zip transformer\n"
        "from google.colab import files\n"
        "files.download('transformer_model.zip')"
    ),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": []},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
})
out = ROOT / "notebooks" / "colab_train.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
