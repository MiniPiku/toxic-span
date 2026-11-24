# Toxic Span Detector - Usage

How to install, run every stage, launch the demo and test the project. Every command below was run
successfully while building the project (Windows 11, Python 3.11, CPU only).

## 1. Prerequisites

- Python 3.10 or newer (`python --version`).
- GNU make. Linux/macOS have it. On Windows use `mingw32-make` (MinGW) or `make` from Git for Windows/MSYS2.
  Every target is a thin wrapper, so you can also run the `python scripts/...` commands shown below directly.
- Internet access for the first run (dataset from GitHub, `distilroberta-base` / `roberta-base` from Hugging Face).
- Optional, for the LLM approach: a Google Gemini API key (free tier, from Google AI Studio) or an Anthropic API key.
- Disk: about 3 GB (PyTorch, models, cache).

## 2. Setup

```bash
make setup            # creates .venv, installs requirements.txt and the package (pip install -e .)
```

On Windows with several Pythons installed, pick 3.11 explicitly:

```powershell
mingw32-make setup PY_BOOT="py -3.11"
```

Optional, for the LLM approach:

```bash
cp .env.example .env  # then set GEMINI_API_KEY=... (free tier), or LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=...
```

## 3. Pipeline stages

| Target | What it does | Direct command |
|---|---|---|
| `make data` | Downloads `tsd_train/trial/test.csv`, writes `data/processed/{train,val,test}.jsonl` and `results/data_stats.md` | `python scripts/download_data.py` |
| `make baseline` | Builds the lexicon, tunes `t`/`c` on val, prints val/test F1, saves `models/baseline/lexicon.json` | `python scripts/train_baseline.py` |
| `make train` | Fine-tunes the transformer (CPU-friendly defaults), tunes threshold + whitespace fill, prints val/test F1 | `python scripts/train_transformer.py --max-train-samples 3000 --epochs 2` |
| `make llm` | LLM (Gemini by default, or Claude) zero-shot and 6-shot on the first 300 test posts (skips cleanly without a key) | `python scripts/run_llm.py --shots 0 --n 300` |
| `make evaluate` | Scores every available model on the same posts, writes all files in `results/`, then the error analysis | `python scripts/evaluate.py --llm-n 300` and `python scripts/error_analysis.py` |
| `make test` | Runs the pytest suite | `python -m pytest -q` |
| `make app` | Starts the Streamlit demo | `python -m streamlit run app/app.py` |
| `make all` | `data baseline train llm evaluate test` in order | |

On Windows replace `python` with `.venv\Scripts\python.exe` (or activate the venv first), and `make` with `mingw32-make`.

### Flags

`make` variables: `MAX_TRAIN` (default 3000), `EPOCHS` (default 2), `LLM_N` (default 300), for example
`make train MAX_TRAIN=8000 EPOCHS=3`.

`scripts/train_transformer.py`

| Flag | Default | Meaning |
|---|---|---|
| `--model-name` | `roberta-base` on GPU, `distilroberta-base` on CPU | any Hugging Face RoBERTa-family checkpoint |
| `--epochs` | 3 | training epochs (best epoch kept by val F1) |
| `--max-train-samples` | all (7939) | seeded subsample of train, for CPU runs |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `--lr` | 3e-5 | learning rate |
| `--batch-size` | 16 on GPU, 8 on CPU | batch size |
| `--pos-weight` | 1.0 | loss weight of the toxic class (class imbalance) |
| `--max-length` | 256 | max tokens; longer posts are truncated (count is logged) |
| `--out-dir` | `models/transformer` | output folder |

`scripts/run_llm.py`: `--shots` (0 = zero-shot, 6 = few-shot), `--n` (first n test posts, default 300), `--split` (`test` or `val`).

`scripts/evaluate.py`: `--llm-n` (size of the same-subset comparison, default 300).

Approximate CPU timings on the build machine (8 threads): data 5 s, baseline 10 s, `make train` defaults about 20 min (19.5 min measured),
evaluate about 1-2 min, tests about 30 s, `make llm` on the Gemini free tier about 45 min (600 throttled calls; cached afterwards).

## 4. Demo app

```bash
make app
# or headless, e.g. on a server:
python -m streamlit run app/app.py --server.headless true
```

Open http://localhost:8501. Pick a model in the sidebar (only trained / configured models are listed; the others show why
they are disabled), type a comment or click one of the three example buttons, then press **Detect**. The app shows the text
with toxic spans highlighted in red, the list of spans with character offsets, and the inference time.

Health check (while the app runs):

```bash
curl localhost:8501/_stcore/health   # -> ok
```

## 5. Tests

```bash
make test
```

Covers the metric edge cases, span parsing, index <-> range conversion, token-offset -> character mapping (with a real
RoBERTa tokenizer on a sentence with leading spaces and punctuation), the Claude predictor with a mocked client (JSON
parsing, offset mapping, cache), and the Streamlit app end to end with the baseline model. The tokenizer test is skipped if
the tokenizer cannot be downloaded; the app tests are skipped until `make baseline` has run.

## 6. Training on Colab (free GPU)

Open `notebooks/colab_train.ipynb` in Google Colab, choose a GPU runtime, set `REPO_URL` (or upload a zip of this folder)
and run all cells. It installs the requirements, downloads the data, trains `roberta-base` on the full training set for
3 epochs with the same script, evaluates, and downloads `transformer_model.zip`. Unzip it into `models/` locally so that
`models/transformer/config.json` exists, then run `make evaluate` and `make app`.
The notebook is generated by `python scripts/make_colab_notebook.py`.

## 7. Environment variables

| Variable | Default | Used by |
|---|---|---|
| `LLM_PROVIDER` | auto | `gemini` or `anthropic`; auto = Gemini if a Gemini key is set, else Anthropic |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | unset | Gemini approach (env or `.env`) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model name |
| `GEMINI_MIN_INTERVAL` | `4.5` | seconds between Gemini calls (free-tier rate limit) |
| `ANTHROPIC_API_KEY` | unset | Claude approach (env or `.env`) |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Claude model name |

Without a key for the selected provider the LLM step is skipped and the other models still run.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `make: command not found` on Windows | use `mingw32-make`, or run the `python scripts/...` commands directly |
| `data/processed/train.jsonl not found` | run `make data` first |
| Output says SYNTHETIC DATA | the download failed; put `tsd_train.csv`, `tsd_trial.csv`, `tsd_test.csv` from github.com/ipavlopoulos/toxic_spans (`SemEval2021/data/`) into `data/raw/`, delete `data/raw/SYNTHETIC_DATA_WARNING.txt`, rerun `make data` |
| Training is too slow on CPU | lower `MAX_TRAIN`/`EPOCHS`, or use the Colab notebook |
| `LLM step skipped` | add `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY` with `LLM_PROVIDER=anthropic`) to `.env`; responses are cached in `data/cache/llm_cache.jsonl`, so reruns are free |
| Rate-limit (429) errors | the client waits and retries; raise `GEMINI_MIN_INTERVAL`, or rerun later: cached posts are skipped. A daily free-tier quota resets the next day |
| Gemini `404 model no longer available` / `503 high demand` | set `GEMINI_MODEL` to another model your key lists (e.g. `gemini-3.5-flash-lite`) |
| Transformer missing in the app | run `make train` (or copy a Colab-trained `models/transformer/`) |
| Port 8501 in use | `python -m streamlit run app/app.py --server.port 8502` |

## 9. Expected output files

| File | Produced by |
|---|---|
| `data/raw/tsd_{train,trial,test}.csv` | `make data` |
| `data/processed/{train,val,test}.jsonl` | `make data` |
| `results/data_stats.md`, `results/data_stats.json` | `make data` |
| `models/baseline/lexicon.json`, `models/baseline/tuning_grid.json` | `make baseline` |
| `models/transformer/config.json`, `models/transformer/threshold_sweep.json`, `models/transformer/hf/` | `make train` |
| `data/cache/llm_cache.jsonl`, `results/llm_test_shots{0,6}.json` | `make llm` (with a key) |
| `results/metrics.json`, `results/results.md`, `results/f1_comparison.png`, `results/predictions_test.json` | `make evaluate` |
| `results/error_analysis.md` | `make evaluate` |
