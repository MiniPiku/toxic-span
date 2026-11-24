# Decisions log

One line per decision or assumption, in the order they were made.

- Project root is `final-yr-proj/toxic-spans/` to match the requested layout exactly.
- No GNU `make` on this Windows machine; the Makefile is GNU-make compatible and was run with `mingw32-make` (uses Git's `sh`), documented in USAGE.md.
- Python 3.11 is used for the venv (3.10+ required; 3.13 avoided for wheel availability of torch).
- The package is installed editable (`pip install -e .`, via a minimal `pyproject.toml`) so scripts, tests and the app import `toxspans` cleanly.
- Added `requests` (download) and `nbformat` (notebook generation) to requirements on top of the requested list.
- Data: real files found at `SemEval2021/data/` (not `data/`) in `ipavlopoulos/toxic_spans`; columns are `spans,text`.
- Splits: train = tsd_train (7939), val = tsd_trial (690), test = tsd_test (2000); the official test file is labelled, so the 80/10/10 fallback was not needed.
- Added `.gitattributes` (LF line endings) so the Makefile and scripts work on Linux/Colab after a Windows commit.
- torch installed as the CPU wheel (no usable GPU from this Windows account); all training below ran on CPU.
- Lexicon baseline: a word counts as a toxic occurrence when at least half its characters are in gold spans; grid t in 0.2..0.9, c in {1,2,3,5,10}; best on val t=0.3, c=5 (410 words).
- Transformer: plain PyTorch loop (simpler to read than `Trainer`), AdamW + linear warmup (10%), grad clip 1.0, CE loss with optional `--pos-weight` (default 1.0).
- Transformer: HF weights saved to `models/transformer/hf/` so our own `models/transformer/config.json` (threshold, fill flag, scores) does not clash with the HF `config.json`.
- Transformer: tokenizer loaded with `add_prefix_space=True` so the first word is tokenized like every other word.
- Transformer: best epoch chosen by val F1 at threshold 0.5; then threshold (0.10-0.90, step 0.05) x whitespace-fill swept on val and the best pair stored.
- Transformer: length-grouped batching (train) and length-sorted inference added after measuring ~2.7 s/step on this CPU; it removes padding waste and keeps results seeded.
- CPU default for `make train`: distilroberta-base, `--max-train-samples 3000`, `--epochs 2` (19.5 min measured here; OneDrive sync was competing for CPU); the script default stays 3 epochs on the full set, as used on Colab GPU.
- LLM: default model `claude-haiku-5-5` as requested; `temperature` is NOT sent because Claude Haiku 5.5 rejects non-default sampling parameters (400). Determinism comes from the response cache instead.
- LLM: `output_config.effort = "low"` (simple extraction task); only `text` blocks of the response are read (thinking blocks ignored).
- LLM: SDK auto-retry disabled; own exponential backoff (2s x 2^attempt + jitter, 6 retries) on rate-limit, connection and 5xx errors.
- LLM: few-shot = 6 seeded examples from train posts of <= 300 chars, 4 with spans + 2 with empty spans (teaches the model that "none" is allowed).
- LLM: returned substrings mapped by exact search, falling back to case-insensitive search; every occurrence is marked (the model returns strings, not positions).
- LLM SKIPPED: no ANTHROPIC_API_KEY in env/.env and no `ant` CLI credentials on this machine; `make llm` prints a message and exits 0. Claude rows are absent from results until a key is added.
- Evaluation: headline F1 = mean per-post F1; the P/R columns are micro-averaged over characters (as required); latency for local models = amortised batch inference time per post on CPU.
- Evaluation: local models are scored on the full test set (2000) and on the first 300 test posts; Claude only on the first `--n` posts, so the same-subset table is the fair side-by-side.
- Error analysis: word-level FP/FN (a word is "marked" if >= half its chars are); function words excluded from the top-10 tables (reported as a count); words are masked only if their train toxic ratio >= 0.1 or unseen.
- App: radio lists only available models; unavailable ones are listed with the reason. Tested end-to-end with Streamlit's `AppTest` (baseline path).
- Final verification ran in a fresh `git clone` (new venv, `make setup` then `make all`) in a temp folder outside OneDrive; training losses and F1 matched the original run exactly.

## Build checklist

- [x] Phase 0: scaffold, venv, requirements, Makefile, .gitignore, .env.example (`make setup` from a clean clone)
- [x] Phase 1: real data downloaded from GitHub, processed splits, `results/data_stats.md`
- [x] Phase 2: official metric + tests (`make test`)
- [x] Phase 3: lexicon baseline, tuned on val, saved
- [x] Phase 4: transformer trained on CPU, threshold + whitespace fill tuned, Colab notebook, offset test
- [x] Phase 5: LLM predictor (Gemini free tier + Claude) with cache/retries/mocked tests; Gemini zero-shot 300/300, 6-shot 184/300 (daily quota)
- [x] Phase 6: `make evaluate` -> metrics.json, results.md, f1_comparison.png, error_analysis.md
- [x] Phase 7: Streamlit demo, health check `ok`, baseline + transformer paths tested
- [x] Phase 8: README, OVERVIEW, USAGE
- [x] Phase 9: fresh install, `make test`, `make all`, health check, no secrets/data tracked, OVERVIEW numbers match metrics.json

## Switch to Gemini (user request, after the first build)

- LLM approach now runs on the Google Gemini API free tier at the user's request; the Claude path is kept and selectable with `LLM_PROVIDER=anthropic`.
- Provider selection: env `LLM_PROVIDER` (`gemini` / `anthropic`); if unset, Gemini when `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set, else Anthropic.
- Gemini model `gemini-3.5-flash-lite` (override with `GEMINI_MODEL`): `gemini-2.5-flash(-lite)` return 404 for new keys and `gemini-3.8-flash` / `gemini-3.1-flash-lite` were returning 503 (overloaded) when probed.
- Gemini calls use temperature 0 and JSON response mode (`response_mime_type=application/json`); the same robust parser is still applied.
- Free-tier throttle: at least `GEMINI_MIN_INTERVAL` = 4.5 s between calls (~13 requests/min); 429/5xx retried with the server's suggested delay or exponential backoff (max 6 retries).
- Reported LLM latency is the API request time only (throttle waits excluded).
- Result keys renamed `claude_*shot` -> `llm_*shot`; SDK `google-genai` added to requirements.
- Security note: the Gemini key was shared in chat; it lives only in the git-ignored `.env`. Rotating it in Google AI Studio after the project is graded is advisable.
- Gemini free tier allows 500 requests/day per model (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`): zero-shot finished all 300 posts, 6-shot stopped at 184/300.
- A per-day 429 now raises `QuotaExhausted`: `make llm` stops at once with a message and exits 0 (so `make all` keeps working); cached posts are never re-sent.
- `make evaluate` never calls the LLM API; it scores cached responses. An unfinished run is marked *partial*, and an extra table scores every model on the posts all LLM runs covered (first 184), which is also the second bar series in the chart.
- Error analysis gained a same-posts table with the LLM (F1 on posts with/without spans, clean posts flagged, predicted chars per post).
- To finish the 6-shot run: rerun `make llm` then `make evaluate` after the daily quota resets; the OVERVIEW numbers then need refreshing.
