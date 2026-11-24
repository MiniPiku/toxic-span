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
