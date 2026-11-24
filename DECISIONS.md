# Decisions log

One line per decision or assumption, in the order they were made.

- Project root is `final-yr-proj/toxic-spans/` to match the requested layout exactly.
- No GNU `make` on this Windows machine; the Makefile is GNU-make compatible and was run with `mingw32-make` (uses Git's `sh`), documented in USAGE.md.
- Python 3.11 is used for the venv (3.10+ required; 3.13 avoided for wheel availability of torch).
- The package is installed editable (`pip install -e .`, via a minimal `pyproject.toml`) so scripts, tests and the app import `toxspans` cleanly.
- Added `requests` (download) and `nbformat` (notebook generation) to requirements on top of the requested list.
