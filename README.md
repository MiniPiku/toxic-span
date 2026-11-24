# Toxic Span Detector

Finds the exact characters that make a comment toxic (SemEval-2021 Task 5, Toxic Spans Detection), with three
approaches compared on the official metric: a data-driven lexicon, a fine-tuned RoBERTa-family token classifier,
and Claude (zero-shot / few-shot). A Streamlit demo highlights the toxic words.

- **[OVERVIEW.md](OVERVIEW.md)**: problem, data, approaches, architecture, results, error analysis, ethics.
- **[USAGE.md](USAGE.md)**: setup, every command and flag, the demo app, tests, Colab training, troubleshooting.
- **[DECISIONS.md](DECISIONS.md)**: one-line log of every design decision and assumption.

Quick start: `make setup && make all && make app` (on Windows: `mingw32-make`).
