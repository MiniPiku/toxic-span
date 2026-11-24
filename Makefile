# Toxic Span Detector - GNU make (on Windows use `mingw32-make` or `make` from Git/MSYS).
ifeq ($(OS),Windows_NT)
PY_BOOT ?= python
PY := .venv/Scripts/python.exe
else
PY_BOOT ?= python3
PY := .venv/bin/python
endif

# Tunables (override like: make train MAX_TRAIN=4000 EPOCHS=2)
MAX_TRAIN ?= 3000
EPOCHS ?= 2
LLM_N ?= 300
SHOTS ?= 0

.PHONY: setup data baseline train llm evaluate app test all clean

setup:
	$(PY_BOOT) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

data:
	$(PY) scripts/download_data.py

baseline:
	$(PY) scripts/train_baseline.py

train:
	$(PY) scripts/train_transformer.py --max-train-samples $(MAX_TRAIN) --epochs $(EPOCHS)

llm:
	$(PY) scripts/run_llm.py --shots 0 --n $(LLM_N)
	$(PY) scripts/run_llm.py --shots 6 --n $(LLM_N)

evaluate:
	$(PY) scripts/evaluate.py --llm-n $(LLM_N)
	$(PY) scripts/error_analysis.py

app:
	$(PY) -m streamlit run app/app.py

test:
	$(PY) -m pytest -q

all: data baseline train llm evaluate test

clean:
	rm -rf results/*.png results/metrics.json results/results.md results/error_analysis.md
