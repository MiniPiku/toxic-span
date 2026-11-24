"""Streamlit demo: highlight toxic spans with the lexicon baseline, the transformer, or Claude."""
from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxspans.baseline import LexiconModel  # noqa: E402
from toxspans.utils import indices_to_ranges  # noqa: E402

st.set_page_config(page_title="Toxic Span Detector", page_icon=":mag:", layout="centered")

EXAMPLES = {
    "Mild insult": "What a stupid idea. Only a clown would vote for this.",
    "Polite disagreement": "I respectfully disagree with the council; the numbers in the report do not add up.",
    "Mixed": "Thanks for the article, but the author is a pathetic loser who never checks facts.",
}


# ----------------------------------------------------------------------------- model loading (cached)
@st.cache_resource(show_spinner="Loading lexicon ...")
def load_baseline():
    return LexiconModel.load()


@st.cache_resource(show_spinner="Loading transformer ...")
def load_transformer():
    from toxspans.transformer import TransformerModel

    return TransformerModel()


@st.cache_resource(show_spinner=False)
def load_claude(shots: int):
    from toxspans.data import load_split
    from toxspans.llm import ClaudeModel

    return ClaudeModel(shots=shots, train_rows=load_split("train") if shots else None)


def model_status() -> dict[str, tuple[bool, str]]:
    from toxspans.llm import api_key_available, model_name
    from toxspans.transformer import TransformerModel

    status = {
        "Baseline (lexicon)": (LexiconModel.available(), "Not trained yet - run `make baseline`."),
        "Transformer": (TransformerModel.available(), "Not trained yet - run `make train`."),
    }
    has_key = api_key_available()
    train_ok = Path(__file__).resolve().parents[1].joinpath("data", "processed", "train.jsonl").exists()
    status[f"Claude zero-shot ({model_name()})"] = (has_key, "No ANTHROPIC_API_KEY in the environment or .env.")
    status[f"Claude 6-shot ({model_name()})"] = (
        has_key and train_ok, "Needs ANTHROPIC_API_KEY and the training data (`make data`).")
    return status


def predict(choice: str, text: str) -> list[int]:
    if choice.startswith("Baseline"):
        return load_baseline().predict(text)
    if choice.startswith("Transformer"):
        return load_transformer().predict(text)
    return load_claude(0 if "zero-shot" in choice else 6).predict(text)


def highlight(text: str, indices: list[int]) -> str:
    parts, last = [], 0
    for s, e in indices_to_ranges(indices):
        parts.append(html.escape(text[last:s]))
        parts.append(
            f'<mark style="background:#e34948;color:#ffffff;padding:0 2px;border-radius:3px">{html.escape(text[s:e])}</mark>'
        )
        last = e
    parts.append(html.escape(text[last:]))
    body = "".join(parts).replace("\n", "<br>")
    return f'<div style="font-size:1.1rem;line-height:1.8;padding:0.75rem 1rem;border:1px solid #ddd;border-radius:6px">{body}</div>'


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Model")
status = model_status()
available = [name for name, (ok, _) in status.items() if ok]
if not available:
    st.sidebar.error("No model is available yet. Run `make data baseline` first.")
    st.stop()
choice = st.sidebar.radio("Choose a model", available)
disabled = {name: why for name, (ok, why) in status.items() if not ok}
if disabled:
    st.sidebar.markdown("**Unavailable**")
    for name, why in disabled.items():
        st.sidebar.caption(f"{name}: {why}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Limitations.** Models were trained on ~8k English news comments (Civil Comments) annotated by "
    "crowd workers. They miss context-dependent toxicity, sarcasm and obfuscated words, and over-flag "
    "rude words used harmlessly. Identity terms can be flagged unfairly. Use as a moderator's aid, "
    "never as an automatic decision. Claude calls send the text to the Anthropic API."
)

# ----------------------------------------------------------------------------- main
st.title("Toxic Span Detector")
st.caption("SemEval-2021 Task 5: find the exact characters that make a comment toxic.")

if "text" not in st.session_state:
    st.session_state["text"] = EXAMPLES["Mild insult"]
cols = st.columns(len(EXAMPLES))
for col, (label, example) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["text"] = example

text = st.text_area("Comment", key="text", height=140)

if st.button("Detect", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            t0 = time.perf_counter()
            spans = predict(choice, text)
            elapsed = time.perf_counter() - t0
        except Exception as e:  # e.g. API errors
            st.error(f"Prediction failed: {type(e).__name__}: {e}")
        else:
            st.markdown(highlight(text, spans), unsafe_allow_html=True)
            ranges = indices_to_ranges(spans)
            c1, c2 = st.columns(2)
            c1.metric("Toxic spans", len(ranges))
            c2.metric("Inference time", f"{elapsed * 1000:.1f} ms" if elapsed < 1 else f"{elapsed:.2f} s")
            if ranges:
                st.markdown("**Detected spans** (start-end, end exclusive)")
                st.table([{"span": text[s:e], "start": s, "end": e} for s, e in ranges])
            else:
                st.success("No toxic spans detected.")
