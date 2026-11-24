"""End-to-end test of the Streamlit app with the baseline model (no browser, no network)."""
from pathlib import Path

import pytest

from toxspans.baseline import LexiconModel

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


@pytest.mark.skipif(not LexiconModel.available(), reason="run `make baseline` first")
def test_app_baseline_highlights_spans():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    radio = at.sidebar.radio[0]
    baseline = next(o for o in radio.options if o.startswith("Baseline"))
    radio.set_value(baseline).run()
    at.text_area[0].set_value("What a stupid idea, you clown.").run()
    detect = next(b for b in at.button if b.label == "Detect")
    detect.click().run()
    assert not at.exception
    html = "".join(m.value for m in at.markdown)
    assert "<mark" in html and "stupid" in html
    assert any(m.label == "Inference time" for m in at.metric)


@pytest.mark.skipif(not LexiconModel.available(), reason="run `make baseline` first")
def test_app_example_button_fills_text():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    polite = next(b for b in at.button if b.label == "Polite disagreement")
    polite.click().run()
    assert "respectfully" in at.text_area[0].value
