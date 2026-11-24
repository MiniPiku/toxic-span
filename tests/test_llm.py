"""LLM predictor tests with a mocked Anthropic client (no network)."""
from types import SimpleNamespace

from toxspans.llm import ClaudeModel, LLMModel, build_messages, cache_key, parse_response, pick_shots, substrings_to_indices
from toxspans.utils import spans_to_strings


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, replies):
        self.messages = FakeMessages(replies)


# ------------------------------------------------------------------ parsing
def test_parse_plain_json():
    assert parse_response('{"toxic_spans": ["dumb", "clown"]}') == (["dumb", "clown"], True)


def test_parse_code_fence_and_prose():
    raw = '```json\n{"toxic_spans": ["idiot"]}\n```'
    assert parse_response(raw) == (["idiot"], True)
    assert parse_response('Here you go: {"toxic_spans": []} hope it helps') == ([], True)


def test_parse_bad_json_counts_as_failure():
    assert parse_response("I cannot comply") == ([], False)
    assert parse_response('{"toxic_spans": ["unterminated') == ([], False)
    assert parse_response('{"other": 1}') == ([], False)
    assert parse_response(None) == ([], False)


# ------------------------------------------------------------------ offset mapping
def test_exact_then_case_insensitive_and_all_occurrences():
    text = "Dumb take. Really dumb."
    idx = substrings_to_indices(text, ["dumb"])  # exact match exists -> only lowercase occurrence
    assert spans_to_strings(text, idx) == ["dumb"]
    idx = substrings_to_indices(text, ["DUMB"])  # no exact match -> case-insensitive, all occurrences
    assert spans_to_strings(text, idx) == ["Dumb", "dumb"]


def test_unknown_substring_ignored():
    assert substrings_to_indices("all good here", ["not present", "  "]) == []


# ------------------------------------------------------------------ predictor with mocked client + cache
def test_predictor_uses_cache_and_counts_failures(tmp_path):
    client = FakeClient(['{"toxic_spans": ["clown"]}', "garbage"])
    m = ClaudeModel(shots=0, client=client, cache_path=tmp_path / "c.jsonl", model="test-model",
                    provider_name="anthropic")
    text = "you absolute clown"
    assert spans_to_strings(text, m.predict(text)) == ["clown"]
    assert m.predict(text) == m.predict(text)  # served from cache
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["model"] == "test-model" and "temperature" not in call
    assert m.predict("second post") == [] and m.failures == 1
    # cache persisted on disk and reloadable without a client
    m2 = ClaudeModel(shots=0, client=FakeClient([]), cache_path=tmp_path / "c.jsonl", model="test-model", offline=True,
                     provider_name="anthropic")
    assert spans_to_strings(text, m2.predict(text)) == ["clown"]


def test_few_shot_messages_are_seeded():
    rows = [{"text": f"post {i} you dumb clown", "spans": [list(range(len(f"post {i} you "), len(f"post {i} you dumb")))][0]}
            for i in range(10)] + [{"text": f"nice post {i}", "spans": []} for i in range(5)]
    a, b = pick_shots(rows, 6), pick_shots(rows, 6)
    assert a == b and len(a) == 6 and sum(1 for r in a if not r["spans"]) == 2
    msgs = build_messages("target", a)
    assert len(msgs) == 13 and msgs[-1]["content"].endswith("target")
    assert msgs[1]["role"] == "assistant" and "toxic_spans" in msgs[1]["content"]


def test_cache_key_depends_on_model_shots_text():
    assert cache_key("m", 0, "t") != cache_key("m", 6, "t") != cache_key("n", 0, "t")


# ------------------------------------------------------------------ Gemini provider (mocked, no network)
class FakeGeminiModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(text=r)


def test_gemini_provider_request_shape_and_retry(tmp_path, monkeypatch):
    from google.genai import errors

    monkeypatch.setattr("toxspans.llm.time.sleep", lambda s: None)  # no real waiting
    rate_limited = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    fake = SimpleNamespace(models=FakeGeminiModels([rate_limited, '{"toxic_spans": ["clown"]}']))
    rows = [{"text": f"you dumb clown {i}", "spans": list(range(4, 8))} for i in range(8)] + \
           [{"text": f"nice post {i}", "spans": []} for i in range(4)]
    m = LLMModel(shots=6, train_rows=rows, client=fake, cache_path=tmp_path / "g.jsonl",
                 model="gemini-test", provider_name="gemini", min_interval=0)
    text = "what a clown"
    assert spans_to_strings(text, m.predict(text)) == ["clown"]
    assert m.api_calls == 1 and len(fake.models.calls) == 2  # one 429, then success
    call = fake.models.calls[-1]
    assert call["model"] == "gemini-test"
    assert call["config"].temperature == 0 and call["config"].response_mime_type == "application/json"
    roles = [c.role for c in call["contents"]]
    assert roles == ["user", "model"] * 6 + ["user"]
    assert m.name.startswith("Gemini gemini-test (6-shot)")


def test_gemini_non_retryable_error_raises(tmp_path):
    import pytest
    from google.genai import errors

    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}})
    fake = SimpleNamespace(models=FakeGeminiModels([bad]))
    m = LLMModel(shots=0, client=fake, cache_path=tmp_path / "g.jsonl", model="g", provider_name="gemini", min_interval=0)
    with pytest.raises(errors.ClientError):
        m.predict("hello")


def test_provider_selection(monkeypatch):
    from toxspans import llm

    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert llm.provider() == "gemini" and llm.api_key_available()
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert llm.provider() == "anthropic" and not llm.api_key_available()


def test_gemini_daily_quota_stops_immediately(tmp_path, monkeypatch):
    import pytest
    from google.genai import errors

    from toxspans.llm import QuotaExhausted

    sleeps = []
    monkeypatch.setattr("toxspans.llm.time.sleep", lambda s: sleeps.append(s))
    daily = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                               "message": "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
    fake = SimpleNamespace(models=FakeGeminiModels([daily]))
    m = LLMModel(shots=0, client=fake, cache_path=tmp_path / "g.jsonl", model="g", provider_name="gemini", min_interval=0)
    with pytest.raises(QuotaExhausted):
        m.predict("hello")
    assert len(fake.models.calls) == 1 and not sleeps  # no retry loop on a daily quota
