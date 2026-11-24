"""LLM toxic span predictor (zero-shot / few-shot) with an on-disk cache.

Two providers, chosen by env LLM_PROVIDER ("gemini" or "anthropic"; if unset, Gemini is used
when GEMINI_API_KEY / GOOGLE_API_KEY is set, else Anthropic):
  * gemini    - Google Gemini API (free tier), model env GEMINI_MODEL (default gemini-3.5-flash-lite),
                temperature 0, JSON response mode, throttled to GEMINI_MIN_INTERVAL seconds per call.
  * anthropic - Claude API, model env ANTHROPIC_MODEL (default claude-haiku-5-5).

The model returns JSON {"toxic_spans": ["exact substring", ...]}. Each substring is mapped
back to character offsets by exact search, falling back to case-insensitive search, and
EVERY occurrence of the substring is marked (the model returns strings, not positions, so
we cannot tell which occurrence it meant; marking all is the simplest consistent rule).

Claude Haiku 5.5 rejects non-default sampling parameters, so `temperature` is not sent to Claude.
Every response is cached, keyed by sha256(model, shots, text), so reruns are free and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from .metrics import evaluate
from .utils import DATA_DIR, ROOT, SEED, spans_to_strings

DEFAULT_MODEL = "claude-haiku-5-5"  # anthropic
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # served on the Gemini free tier (Nov 2026)
CACHE_PATH = DATA_DIR / "cache" / "llm_cache.jsonl"

SYSTEM_PROMPT = (
    "You are an expert content-moderation annotator for the SemEval-2021 Toxic Spans task. "
    "Given a user comment, identify the exact sequences of characters that make the comment toxic "
    "(insults, slurs, profanity, threats, identity attacks, demeaning words). Annotators typically mark "
    "short spans: the offensive word or short phrase itself, not the whole sentence. If the comment is "
    "toxic as a whole but no particular words carry the toxicity, or the comment is not toxic, return an "
    "empty list.\n\n"
    "Respond with ONLY a JSON object of the form {\"toxic_spans\": [\"exact substring\", ...]}. "
    "Each substring must be copied character-for-character from the comment (same case, spelling and "
    "punctuation). No explanations, no markdown."
)


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def gemini_key() -> str:
    load_env()
    return _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")


def provider() -> str:
    """'gemini' or 'anthropic' (env LLM_PROVIDER, else whichever key is present, Gemini first)."""
    load_env()
    p = _env("LLM_PROVIDER").lower()
    if p in ("gemini", "google"):
        return "gemini"
    if p in ("anthropic", "claude"):
        return "anthropic"
    return "gemini" if gemini_key() else "anthropic"


def api_key_available(prov: str | None = None) -> bool:
    load_env()
    prov = prov or provider()
    return bool(gemini_key()) if prov == "gemini" else bool(_env("ANTHROPIC_API_KEY"))


def model_name(prov: str | None = None) -> str:
    load_env()
    if (prov or provider()) == "gemini":
        return _env("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    return _env("ANTHROPIC_MODEL") or DEFAULT_MODEL


def display_name(prov: str | None = None) -> str:
    return "Gemini" if (prov or provider()) == "gemini" else "Claude"


# ----------------------------------------------------------------------------- parsing + mapping
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_response(raw: str) -> tuple[list[str], bool]:
    """Return (substrings, ok). Strips code fences; extracts the first {...} block; never raises."""
    if raw is None:
        return [], False
    s = _FENCE_RE.sub("", raw.strip()).strip()
    candidates = [s]
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m and m.group(0) != s:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("toxic_spans"), list):
            return [x for x in obj["toxic_spans"] if isinstance(x, str) and x.strip()], True
        if isinstance(obj, list):  # tolerate a bare list
            return [x for x in obj if isinstance(x, str) and x.strip()], True
    return [], False


def substrings_to_indices(text: str, substrings: list[str]) -> list[int]:
    """Exact search first, then case-insensitive; every occurrence is marked."""
    out: set[int] = set()
    lower = text.lower()
    for sub in substrings:
        sub = sub.strip()
        if not sub:
            continue
        hay, needle = text, sub
        if text.find(sub) == -1:
            hay, needle = lower, sub.lower()
        start = hay.find(needle)
        while start != -1:
            out.update(range(start, start + len(needle)))
            start = hay.find(needle, start + 1)
    return sorted(out)


# ----------------------------------------------------------------------------- prompting
def pick_shots(train_rows: list[dict], k: int, seed: int = SEED, max_chars: int = 300) -> list[dict]:
    """Seeded few-shot sample: ~2/3 posts with spans, ~1/3 with empty spans, short posts only."""
    if k <= 0:
        return []
    rng = random.Random(seed)
    short = [r for r in train_rows if len(r["text"]) <= max_chars]
    toxic = [r for r in short if r["spans"]]
    empty = [r for r in short if not r["spans"]]
    n_empty = min(len(empty), k // 3)
    shots = rng.sample(toxic, min(len(toxic), k - n_empty)) + rng.sample(empty, n_empty)
    rng.shuffle(shots)
    return shots


def build_messages(text: str, shots: list[dict]) -> list[dict]:
    msgs: list[dict] = []
    for ex in shots:
        msgs.append({"role": "user", "content": f"Comment:\n{ex['text']}"})
        msgs.append({"role": "assistant", "content": json.dumps({"toxic_spans": spans_to_strings(ex["text"], ex["spans"])})})
    msgs.append({"role": "user", "content": f"Comment:\n{text}"})
    return msgs


def cache_key(model: str, shots: int, text: str) -> str:
    return hashlib.sha256(json.dumps([model, shots, text], ensure_ascii=False).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- predictor
class QuotaExhausted(RuntimeError):
    """The provider's daily request quota is used up; cached progress is kept, rerun later."""


class LLMModel:
    def __init__(
        self,
        shots: int = 0,
        train_rows: list[dict] | None = None,
        model: str | None = None,
        client=None,
        cache_path: Path | None = CACHE_PATH,
        max_retries: int = 6,
        offline: bool = False,
        provider_name: str | None = None,
        min_interval: float | None = None,
    ):
        self.shots = shots
        self.provider = provider_name or provider()
        self.model = model or model_name(self.provider)
        if min_interval is None:
            min_interval = float(_env("GEMINI_MIN_INTERVAL") or 4.5) if self.provider == "gemini" else 0.0
        self.min_interval = min_interval  # free-tier rate limit (~15 requests/min)
        self._last_call = 0.0
        self.examples = pick_shots(train_rows or [], shots)
        if shots and len(self.examples) < shots:
            raise ValueError("few-shot mode needs training rows (pass train_rows)")
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = self._load_cache()
        self.max_retries = max_retries
        self.offline = offline  # True -> only use the cache, never call the API
        self._client = client
        self.failures = 0  # JSON parse failures
        self.api_calls = 0
        self.name = f"{display_name(self.provider)} {self.model} ({'zero' if shots == 0 else f'{shots}'}-shot)"

    # ------------------------------------------------------------- cache
    def _load_cache(self) -> dict[str, dict]:
        cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        cache[e["key"]] = e
                    except (json.JSONDecodeError, KeyError):
                        continue
        return cache

    def _append_cache(self, entry: dict) -> None:
        self.cache[entry["key"]] = entry
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def cached(self, text: str) -> bool:
        return cache_key(self.model, self.shots, text) in self.cache

    # ------------------------------------------------------------- api
    @property
    def client(self):
        if self._client is None:
            load_env()
            if self.provider == "gemini":
                from google import genai

                self._client = genai.Client(api_key=gemini_key())
            else:
                import anthropic

                # SDK retries are disabled; we do our own exponential backoff below.
                self._client = anthropic.Anthropic(max_retries=0)
        return self._client

    def _call(self, text: str) -> str:
        return self._call_gemini(text) if self.provider == "gemini" else self._call_anthropic(text)

    def _call_gemini(self, text: str) -> str:
        from google.genai import errors, types

        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in build_messages(text, self.examples)
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0, max_output_tokens=1024,
            response_mime_type="application/json",
        )
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                t0 = time.perf_counter()
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                self.last_latency = time.perf_counter() - t0
                self.api_calls += 1
                return resp.text or ""
            except (errors.ClientError, errors.ServerError) as e:
                code = getattr(e, "code", None)
                if code == 429 and "PerDay" in str(e):
                    raise QuotaExhausted(f"daily free-tier quota for {self.model} is used up") from e
                if code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise
                m = re.search(r"retry in ([0-9.]+)s|'retryDelay': '([0-9.]+)s'", str(e))
                delay = float(next(g for g in m.groups() if g)) + 1 if m else 5.0 * (2**attempt)
                delay = min(delay, 120.0) + random.random()
                print(f"  Gemini {code}; retry {attempt + 1}/{self.max_retries} in {delay:.0f}s", flush=True)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def _call_anthropic(self, text: str) -> str:
        import anthropic

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=build_messages(text, self.examples),
                    output_config={"effort": "low"},
                )
                self.last_latency = time.perf_counter() - t0
                self.api_calls += 1
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                if attempt == self.max_retries:
                    raise
                wait = delay * (2**attempt) + random.random()
                print(f"  {type(e).__name__}; retry {attempt + 1}/{self.max_retries} in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------- predict
    def predict_with_meta(self, text: str) -> dict:
        key = cache_key(self.model, self.shots, text)
        entry = self.cache.get(key)
        if entry is None:
            if self.offline:
                raise KeyError("not cached")
            self.last_latency = None
            t0 = time.perf_counter()
            raw = self._call(text)
            latency = self.last_latency if self.last_latency is not None else time.perf_counter() - t0
            entry = {"key": key, "provider": self.provider, "model": self.model, "shots": self.shots, "raw": raw,
                     "latency": latency}
            self._append_cache(entry)
        subs, ok = parse_response(entry["raw"])
        if not ok:
            self.failures += 1
        return {"spans": substrings_to_indices(text, subs), "substrings": subs, "ok": ok,
                "latency": entry.get("latency", 0.0)}

    def predict(self, text: str) -> list[int]:
        return self.predict_with_meta(text)["spans"]

    def predict_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.predict(t) for t in texts]

    def score(self, rows: list[dict]) -> dict:
        res = evaluate(self.predict_batch([r["text"] for r in rows]), [r["spans"] for r in rows])
        res["json_failures"] = self.failures
        return res


ClaudeModel = LLMModel  # backwards-compatible alias
