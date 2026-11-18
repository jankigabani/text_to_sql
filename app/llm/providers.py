"""LLM adapters and the model registry.

A model is named by a *spec*: ``provider:model``, for example

    ollama:qwen2.5-coder:7b
    ollama:llama3.1:latest
    gemini:gemini-3.5-flash-lite
    mock:demo

Everything after the first colon is the provider's own model name (Ollama names contain
colons themselves). The UI lists the specs that are actually available on this machine —
the models Ollama has pulled, and the Gemini models the API key can use — so nothing is
hard-coded except a sensible default.

Every provider returns plain text. `json_mode=True` asks the provider to constrain the
output to JSON where it can (Ollama `format: "json"`, Gemini `responseMimeType`); the
caller still parses defensively, because small local models do not always comply.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import httpx

from app.core.config import get_settings
from app.llm.base import LLMResult

log = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ── Ollama (local) ────────────────────────────────────────────────────────────
class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, temperature: float, timeout: int,
                 num_ctx: int = 16384) -> None:
        self.base_url = base_url.rstrip("/")
        self.model, self.temperature, self.timeout = model, temperature, timeout
        self.num_ctx = num_ctx

    @property
    def spec(self) -> str:
        return f"ollama:{self.model}"

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        t0 = time.perf_counter()
        body: dict = {
            "model": self.model,
            "stream": False,
            # num_ctx matters: Ollama's default context is small, and a whole-schema
            # prompt for the shrink step is several thousand tokens.
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            body["format"] = "json"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/api/chat", json=body)
            if r.status_code == 404:
                raise RuntimeError(
                    f"Ollama has no model '{self.model}'. Run: ollama pull {self.model}")
            r.raise_for_status()
            data = r.json()
        text = data.get("message", {}).get("content", "")
        return LLMResult(
            text=text, model=self.spec,
            prompt_chars=len(system) + len(user), completion_chars=len(text),
            latency_ms=(time.perf_counter() - t0) * 1000,
            meta={"prompt_tokens": data.get("prompt_eval_count"),
                  "completion_tokens": data.get("eval_count")},
        )

    def health(self) -> tuple[bool, str]:
        try:
            names = list_ollama_models(self.base_url)
            if not any(n == self.model or n.startswith(self.model) for n in names):
                return False, f"model '{self.model}' not pulled. Run: ollama pull {self.model}"
            return True, f"ollama up, {len(names)} models"
        except Exception as exc:
            return False, f"ollama unreachable at {self.base_url}: {exc}"


def list_ollama_models(base_url: str, timeout: float = 4.0) -> list[str]:
    with httpx.Client(timeout=timeout) as client:
        r = client.get(f"{base_url.rstrip('/')}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


# ── Gemini (Google AI Studio, free tier) ──────────────────────────────────────
class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float, timeout: int,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.api_key, self.model = api_key, model.removeprefix("models/")
        self.temperature, self.timeout = temperature, timeout
        self._transport = transport  # injected by the tests

    @property
    def spec(self) -> str:
        return f"gemini:{self.model}"

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self._transport)

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        t0 = time.perf_counter()
        gen_cfg: dict = {"temperature": self.temperature, "maxOutputTokens": 4096}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen_cfg,
        }
        url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}

        data: dict = {}
        _gemini_pace(get_settings().gemini_min_interval_s)
        with self._client() as client:
            # The free tier is rate-limited per minute (429) and the service is sometimes
            # overloaded (500/503 "high demand"). Both are temporary, so wait and retry.
            retries = get_settings().gemini_retries
            for attempt in range(retries + 1):
                try:
                    r = client.post(url, headers=headers, json=body)
                except httpx.TransportError as exc:
                    if attempt >= retries:
                        raise RuntimeError(f"Gemini unreachable: {exc}") from exc
                    time.sleep(min(5 * 2 ** attempt, 60))
                    continue
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    wait = (_retry_delay(r) if r.status_code == 429 else None) or min(5 * 2 ** attempt, 60)
                    log.info("gemini %s; retrying in %.0fs", r.status_code, wait)
                    time.sleep(min(wait, 60))
                    continue
                if r.status_code >= 400:
                    raise RuntimeError(f"Gemini error {r.status_code}: {_gemini_error(r)}")
                data = r.json()
                break

        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        # Thinking models return their reasoning as separate parts flagged `thought`.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text:
            reason = (data.get("candidates") or [{}])[0].get("finishReason", "unknown")
            raise RuntimeError(f"Gemini returned no text (finishReason={reason})")
        usage = data.get("usageMetadata", {})
        return LLMResult(
            text=text, model=self.spec,
            prompt_chars=len(system) + len(user), completion_chars=len(text),
            latency_ms=(time.perf_counter() - t0) * 1000,
            meta={"prompt_tokens": usage.get("promptTokenCount"),
                  "completion_tokens": usage.get("candidatesTokenCount")},
        )

    def health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "no GEMINI_API_KEY set"
        try:
            names = list_gemini_models(self.api_key, transport=self._transport)
            return True, f"key valid, {len(names)} text models"
        except Exception as exc:
            return False, str(exc)


_GEMINI_LAST_CALL = [0.0]


def _gemini_pace(min_interval: float) -> None:
    """Space calls out so a long benchmark stays under the free tier's per-minute limit."""
    if min_interval <= 0:
        return
    wait = _GEMINI_LAST_CALL[0] + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _GEMINI_LAST_CALL[0] = time.monotonic()


def list_gemini_models(api_key: str, timeout: float = 8.0,
                       transport: httpx.BaseTransport | None = None) -> list[str]:
    with httpx.Client(timeout=timeout, transport=transport) as client:
        r = client.get(f"{GEMINI_BASE}/models", params={"pageSize": 200},
                       headers={"x-goog-api-key": api_key})
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini error {r.status_code}: {_gemini_error(r)}")
        out = []
        for m in r.json().get("models", []):
            name = m.get("name", "").removeprefix("models/")
            if "generateContent" in m.get("supportedGenerationMethods", []) \
                    and name.startswith("gemini") and "embedding" not in name \
                    and "tts" not in name and "image" not in name:
                out.append(name)
        return sorted(out)


def _retry_delay(r: httpx.Response) -> float | None:
    try:
        for d in r.json().get("error", {}).get("details", []):
            if "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s"))
    except Exception:
        return None
    return None


def _gemini_error(r: httpx.Response) -> str:
    try:
        return r.json().get("error", {}).get("message", r.text)[:300]
    except Exception:
        return r.text[:300]


# ── OpenAI / Anthropic (optional, for the cost comparison) ────────────────────
class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, temperature: float, timeout: int) -> None:
        self.api_key, self.model = api_key, model
        self.temperature, self.timeout = temperature, timeout

    @property
    def spec(self) -> str:
        return f"openai:{self.model}"

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        t0 = time.perf_counter()
        body: dict = {
            "model": self.model, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post("https://api.openai.com/v1/chat/completions",
                            headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResult(text=text, model=self.spec,
                         prompt_chars=len(system) + len(user), completion_chars=len(text),
                         latency_ms=(time.perf_counter() - t0) * 1000, meta=data.get("usage", {}))

    def health(self) -> tuple[bool, str]:
        return (bool(self.api_key), "no OPENAI_API_KEY set" if not self.api_key else "key present")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, temperature: float, timeout: int) -> None:
        self.api_key, self.model = api_key, model
        self.temperature, self.timeout = temperature, timeout

    @property
    def spec(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": self.model, "max_tokens": 2048, "temperature": self.temperature,
                      "system": system, "messages": [{"role": "user", "content": user}]},
            )
            r.raise_for_status()
            data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return LLMResult(text=text, model=self.spec,
                         prompt_chars=len(system) + len(user), completion_chars=len(text),
                         latency_ms=(time.perf_counter() - t0) * 1000, meta=data.get("usage", {}))

    def health(self) -> tuple[bool, str]:
        return (bool(self.api_key), "no ANTHROPIC_API_KEY set" if not self.api_key else "key present")


# ── Mock (tests and offline demos) ────────────────────────────────────────────
class MockProvider:
    """Deterministic stand-in so every stage can be tested without a model.

    Either a queue of canned responses, or a `router(system, user, json_mode) -> str`
    that decides the reply from the prompt. With neither, it returns a trivial query.
    """
    name = "mock"

    def __init__(self, model: str = "mock", responses: list[str] | None = None,
                 router: Callable[[str, str, bool], str] | None = None) -> None:
        self.model = model
        self.responses = list(responses or [])
        self.router = router
        self.calls: list[tuple[str, str]] = []

    @property
    def spec(self) -> str:
        return f"mock:{self.model}"

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        self.calls.append((system, user))
        if self.responses:
            text = self.responses.pop(0)
        elif self.router is not None:
            text = self.router(system, user, json_mode)
        else:
            text = "```sql\nSELECT 1 AS answer;\n```"
        return LLMResult(text=text, model=self.spec,
                         prompt_chars=len(system) + len(user), completion_chars=len(text))

    def health(self) -> tuple[bool, str]:
        return True, "mock provider"


# ── registry ──────────────────────────────────────────────────────────────────
def parse_spec(spec: str) -> tuple[str, str]:
    """'ollama:llama3.1:latest' -> ('ollama', 'llama3.1:latest')."""
    spec = (spec or "").strip()
    if ":" not in spec:
        return get_settings().llm_provider, spec
    provider, model = spec.split(":", 1)
    return provider.lower(), model


def provider_for_spec(spec: str):
    s = get_settings()
    provider, model = parse_spec(spec)
    if provider == "ollama":
        return OllamaProvider(s.ollama_base_url, model, s.llm_temperature, s.llm_timeout_s,
                              num_ctx=s.ollama_num_ctx)
    if provider == "gemini":
        return GeminiProvider(s.gemini_api_key, model, s.llm_temperature, s.llm_timeout_s)
    if provider == "openai":
        return OpenAIProvider(s.openai_api_key, model, s.llm_temperature, s.llm_timeout_s)
    if provider == "anthropic":
        return AnthropicProvider(s.anthropic_api_key, model, s.llm_temperature, s.llm_timeout_s)
    if provider == "mock":
        if model == "demo":
            from app.llm.demo import router
            return MockProvider(model="demo", router=router)
        return MockProvider(model=model or "mock")
    raise ValueError(f"unknown model provider '{provider}' in '{spec}'")


def default_spec() -> str:
    s = get_settings()
    if s.default_model:
        return s.default_model
    return f"{s.llm_provider}:{s.llm_model}"


def get_provider(override: str | None = None):
    """Backwards-compatible entry point used by the v1 pipeline."""
    s = get_settings()
    kind = override or s.llm_provider
    if kind in ("ollama", "gemini", "openai", "anthropic", "mock"):
        return provider_for_spec(f"{kind}:{s.llm_model}" if kind != "mock" else "mock:mock")
    return provider_for_spec(kind)


_MODELS_CACHE: tuple[float, dict] | None = None


def available_models(refresh: bool = False) -> dict:
    """What the UI's model picker shows: only models that will actually answer.
    Cached for 60 s so the page stays fast; `refresh=True` asks the providers again."""
    global _MODELS_CACHE
    if not refresh and _MODELS_CACHE and time.monotonic() - _MODELS_CACHE[0] < 60:
        return _MODELS_CACHE[1]
    out = _available_models()
    _MODELS_CACHE = (time.monotonic(), out)
    return out


def _available_models() -> dict:
    s = get_settings()
    models: list[dict] = []
    providers: dict[str, dict] = {}

    try:
        names = list_ollama_models(s.ollama_base_url)
        providers["ollama"] = {"ok": True, "detail": f"{len(names)} models pulled"}
        for n in names:
            if "embed" in n:
                continue  # embedding models cannot write SQL
            models.append({"id": f"ollama:{n}", "provider": "ollama", "label": n,
                           "local": True})
    except Exception as exc:
        providers["ollama"] = {"ok": False, "detail": f"not reachable: {exc}"}

    if s.gemini_api_key:
        try:
            names = list_gemini_models(s.gemini_api_key)
            providers["gemini"] = {"ok": True, "detail": f"{len(names)} models available"}
            preferred = [n for n in names if "flash" in n] or names
            # the configured model first, then the rest
            preferred.sort(key=lambda n: (n != s.gemini_model, "preview" in n, n))
            for n in preferred:
                models.append({"id": f"gemini:{n}", "provider": "gemini", "label": n,
                               "local": False})
        except Exception as exc:
            providers["gemini"] = {"ok": False, "detail": str(exc)}
            # Still offer the default model: listing can be blocked while generation works.
            models.append({"id": f"gemini:{s.gemini_model}", "provider": "gemini",
                           "label": f"{s.gemini_model} (unverified)", "local": False})
    else:
        providers["gemini"] = {"ok": False, "detail": "no GEMINI_API_KEY in .env"}

    models.append({"id": "mock:demo", "provider": "mock",
                   "label": "scripted demo (example questions only, no AI)", "local": True})
    return {"default": default_spec(), "models": models, "providers": providers}
