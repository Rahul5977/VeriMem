"""The teacher/judge client: one OpenAI-compatible endpoint for the whole project.

Rule 8 fixes one teacher/judge model. The endpoint may be hosted (OpenRouter, Alibaba
Model Studio) or served locally (Ollama on a GPU laptop) -- both speak the OpenAI chat
API, so only `VERIMEM_API_BASE_URL` and `VERIMEM_API_MODEL` change between them. Nothing
downstream knows the difference.

What this enforces, so callers do not have to:

* **Caching (rule 1).** Every call is keyed by a hash of model, provider, messages and
  decoding parameters. A repeat call with identical inputs never reaches the network.
* **Token logging (rule 8).** Every uncached call appends to `cache/usage/ledger.jsonl`.
* **Spend cap (rule 2 / P0.6).** Cumulative spend is tracked across runs; a call that
  would exceed `VERIMEM_API_SPEND_CAP_USD` raises instead of silently spending. A warning
  fires at 50%, which is the point the plan says to drop to a cheaper model.
* **Thinking off for judge calls (rule 8).** The parameter differs per provider, so it is
  derived from the base URL and can be overridden.

Local endpoints are free, so the cap and the ledger cost nothing but still record volume.

    from core.llm import LLMClient

    llm = LLMClient()
    reply = llm.complete("Is this claim supported?", system="You are a verifier.")
    print(reply.text, reply.total_tokens, reply.cost_usd)
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from core.cache import get_cache
from core.config import RunConfig
from core.paths import CACHE_DIR

Message = dict[str, str]
Transport = Callable[[dict[str, Any]], dict[str, Any]]

# USD per 1M tokens, (input, output). Local endpoints are free. Unknown models are
# treated as free and warned about once -- better than inventing a price and reporting it.
PRICING: dict[str, tuple[float, float]] = {
    # Alibaba Model Studio
    "qwen-flash": (0.05, 0.40),
    "qwen-turbo": (0.05, 0.20),
    "qwen-plus": (0.40, 1.20),
    "qwen-max": (1.60, 6.40),
    # OpenRouter
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
    "qwen/qwen-2.5-72b-instruct": (0.12, 0.39),
}

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal", "::1"}
_warned: set[str] = set()


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def provider_of(base_url: str) -> str:
    """Short provider tag, used in the cache key so two endpoints never share entries."""
    host = (urlparse(base_url).hostname or "").lower()
    if host in _LOCAL_HOSTS:
        return "local"
    if "openrouter" in host:
        return "openrouter"
    if "aliyuncs" in host or "dashscope" in host:
        return "dashscope"
    return host or "unknown"


def is_free(base_url: str) -> bool:
    return provider_of(base_url) == "local"


def price_of(model: str) -> tuple[float, float]:
    if model in PRICING:
        return PRICING[model]
    if model not in _warned:
        _warned.add(model)
        warnings.warn(
            f"No pricing known for {model!r}; cost will be reported as 0. Add it to "
            "core.llm.PRICING so the spend cap means something.",
            stacklevel=2,
        )
    return (0.0, 0.0)


def thinking_off_params(base_url: str) -> dict[str, Any]:
    """Provider-specific way to disable reasoning mode (rule 8).

    The parameter name is not standardised across OpenAI-compatible providers. These are
    the documented forms; pass `extra_body=` to `complete()` to override if a provider
    changes it. An unknown provider gets nothing rather than a guessed parameter, since
    an unrecognised field is rejected outright by some servers.
    """
    provider = provider_of(base_url)
    if provider == "dashscope":
        return {"enable_thinking": False}
    if provider == "openrouter":
        return {"reasoning": {"exclude": True}}
    return {}


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = "stop"
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Usage:
    """In-process totals. The on-disk ledger is the cross-run record."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        if r.cached:
            self.cached_calls += 1
            return
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class SpendCapExceeded(RuntimeError):
    pass


class SpendLedger:
    """Cumulative spend across runs, so the cap survives a restart."""

    def __init__(self, path: Path | None = None, cap_usd: float | None = None) -> None:
        self.dir = (path or CACHE_DIR) / "usage"
        self.ledger = self.dir / "ledger.jsonl"
        self.totals_path = self.dir / "totals.json"
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._warned_half = False

    def total(self) -> float:
        try:
            return float(json.loads(self.totals_path.read_text())["cost_usd"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return 0.0

    def check(self, projected: float = 0.0) -> None:
        if not self.cap_usd:
            return
        spent = self.total()
        if spent + projected > self.cap_usd:
            raise SpendCapExceeded(
                f"This call would take spend to ${spent + projected:.4f}, over the "
                f"${self.cap_usd:.2f} cap. Raise VERIMEM_API_SPEND_CAP_USD, or switch "
                f"VERIMEM_API_BASE_URL to a local endpoint, which is free."
            )
        if not self._warned_half and spent > self.cap_usd / 2:
            self._warned_half = True
            warnings.warn(
                f"API spend ${spent:.4f} is past half of the ${self.cap_usd:.2f} cap -- "
                "the plan says drop to a cheaper model or a local endpoint here.",
                stacklevel=2,
            )

    def record(self, response: LLMResponse, provider: str) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "at": datetime.now(UTC).isoformat(timespec="seconds"),
                            "provider": provider,
                            "model": response.model,
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                            "cost_usd": round(response.cost_usd, 8),
                        }
                    )
                    + "\n"
                )
            total = self.total() + response.cost_usd
            tmp = self.totals_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cost_usd": round(total, 8)}), encoding="utf-8")
            tmp.replace(self.totals_path)


class LLMClient:
    """Cached, token-logged, spend-capped client for the teacher/judge model."""

    def __init__(
        self,
        cfg: RunConfig | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        cap_usd: float | None = None,
        transport: Transport | None = None,
        cache_namespace: str = "llm",
    ) -> None:
        _load_env()
        cfg = cfg or RunConfig()
        self.base_url = base_url or os.environ.get("VERIMEM_API_BASE_URL") or cfg.api_base_url
        self.model = model or os.environ.get("VERIMEM_API_MODEL") or cfg.api_model
        self.disable_thinking = cfg.disable_thinking
        self.provider = provider_of(self.base_url)
        self._api_key = api_key or os.environ.get("VERIMEM_API_KEY") or ""
        self._transport = transport
        self._client = None
        self.cache = get_cache(cache_namespace)
        self.usage = Usage()
        # Guards `usage`, which several worker threads add to at once when a runner is
        # given --workers. The ledger and the cache hold their own locks already.
        self._usage_lock = threading.Lock()

        if cap_usd is None:
            raw = os.environ.get("VERIMEM_API_SPEND_CAP_USD", "").strip()
            cap_usd = float(raw) if raw else None
        self.ledger = SpendLedger(cap_usd=None if is_free(self.base_url) else cap_usd)

        if self.model in ("", "TBD"):
            raise ValueError(
                "No teacher model set. Put VERIMEM_API_MODEL in .env -- "
                "e.g. 'qwen2.5:7b' for a local Ollama server, or 'qwen-flash' for "
                "Alibaba Model Studio. P0.13 decides which."
            )

    # --- public API ----------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = 13,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """One chat call. Deterministic by default, so reruns hit the cache."""
        body = dict(thinking_off_params(self.base_url) if self.disable_thinking else {})
        if extra_body:
            body.update(extra_body)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if body:
            payload["extra_body"] = body

        key = {"provider": self.provider, **payload}
        hit = self.cache.get(key)
        if hit is not None:
            response = LLMResponse(**{**hit, "cached": True})
            with self._usage_lock:
                self.usage.add(response)
            return response

        self.ledger.check(self._estimate_cost(payload, max_tokens))
        raw = (self._transport or self._default_transport)(payload)
        response = self._parse(raw)
        self.cache.set(key, {k: v for k, v in vars(response).items() if k != "cached"})
        self.ledger.record(response, self.provider)
        with self._usage_lock:
            self.usage.add(response)
        return response

    def estimate(self, n_calls: int, avg_input_tokens: int, avg_output_tokens: int) -> float:
        """Projected USD for a bulk job. Rule 2: report this before launching."""
        if is_free(self.base_url):
            return 0.0
        pin, pout = price_of(self.model)
        return n_calls * (avg_input_tokens * pin + avg_output_tokens * pout) / 1_000_000

    # --- internals -----------------------------------------------------------------

    def _estimate_cost(self, payload: dict[str, Any], max_tokens: int) -> float:
        if is_free(self.base_url):
            return 0.0
        from experience.format import estimate_tokens

        text = "".join(m.get("content", "") for m in payload["messages"])
        pin, pout = price_of(self.model)
        return (estimate_tokens(text) * pin + max_tokens * pout) / 1_000_000

    def _parse(self, raw: dict[str, Any]) -> LLMResponse:
        choice = (raw.get("choices") or [{}])[0]
        usage = raw.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if is_free(self.base_url):
            cost = 0.0  # don't ask for a price, and don't warn, on a free endpoint
        else:
            pin, pout = price_of(self.model)
            cost = (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
        return LLMResponse(
            text=(choice.get("message") or {}).get("content") or "",
            model=raw.get("model") or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def _default_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env without the SDK
                raise ImportError(
                    "The openai package is required for live calls: pip install openai"
                ) from exc
            # Local servers ignore the key but the SDK insists on a non-empty one.
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self._api_key or ("ollama" if is_free(self.base_url) else ""),
                max_retries=3,
            )
        kwargs = dict(payload)
        extra = kwargs.pop("extra_body", None)
        completion = self._client.chat.completions.create(
            **kwargs, **({"extra_body": extra} if extra else {})
        )
        return completion.model_dump()


def ping(client: LLMClient | None = None) -> LLMResponse:
    """Smallest possible live call, for P0.7-style 'does the key work' checks."""
    client = client or LLMClient()
    return client.complete("Reply with the single word: ok", max_tokens=8)
