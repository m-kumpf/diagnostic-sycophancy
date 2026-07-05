"""Thin async client for OpenAI-compatible /chat/completions endpoints.

Used against OpenRouter by default, but works against any provider that
mirrors the OpenAI chat-completions schema.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass

import httpx

from .config import ModelConfig


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class CallResult:
    raw_response: str | None
    finish_reason: str | None
    latency_ms: int
    attempts: int
    error: str | None  # None on success


def _api_key(model_cfg: ModelConfig) -> str:
    key = os.environ.get(model_cfg.api_key_env)
    if not key:
        raise RuntimeError(
            f"Missing API key: env var {model_cfg.api_key_env} is not set. "
            f"Add it to .env (see .env.example)."
        )
    return key


def _headers(model_cfg: ModelConfig) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {_api_key(model_cfg)}",
        "Content-Type": "application/json",
    }
    # OpenRouter recommends these for attribution; pulled from env if present.
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    headers.update(model_cfg.extra_headers)
    return headers


async def call_model(
    client: httpx.AsyncClient,
    model_cfg: ModelConfig,
    prompt: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> CallResult:
    body = {
        "model": model_cfg.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        **model_cfg.extra_params,
    }
    url = f"{model_cfg.base_url.rstrip('/')}/chat/completions"
    headers = _headers(model_cfg)

    start = time.perf_counter()
    last_error: str | None = None

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(url, json=body, headers=headers)
                if resp.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    await _backoff(attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
                finish_reason = choice.get("finish_reason")
                return CallResult(
                    raw_response=content,
                    finish_reason=finish_reason,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    attempts=attempt,
                    error=None,
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_error = f"{type(e).__name__}: {e}"
                await _backoff(attempt)
            except httpx.HTTPStatusError as e:
                # Non-retryable 4xx — give up immediately.
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                break
            except Exception as e:  # noqa: BLE001
                last_error = f"{type(e).__name__}: {e}"
                await _backoff(attempt)

        return CallResult(
            raw_response=None,
            finish_reason=None,
            latency_ms=int((time.perf_counter() - start) * 1000),
            attempts=max_retries,
            error=last_error or "unknown error",
        )


async def _backoff(attempt: int) -> None:
    # Exponential backoff with jitter: 0.5s, 1s, 2s, 4s, 8s (+/- 25%).
    base = 0.5 * (2 ** (attempt - 1))
    jitter = base * 0.25 * (2 * random.random() - 1)
    await asyncio.sleep(max(0.0, base + jitter))
