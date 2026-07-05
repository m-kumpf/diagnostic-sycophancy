"""Model registry and run configuration.

All LLM calls go through OpenRouter (https://openrouter.ai/api/v1), which
exposes every provider behind one OpenAI-compatible endpoint. Adding a new
model means appending one entry to ``MODELS`` — no other code changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    extra_params: dict = field(default_factory=dict)
    extra_headers: dict = field(default_factory=dict)
    requests_per_minute: int | None = None


MODELS: dict[str, ModelConfig] = {
    "gemini-3.1-pro-preview": ModelConfig(
        name="gemini-3.1-pro-preview",
        model_id="google/gemini-3.1-pro-preview"),
    "deepseek-v4-pro": ModelConfig(
        name="deepseek-v4-pro",
        model_id="deepseek/deepseek-v4-pro"),
    "qwen3.6-plus": ModelConfig(
        name="qwen3.6-plus",
        model_id="qwen/qwen3.6-plus"),
    "gpt-5.4": ModelConfig(
        name="gpt-5.4",
        model_id="openai/gpt-5.4"),
    "glm-5.1": ModelConfig(
        name="glm-5.1",
        model_id="z-ai/glm-5.1")
}


@dataclass(frozen=True)
class RunConfig:
    n_reps: int = 10
    temperature: float = 1.0
    concurrency: int = 8
    pilot: bool = False
    pilot_n_cases: int = 3
    pilot_n_reps: int = 1
    results_dir: Path = Path("results/raw")
    master_seed: int = 20260524
    max_retries: int = 5
    request_timeout_s: float = 60.0
