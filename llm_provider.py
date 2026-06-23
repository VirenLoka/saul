"""LLM provider abstraction layer.

This module decouples the rest of the application from any specific inference
backend. Everything downstream depends only on the :class:`LLMProvider`
interface; the concrete backend (local Qwen via Ollama/vLLM/Transformers, or an
external API such as OpenAI / Anthropic) is selected at runtime from config.

Design goals
------------
* One narrow interface: ``generate(system_prompt, user_prompt) -> str``.
* Backends are constructed by a single :func:`get_provider` factory driven by
  ``AppConfig``, so swapping models never touches call sites.
* Heavy / optional dependencies (``requests``, ``openai``, ``anthropic``,
  ``transformers``) are imported lazily *inside* each provider, so importing
  this module — and running the unit/smoke tests — needs none of them.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a hard import cycle / runtime dependency
    from config_loader import AppConfig


class LLMProviderError(RuntimeError):
    """Raised when a provider is misconfigured or a backend call fails."""


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class LLMProvider(abc.ABC):
    """Abstract base every concrete LLM backend implements."""

    #: Human-readable name, set by subclasses for logging/reporting.
    name: str = "abstract"

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's completion for the given prompts."""
        raise NotImplementedError

    def describe(self) -> str:
        """Short, log-friendly description of the active backend."""
        return self.name


# --------------------------------------------------------------------------- #
# Local backend (default): Qwen/Qwen2.5-7B-Instruct
# --------------------------------------------------------------------------- #
class LocalQwenProvider(LLMProvider):
    """Serves ``Qwen/Qwen2.5-7B-Instruct`` through a local runner.

    Supported runners (config: ``local_inference_settings.runner``):
      * ``ollama``       — HTTP call to a local Ollama daemon.
      * ``vllm``         — HTTP call to a vLLM OpenAI-compatible server.
      * ``transformers`` — in-process Hugging Face Transformers pipeline.
    """

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.model_id = config.model_selection.local_model
        self.settings = config.local_inference
        self.name = f"local:{self.settings.runner}:{self.model_id}"

    # -- public ------------------------------------------------------------- #
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        runner = self.settings.runner
        if runner == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)
        if runner == "vllm":
            return self._generate_vllm(system_prompt, user_prompt)
        if runner == "transformers":
            return self._generate_transformers(system_prompt, user_prompt)
        raise LLMProviderError(
            f"Unknown local runner '{runner}'. "
            "Expected one of: ollama, vllm, transformers."
        )

    # -- runners ------------------------------------------------------------ #
    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        import requests  # lazy import

        url = f"{self.settings.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.settings.ollama_tag,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }
        try:
            resp = requests.post(
                url, json=payload, timeout=self.settings.request_timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - surface as provider error
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc
        return data.get("message", {}).get("content", "").strip()

    def _generate_vllm(self, system_prompt: str, user_prompt: str) -> str:
        import requests  # lazy import

        url = f"{self.settings.host.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.model_id,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            resp = requests.post(
                url, json=payload, timeout=self.settings.request_timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"vLLM request failed: {exc}") from exc
        return data["choices"][0]["message"]["content"].strip()

    def _generate_transformers(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from transformers import pipeline  # lazy import
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                "transformers is not installed. "
                "Install with: pip install 'transformers[torch]'"
            ) from exc

        # Cache the pipeline on the instance so weights load once.
        pipe = getattr(self, "_pipe", None)
        if pipe is None:
            pipe = pipeline("text-generation", model=self.model_id)
            self._pipe = pipe

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        out = pipe(
            messages,
            max_new_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )
        # transformers returns the full chat; take the last assistant turn.
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()


# --------------------------------------------------------------------------- #
# External API fallbacks
# --------------------------------------------------------------------------- #
class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions backend (fallback)."""

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.model = config.model_selection.openai_model
        self.api_key = config.api_credentials.openai_api_key
        self.base_url = config.api_credentials.openai_base_url or None
        self.name = f"openai:{self.model}"
        if not self.api_key:
            raise LLMProviderError(
                "OpenAI selected but no API key found. "
                "Set OPENAI_API_KEY or api_credentials.openai.api_key."
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI  # lazy import
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                "openai package not installed. Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.config.local_inference.temperature,
            max_tokens=self.config.local_inference.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API backend (fallback)."""

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.model = config.model_selection.anthropic_model
        self.api_key = config.api_credentials.anthropic_api_key
        self.name = f"anthropic:{self.model}"
        if not self.api_key:
            raise LLMProviderError(
                "Anthropic selected but no API key found. "
                "Set ANTHROPIC_API_KEY or api_credentials.anthropic.api_key."
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import anthropic  # lazy import
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.config.local_inference.max_tokens,
            temperature=self.config.local_inference.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate any text blocks in the response.
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts).strip()


# --------------------------------------------------------------------------- #
# Offline stub — used for tests / CI / no-network demos
# --------------------------------------------------------------------------- #
class MockProvider(LLMProvider):
    """Deterministic, dependency-free provider.

    Performs **no** model forward pass. It echoes a fixed, clearly-labelled
    template so that the end-to-end pipeline (config -> ingest -> analyze ->
    report) can be exercised on machines that cannot run the real model.
    """

    name = "mock:offline"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "[MOCK LLM OUTPUT — no model was run]\n"
            "This is a deterministic stub used for offline testing. "
            "Configure model_selection.provider in config.yaml to use a real "
            "model (local Qwen or an API fallback).\n"
            f"(system prompt chars={len(system_prompt)}, "
            f"user prompt chars={len(user_prompt)})"
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_PROVIDERS = {
    "local": LocalQwenProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


def get_provider(config: "AppConfig") -> LLMProvider:
    """Construct the provider selected by ``model_selection.provider``."""
    provider_key = config.model_selection.provider
    cls = _PROVIDERS.get(provider_key)
    if cls is None:
        raise LLMProviderError(
            f"Unsupported provider '{provider_key}'. "
            f"Choose one of: {sorted(_PROVIDERS)}."
        )
    # MockProvider takes no config; the rest are config-driven.
    if cls is MockProvider:
        return MockProvider()
    return cls(config)
