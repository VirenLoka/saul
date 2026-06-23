"""LLM provider abstraction layer.

This module decouples the rest of the application from any specific inference
backend. Everything downstream depends only on the :class:`LLMProvider`
interface; the concrete backend is selected at runtime from config.

Local backend
-------------
The sole local runner is **vLLM**, launched via ``serve.sh``. It exposes an
OpenAI-compatible ``/v1/chat/completions`` endpoint on
``http://<host>:<port>`` and is protected by a bearer token
(``SPARKS_API_KEY`` env var / ``local_inference_settings.api_key`` in config).

Design goals
------------
* One narrow interface: ``generate(system_prompt, user_prompt) -> str``.
* Backends are constructed by a single :func:`get_provider` factory driven by
  ``AppConfig``, so swapping models never touches call sites.
* Heavy / optional dependencies (``requests``, ``openai``, ``anthropic``)
  are imported lazily *inside* each provider, so importing this module —
  and running the unit/smoke tests — needs none of them.
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
# Local backend: Qwen/Qwen2.5-7B-Instruct served by vLLM
# --------------------------------------------------------------------------- #
class LocalQwenProvider(LLMProvider):
    """Calls the local vLLM server (launched via serve.sh).

    The server exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.
    If ``local_inference_settings.api_key`` is set (or ``SPARKS_API_KEY`` env
    var is present), every request carries ``Authorization: Bearer <key>``.
    """

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.model_id = config.model_selection.local_model
        self.settings = config.local_inference
        self.name = f"local:vllm:{self.model_id}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests  # lazy import

        url = f"{self.settings.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

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
                url,
                json=payload,
                headers=headers,
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                f"vLLM request to {url} failed: {exc}"
            ) from exc
        return data["choices"][0]["message"]["content"].strip()


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
