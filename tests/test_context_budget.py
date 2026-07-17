"""Tests for the request-size guards — trimming, truncation, completion clamp,
and the provider's fatal rate-limit/size error classification (no network)."""

from __future__ import annotations

from context_budget import (
    MIN_COMPLETION_TOKENS,
    estimate_request_tokens,
    fit_request,
    truncate_tool_results,
)
from llm_provider import StreamEvent


# --------------------------------------------------------------------------- #
# Tool-result truncation
# --------------------------------------------------------------------------- #
def test_truncate_tool_results_caps_content():
    big = "x" * 10_000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "tool_call_id": "1", "name": "t", "content": big},
        {"role": "user", "content": "hi"},
    ]
    out = truncate_tool_results(msgs, 500)
    assert len(out[1]["content"]) < len(big)
    assert "truncated" in out[1]["content"]
    # Non-tool messages are untouched; original list is not mutated.
    assert out[0]["content"] == "sys" and out[2]["content"] == "hi"
    assert len(msgs[1]["content"]) == 10_000


def test_truncate_disabled_when_zero():
    msgs = [{"role": "tool", "content": "y" * 100, "tool_call_id": "1", "name": "t"}]
    assert truncate_tool_results(msgs, 0)[0]["content"] == "y" * 100


# --------------------------------------------------------------------------- #
# fit_request: history trimming + completion clamp
# --------------------------------------------------------------------------- #
def test_fit_request_noop_when_disabled():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out, eff = fit_request(msgs, None, max_request_tokens=0, configured_max_tokens=4096)
    assert out == msgs
    assert eff == 4096


def test_fit_request_trims_old_history_but_keeps_system_and_last():
    msgs = [{"role": "system", "content": "system"}]
    # Many large user/assistant turns; only the newest should survive a tight budget.
    for i in range(40):
        msgs.append({"role": "user", "content": f"question {i} " + "z" * 400})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "z" * 400})
    out, eff = fit_request(msgs, None, max_request_tokens=800, configured_max_tokens=4096)
    assert out[0]["role"] == "system"          # system preserved
    assert out[-1] == msgs[-1]                 # newest message preserved
    assert len(out) < len(msgs)                # old history dropped
    assert estimate_request_tokens(out) + MIN_COMPLETION_TOKENS <= 800 or len(out) == 2


def test_fit_request_clamps_completion_budget():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u" * 4000}]
    out, eff = fit_request(msgs, None, max_request_tokens=2000, configured_max_tokens=4096)
    # prompt (~1000) + eff must fit under the cap, so completion is clamped down.
    assert eff < 4096
    assert estimate_request_tokens(out) + eff <= 2000 + MIN_COMPLETION_TOKENS


def test_fit_request_never_orphans_a_tool_message():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "old " + "z" * 2000},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "t", "content": "z" * 2000},
        {"role": "user", "content": "new question"},
    ]
    out, _ = fit_request(msgs, None, max_request_tokens=300, configured_max_tokens=1024)
    # After the system message, the body must not start with an orphan tool msg.
    body = [m for m in out if m["role"] != "system"]
    assert not body or body[0]["role"] != "tool"


# --------------------------------------------------------------------------- #
# Fatal-error classification in the provider
# --------------------------------------------------------------------------- #
class _RateLimitError(Exception):
    status_code = 429


class _TooLargeError(Exception):
    status_code = 413


class _AuthError(Exception):
    status_code = 401


class _ServerError(Exception):
    status_code = 500


def _fake_config(exc):
    """Build a config whose OpenAI client raises ``exc`` on create()."""
    from config_loader import load_config

    cfg = load_config(provider_override="groq")

    class _Boom:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_kwargs):
                    raise exc

    return cfg, _Boom()


def _run(provider):
    return list(provider.stream_chat([{"role": "user", "content": "hi"}], tools=None))


def _events_for(exc):
    from llm_provider import GroqProvider

    cfg, client = _fake_config(exc)
    provider = GroqProvider(cfg)
    provider._client = lambda: client  # bypass the real OpenAI client
    return _run(provider)


def test_rate_limit_is_fatal():
    errs = [e for e in _events_for(_RateLimitError("rate_limit_exceeded")) if e.type == "error"]
    assert errs and errs[0].fatal is True
    assert "rate" in errs[0].text.lower() or "large" in errs[0].text.lower()


def test_request_too_large_is_fatal():
    errs = [e for e in _events_for(_TooLargeError("Request too large, tokens per minute")) if e.type == "error"]
    assert errs and errs[0].fatal is True


def test_auth_4xx_is_fatal():
    errs = [e for e in _events_for(_AuthError("invalid api key")) if e.type == "error"]
    assert errs and errs[0].fatal is True


def test_server_5xx_is_not_fatal():
    errs = [e for e in _events_for(_ServerError("upstream boom")) if e.type == "error"]
    assert errs and errs[0].fatal is False


def test_stream_event_fatal_defaults_false():
    assert StreamEvent(type="error", text="x").fatal is False
