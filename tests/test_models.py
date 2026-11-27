import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from researcher import http
from researcher.config import (
    ConfigError,
    EndpointSettings,
    ModelSettings,
    Settings,
)
from researcher.http import HTTPClient
from researcher.models import (
    USER_AGENT,
    Message,
    ModelClient,
    ModelError,
    ModelRole,
)

_RELAY = "relay"
_HANDLER = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


@pytest.fixture(autouse=True)
def api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "go-key")
    monkeypatch.setenv("RELAY_KEY", "relay-key")


@pytest.fixture
def xdg_config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate the default credentials location and leave OPENCODE_API_KEY unset."""
    config_home = tmp_path / "config-home"
    (config_home / "researcher").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("OPENCODE_API_KEY")
    return config_home


def _credentials_file(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


class Clock:
    """Advance monotonic time by each requested delay instead of waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def _settings(
    endpoint: EndpointSettings,
    *,
    flash: ModelSettings | None = None,
    evaluator: ModelSettings | None = None,
) -> Settings:
    return Settings(
        endpoints={_RELAY: endpoint},
        flash_model=flash or ModelSettings(_RELAY, "flash-model"),
        evaluator_model=evaluator or ModelSettings(_RELAY, "eval-model"),
    )


def _openai_reply(content: str = "Screened.") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ]
        },
    )


def _anthropic_reply(
    blocks: list[dict[str, object]], stop_reason: object = "end_turn"
) -> httpx.Response:
    return httpx.Response(200, json={"stop_reason": stop_reason, "content": blocks})


def _gemini_reply(
    parts: list[dict[str, object]], finish_reason: str = "STOP"
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": parts}, "finishReason": finish_reason}]
        },
    )


def _run(
    settings: Settings,
    handler: _HANDLER,
    action: Callable[[ModelClient], Awaitable[Any]],
    *,
    total_timeout: float = 60.0,
    credentials_path: Path | None = None,
) -> Any:
    async def main() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as session:
            client = ModelClient(
                HTTPClient(session, total_timeout=total_timeout),
                settings,
                credentials_path=credentials_path,
            )
            return await action(client)

    return asyncio.run(main())


def _complete(
    settings: Settings,
    handler: _HANDLER,
    *,
    role: ModelRole = "flash",
    system: str | None = "Be brief.",
    credentials_path: Path | None = None,
) -> str:
    async def action(client: ModelClient) -> str:
        return await client.complete(
            role,
            [Message("user", "Summarise the abstract.")],
            system=system,
        )

    result = _run(settings, handler, action, credentials_path=credentials_path)
    assert isinstance(result, str)
    return result


def test_flash_request_uses_default_go_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    assert _complete(Settings(), handler) == "Screened."
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://opencode.ai/inference/go/openai/v1/chat/completions"
    )
    assert request.headers["authorization"] == "Bearer go-key"
    assert request.headers["user-agent"] == USER_AGENT
    assert request.headers["content-type"] == "application/json"
    assert re.fullmatch(r"ses_[0-9a-f]{32}", request.headers["x-opencode-session"])
    body = json.loads(request.content)
    assert body == {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Summarise the abstract."},
        ],
        "stream": False,
    }


def test_openai_compatible_endpoint_can_use_local_http() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:8080/v1/chat/completions"
        return _openai_reply()

    endpoint = EndpointSettings(
        api="openai", base_url="http://127.0.0.1:8080/v1", api_key_env="RELAY_KEY"
    )
    assert _complete(_settings(endpoint), handler) == "Screened."


@pytest.mark.parametrize(
    ("base_url", "detail"),
    [
        ("https://relay.test:bad/v1", "InvalidURL"),
        ("/v1", "absolute URL"),
        ("https://user:secret@relay.test/v1", "embedded credentials"),
    ],
)
def test_unusable_endpoint_url_fails_on_use(base_url: str, detail: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("an unusable endpoint URL reached the transport")

    endpoint = EndpointSettings(
        api="openai", base_url=base_url, api_key_env="RELAY_KEY"
    )
    with pytest.raises(ModelError, match=detail) as caught:
        _complete(_settings(endpoint), handler)
    assert "secret" not in str(caught.value)


def test_evaluator_request_uses_evaluator_model() -> None:
    bodies: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _openai_reply()

    _complete(Settings(), handler, role="evaluator")
    assert bodies[0]["model"] == "mimo-v2.6-pro"


def test_independent_completions_get_fresh_conversation_ids() -> None:
    sessions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["x-opencode-session"])
        return _openai_reply()

    async def action(client: ModelClient) -> None:
        messages = [Message("user", "screen")]
        await client.complete("flash", messages)
        await client.complete("flash", messages)

    settings = Settings()
    _run(settings, handler, action)
    first, second = sessions
    assert first != second
    assert settings.endpoints["opencode"].headers["x-opencode-session"] == (
        "ses_{conversation_id}"
    )


@pytest.mark.parametrize(
    ("api", "auth", "expected"),
    [
        ("openai", None, {"authorization": "Bearer relay-key"}),
        ("openai", "x-api-key", {"x-api-key": "relay-key"}),
        ("anthropic", None, {"x-api-key": "relay-key"}),
        ("gemini", None, {"x-goog-api-key": "relay-key"}),
    ],
)
def test_auth_header_follows_api_and_override(
    api: str, auth: str | None, expected: dict[str, str]
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if api == "anthropic":
            return _anthropic_reply([{"type": "text", "text": "ok"}])
        if api == "gemini":
            return _gemini_reply([{"text": "ok"}])
        return _openai_reply("ok")

    endpoint = EndpointSettings(
        api=api,
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
        auth=auth,
    )
    settings = (
        _settings(
            endpoint,
            flash=ModelSettings(_RELAY, "claude-test", 64),
            evaluator=ModelSettings(_RELAY, "claude-test", 64),
        )
        if api == "anthropic"
        else _settings(endpoint)
    )
    assert _complete(settings, handler) == "ok"
    headers = seen[0].headers
    for name in ("authorization", "x-api-key", "x-goog-api-key"):
        if name in expected:
            assert headers[name] == expected[name]
        else:
            assert name not in headers


def test_custom_headers_and_user_agent_override() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    endpoint = EndpointSettings(
        api="openai",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
        headers={
            "User-Agent": "custom-agent/1.0",
            "X-Conversation": "id_{conversation_id}",
            "X-Static": "fixed\t{literal}",
        },
    )
    _complete(_settings(endpoint), handler)
    headers = seen[0].headers
    assert headers["user-agent"] == "custom-agent/1.0"
    assert re.fullmatch(r"id_[0-9a-f]{32}", headers["x-conversation"])
    assert headers["x-static"] == "fixed\t{literal}"


def test_header_encoding_error_is_reported_on_use() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("an unencodable header reached the transport")

    endpoint = EndpointSettings(
        api="openai",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
        headers={"x-test": "private café"},
    )
    with pytest.raises(ModelError, match="UnicodeEncodeError") as caught:
        _complete(_settings(endpoint), handler)
    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_openai_output_limit_is_sent_only_when_configured() -> None:
    bodies: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _openai_reply()

    endpoint = EndpointSettings(
        api="openai", base_url="https://relay.test/v1", api_key_env="RELAY_KEY"
    )
    settings = _settings(endpoint, flash=ModelSettings(_RELAY, "fast", 128))
    _complete(settings, handler)
    assert bodies == [
        {
            "model": "fast",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Summarise the abstract."},
            ],
            "stream": False,
            "max_completion_tokens": 128,
        }
    ]


def test_openai_reasoning_content_is_ignored() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Answer.",
                            "reasoning_content": "Hidden thinking.",
                        },
                    }
                ]
            },
        )

    assert _complete(Settings(), handler) == "Answer."


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "Answer.", "refusal": "I cannot."},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "assistant"},
    ],
)
def test_openai_refusal_and_missing_text_are_errors(message: dict[str, object]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": message}]}
        )

    with pytest.raises(ModelError):
        _complete(Settings(), handler)


@pytest.mark.parametrize(
    ("finish_reason", "detail"),
    [("length", "incomplete"), ("content_filter", "blocked"), (None, "incomplete")],
)
def test_openai_non_stop_finish_reason_is_an_error(
    finish_reason: str | None, detail: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {"content": "Answer."},
                    }
                ]
            },
        )

    with pytest.raises(ModelError, match=detail):
        _complete(Settings(), handler)


def test_openai_requires_exactly_one_choice() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ModelError, match="expected one choice"):
        _complete(Settings(), handler)


@pytest.mark.parametrize("body", [b"not json", b"\xff", b"[]"])
def test_malformed_reply_is_an_error(body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with pytest.raises(ModelError):
        _complete(Settings(), handler)


def test_reply_text_is_preserved_verbatim() -> None:
    reply = "  first line\nsecond line  "

    def handler(request: httpx.Request) -> httpx.Response:
        return _openai_reply(reply)

    assert _complete(Settings(), handler) == reply


def test_anthropic_request_and_text_blocks() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _anthropic_reply(
            [
                {"type": "thinking", "thinking": "Hidden."},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]
        )

    endpoint = EndpointSettings(
        api="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="RELAY_KEY",
    )
    settings = _settings(
        endpoint,
        flash=ModelSettings(_RELAY, "claude-test", 512),
        evaluator=ModelSettings(_RELAY, "claude-test", 4096),
    )

    async def action(client: ModelClient) -> str:
        return await client.complete(
            "evaluator",
            [
                Message("user", "Earlier question."),
                Message("assistant", "Earlier answer."),
                Message("user", "Assess this paper."),
            ],
            system="Assess carefully.",
        )

    assert _run(settings, handler, action) == "Hello world"
    request = seen[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "relay-key"
    assert request.headers["anthropic-version"] == "2026-10-21"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "claude-test",
        "messages": [
            {"role": "user", "content": "Earlier question."},
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Assess this paper."},
        ],
        "max_tokens": 4096,
        "system": "Assess carefully.",
    }


def test_anthropic_version_header_may_be_overridden() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _anthropic_reply([{"type": "text", "text": "ok"}])

    endpoint = EndpointSettings(
        api="anthropic",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
        headers={"Anthropic-Version": "2025-01-01"},
    )
    settings = _settings(
        endpoint,
        flash=ModelSettings(_RELAY, "claude-test", 64),
        evaluator=ModelSettings(_RELAY, "claude-test", 64),
    )
    assert _complete(settings, handler) == "ok"
    assert seen[0].headers["anthropic-version"] == "2025-01-01"


def test_anthropic_without_output_limit_is_rejected_by_configuration() -> None:
    endpoint = EndpointSettings(
        api="anthropic",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
    )
    with pytest.raises(ConfigError, match="max_output_tokens"):
        _settings(endpoint)


@pytest.mark.parametrize(
    ("stop_reason", "detail"),
    [
        ("max_tokens", "turn did not finish"),
        ("tool_use", "turn did not finish"),
        ("pause_turn", "turn did not finish"),
        ("refusal", "refused"),
        (None, "turn did not finish"),
        ([], "turn did not finish"),
    ],
)
def test_anthropic_incomplete_stop_reason_is_an_error(
    stop_reason: object, detail: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _anthropic_reply(
            [{"type": "text", "text": "partial"}], stop_reason=stop_reason
        )

    endpoint = EndpointSettings(
        api="anthropic",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
    )
    settings = _settings(
        endpoint,
        flash=ModelSettings(_RELAY, "claude", 64),
        evaluator=ModelSettings(_RELAY, "claude", 64),
    )
    with pytest.raises(ModelError, match=detail):
        _complete(settings, handler)


def test_anthropic_stop_sequence_is_a_complete_turn() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _anthropic_reply(
            [{"type": "text", "text": "done"}], stop_reason="stop_sequence"
        )

    endpoint = EndpointSettings(
        api="anthropic",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
    )
    settings = _settings(
        endpoint,
        flash=ModelSettings(_RELAY, "claude", 64),
        evaluator=ModelSettings(_RELAY, "claude", 64),
    )
    assert _complete(settings, handler) == "done"


def test_gemini_request_uses_path_model_and_instruction() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _gemini_reply([{"text": "ok"}])

    endpoint = EndpointSettings(
        api="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="RELAY_KEY",
    )
    settings = _settings(
        endpoint,
        evaluator=ModelSettings(_RELAY, "models/gemini-4-argon", 256),
    )

    async def action(client: ModelClient) -> str:
        return await client.complete(
            "evaluator",
            [
                Message("user", "Assess."),
                Message("assistant", "Prior."),
                Message("user", "Continue."),
            ],
            system="Be careful.",
        )

    assert _run(settings, handler, action) == "ok"
    request = seen[0]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-4-argon:generateContent"
    )
    assert request.headers["x-goog-api-key"] == "relay-key"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert "model" not in body
    assert body == {
        "contents": [
            {"role": "user", "parts": [{"text": "Assess."}]},
            {"role": "model", "parts": [{"text": "Prior."}]},
            {"role": "user", "parts": [{"text": "Continue."}]},
        ],
        "systemInstruction": {"parts": [{"text": "Be careful."}]},
        "generationConfig": {"maxOutputTokens": 256},
    }


def test_gemini_model_id_is_percent_encoded_in_the_path() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _gemini_reply([{"text": "ok"}])

    endpoint = EndpointSettings(
        api="gemini",
        base_url="https://relay.test/v1beta",
        api_key_env="RELAY_KEY",
    )
    settings = _settings(endpoint, flash=ModelSettings(_RELAY, "gemini 4/argon?x"))
    assert _complete(settings, handler) == "ok"
    assert str(seen[0].url) == (
        "https://relay.test/v1beta/models/gemini%204%2Fargon%3Fx:generateContent"
    )


def test_gemini_skips_thought_parts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _gemini_reply(
            [
                {"text": "Hidden.", "thought": True},
                {"text": "Visible "},
                {"inlineData": {"mimeType": "image/png", "data": "..."}},
                {"text": "answer"},
            ]
        )

    endpoint = EndpointSettings(
        api="gemini",
        base_url="https://relay.test/v1beta",
        api_key_env="RELAY_KEY",
    )
    assert _complete(_settings(endpoint), handler) == "Visible answer"


def test_gemini_blocked_prompt_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []},
        )

    endpoint = EndpointSettings(
        api="gemini",
        base_url="https://relay.test/v1beta",
        api_key_env="RELAY_KEY",
    )
    with pytest.raises(ModelError, match="blocked"):
        _complete(_settings(endpoint), handler)


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({}, "expected one candidate"),
        ({"candidates": []}, "expected one candidate"),
        (
            {"promptFeedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"}},
            "no candidate",
        ),
        (
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "cut"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            },
            "did not finish",
        ),
        (
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "x"}]},
                        "finishReason": "SAFETY",
                    }
                ]
            },
            "blocked",
        ),
    ],
)
def test_gemini_incomplete_candidates_are_errors(
    body: dict[str, object], detail: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    endpoint = EndpointSettings(
        api="gemini",
        base_url="https://relay.test/v1beta",
        api_key_env="RELAY_KEY",
    )
    with pytest.raises(ModelError, match=detail):
        _complete(_settings(endpoint), handler)


@pytest.mark.parametrize(
    ("api", "reply"),
    [
        ("anthropic", _anthropic_reply([{"type": "thinking", "thinking": "hidden"}])),
        ("gemini", _gemini_reply([{"text": "hidden", "thought": True}])),
    ],
)
def test_reasoning_only_reply_has_no_visible_text(
    api: str, reply: httpx.Response
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return reply

    endpoint = EndpointSettings(
        api=api, base_url="https://relay.test/v1", api_key_env="RELAY_KEY"
    )
    settings = _settings(
        endpoint,
        flash=ModelSettings(_RELAY, "flash-model", 64),
        evaluator=ModelSettings(_RELAY, "eval-model", 64),
    )
    with pytest.raises(ModelError, match="no visible text"):
        _complete(settings, handler)


def test_gemini_unspecified_block_reason_does_not_hide_valid_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _gemini_reply([{"text": "ok"}]).json()
        body["promptFeedback"] = {"blockReason": "BLOCK_REASON_UNSPECIFIED"}
        return httpx.Response(200, json=body)

    endpoint = EndpointSettings(
        api="gemini",
        base_url="https://relay.test/v1beta",
        api_key_env="RELAY_KEY",
    )
    assert _complete(_settings(endpoint), handler) == "ok"


@pytest.mark.parametrize(
    ("role", "messages", "detail"),
    [
        ("unknown", [Message("user", "screen")], "model role"),
        ("flash", [], "no messages"),
    ],
)
def test_invalid_completion_arguments_do_not_send_a_request(
    role: ModelRole, messages: list[Message], detail: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("invalid completion sent an HTTP request")

    async def action(client: ModelClient) -> str:
        return await client.complete(role, messages)

    with pytest.raises(ModelError, match=detail):
        _run(Settings(), handler, action)


def test_http_error_reports_status_without_server_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": {"message": "key sk-secret exhausted"}}
        )

    with pytest.raises(ModelError) as caught:
        _complete(Settings(), handler)
    message = str(caught.value)
    assert "HTTP status 429" in message
    assert "sk-secret" not in message
    assert "Summarise" not in message


def test_post_is_never_replayed() -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, headers={"retry-after": "0"})

    with pytest.raises(ModelError, match="HTTP status 503"):
        _complete(Settings(), handler)
    assert len(attempts) == 1


def test_rate_limit_pauses_the_other_role_on_the_shared_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(http.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(http.asyncio, "sleep", clock.sleep)
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        attempts.append(model)
        if model == "mimo-v2.6-pro":
            return httpx.Response(429, headers={"retry-after": "2"})
        return _openai_reply()

    async def action(client: ModelClient) -> str:
        with pytest.raises(ModelError, match="HTTP status 429"):
            await client.complete("evaluator", [Message("user", "assess")])
        return await client.complete("flash", [Message("user", "screen")])

    assert _run(Settings(), handler, action) == "Screened."
    assert attempts == ["mimo-v2.6-pro", "deepseek-v4.1-flash"]
    assert clock.sleeps == [2.0]


def test_rate_limit_scope_can_span_endpoint_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(http.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(http.asyncio, "sleep", clock.sleep)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "flash.test":
            return httpx.Response(429, headers={"retry-after": "3"})
        return _openai_reply()

    settings = Settings(
        endpoints={
            "flash-endpoint": EndpointSettings(
                api="openai",
                base_url="https://flash.test/v1",
                api_key_env="RELAY_KEY",
                quota_scope="shared-go",
            ),
            "evaluator-endpoint": EndpointSettings(
                api="openai",
                base_url="https://eval.test/v1",
                api_key_env="RELAY_KEY",
                quota_scope="shared-go",
            ),
        },
        flash_model=ModelSettings("flash-endpoint", "fast"),
        evaluator_model=ModelSettings("evaluator-endpoint", "strong"),
    )

    async def action(client: ModelClient) -> str:
        with pytest.raises(ModelError, match="HTTP status 429"):
            await client.complete("flash", [Message("user", "screen")])
        return await client.complete("evaluator", [Message("user", "assess")])

    assert _run(settings, handler, action) == "Screened."
    assert seen == ["flash.test", "eval.test"]
    assert clock.sleeps == [3.0]


@pytest.mark.parametrize("scope", [None, "endpoint:evaluator-endpoint"])
def test_separate_profiles_do_not_share_an_implicit_quota(
    monkeypatch: pytest.MonkeyPatch, scope: str | None
) -> None:
    clock = Clock()
    monkeypatch.setattr(http.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(http.asyncio, "sleep", clock.sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "flash.test":
            return httpx.Response(429, headers={"retry-after": "3"})
        return _openai_reply()

    settings = Settings(
        endpoints={
            "flash-endpoint": EndpointSettings(
                api="openai",
                base_url="https://flash.test/v1",
                api_key_env="RELAY_KEY",
                quota_scope=scope,
            ),
            "evaluator-endpoint": EndpointSettings(
                api="openai",
                base_url="https://eval.test/v1",
                api_key_env="RELAY_KEY",
            ),
        },
        flash_model=ModelSettings("flash-endpoint", "fast"),
        evaluator_model=ModelSettings("evaluator-endpoint", "strong"),
    )

    async def action(client: ModelClient) -> str:
        with pytest.raises(ModelError, match="HTTP status 429"):
            await client.complete("flash", [Message("user", "screen")])
        return await client.complete("evaluator", [Message("user", "assess")])

    assert _run(settings, handler, action) == "Screened."
    assert clock.sleeps == []


def test_missing_api_key_names_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RELAY_KEY")
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _openai_reply()

    endpoint = EndpointSettings(
        api="openai", base_url="https://relay.test/v1", api_key_env="RELAY_KEY"
    )
    with pytest.raises(ModelError, match="RELAY_KEY"):
        _complete(_settings(endpoint), handler)
    assert calls == []


def test_named_credential_is_read_from_the_default_location(
    xdg_config_home: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    _credentials_file(
        xdg_config_home / "researcher" / "credentials.toml",
        'opencode = "go-file-key"\n',
    )
    assert _complete(Settings(), handler) == "Screened."
    assert seen[0].headers["authorization"] == "Bearer go-file-key"


def test_explicit_credentials_path_overrides_the_default_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    custom = tmp_path / "custom"
    custom.mkdir()
    path = _credentials_file(custom / "credentials.toml", 'opencode = "go-file-key"\n')
    assert _complete(Settings(), handler, credentials_path=path) == "Screened."
    assert seen[0].headers["authorization"] == "Bearer go-file-key"


def test_environment_key_is_used_without_reading_the_file(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    absent = tmp_path / "absent" / "credentials.toml"
    assert _complete(Settings(), handler, credentials_path=absent) == "Screened."
    assert seen[0].headers["authorization"] == "Bearer go-key"


def test_credential_only_endpoint_uses_the_named_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RELAY_KEY")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    path = _credentials_file(
        tmp_path / "credentials.toml", 'relay = "relay-file-key"\n'
    )
    endpoint = EndpointSettings(
        api="openai", base_url="https://relay.test/v1", credential="relay"
    )
    assert _complete(_settings(endpoint), handler, credentials_path=path) == "Screened."
    assert seen[0].headers["authorization"] == "Bearer relay-file-key"


def test_environment_only_endpoint_needs_no_credentials_file(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    endpoint = EndpointSettings(
        api="openai", base_url="https://relay.test/v1", api_key_env="RELAY_KEY"
    )
    absent = tmp_path / "absent.toml"
    assert (
        _complete(_settings(endpoint), handler, credentials_path=absent) == "Screened."
    )
    assert seen[0].headers["authorization"] == "Bearer relay-key"


def test_named_credential_is_shared_by_endpoint_profiles(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _openai_reply()

    path = _credentials_file(tmp_path / "credentials.toml", 'shared = "shared-key"\n')
    settings = Settings(
        endpoints={
            "flash-endpoint": EndpointSettings(
                api="openai", base_url="https://flash.test/v1", credential="shared"
            ),
            "evaluator-endpoint": EndpointSettings(
                api="openai", base_url="https://eval.test/v1", credential="shared"
            ),
        },
        flash_model=ModelSettings("flash-endpoint", "fast"),
        evaluator_model=ModelSettings("evaluator-endpoint", "strong"),
    )

    async def action(client: ModelClient) -> None:
        await client.complete("flash", [Message("user", "screen")])
        await client.complete("evaluator", [Message("user", "assess")])

    _run(settings, handler, action, credentials_path=path)
    assert [request.headers["authorization"] for request in seen] == [
        "Bearer shared-key",
        "Bearer shared-key",
    ]


@pytest.mark.parametrize("value", ["", "go-key\r\nx-injected: yes"])
def test_unusable_environment_key_does_not_fall_back_to_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", value)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _openai_reply()

    path = _credentials_file(
        tmp_path / "credentials.toml", 'opencode = "go-file-key"\n'
    )
    with pytest.raises(ModelError, match="OPENCODE_API_KEY") as caught:
        _complete(Settings(), handler, credentials_path=path)
    assert calls == []
    assert "go-file-key" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_missing_named_credential_is_reported_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a request was sent without an API key")

    path = _credentials_file(tmp_path / "credentials.toml", 'other = "other-secret"\n')
    with pytest.raises(ModelError) as caught:
        _complete(Settings(), handler, credentials_path=path)
    message = str(caught.value)
    assert "opencode" in message
    assert str(path) in message
    assert "other-secret" not in message


def test_missing_credentials_file_is_reported_without_a_chained_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a request was sent without an API key")

    absent = tmp_path / "absent" / "credentials.toml"
    with pytest.raises(ModelError, match="credentials file not found") as caught:
        _complete(Settings(), handler, credentials_path=absent)
    assert str(absent) in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_invalid_named_credential_is_not_echoed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a request was sent with an invalid API key")

    path = _credentials_file(
        tmp_path / "credentials.toml", 'opencode = "go key with spaces"\n'
    )
    with pytest.raises(ModelError, match="does not hold an API key") as caught:
        _complete(Settings(), handler, credentials_path=path)
    assert "go key with spaces" not in str(caught.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_insecure_credentials_file_is_reported_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a request was sent from an insecure credentials file")

    path = _credentials_file(
        tmp_path / "credentials.toml", 'opencode = "go-file-key"\n', 0o644
    )
    with pytest.raises(ModelError, match="chmod 600") as caught:
        _complete(Settings(), handler, credentials_path=path)
    assert "go-file-key" not in str(caught.value)


def test_named_credential_without_a_location_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a request was sent without an API key")

    with pytest.raises(ModelError, match="XDG_CONFIG_HOME") as caught:
        _complete(Settings(), handler)
    assert "opencode" in str(caught.value)


def test_invalid_api_key_is_not_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "go-key\r\nx-injected: yes")

    def handler(request: httpx.Request) -> httpx.Response:
        return _openai_reply()

    with pytest.raises(ModelError) as caught:
        _complete(Settings(), handler)
    assert "injected" not in str(caught.value)
    assert "go-key" not in str(caught.value)


@pytest.mark.parametrize(
    "error_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError]
)
def test_transport_failures_become_model_errors(
    error_type: type[httpx.TransportError],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error_type("transport details containing go-key", request=request)

    with pytest.raises(ModelError) as caught:
        _complete(Settings(), handler)
    assert error_type.__name__ in str(caught.value)
    assert "go-key" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_total_timeout_becomes_a_model_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return _openai_reply()

    async def action(client: ModelClient) -> str:
        return await client.complete("flash", [Message("user", "screen")])

    with pytest.raises(ModelError, match="timed out"):
        _run(Settings(), handler, action, total_timeout=0.01)
