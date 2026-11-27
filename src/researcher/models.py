"""Text-only model completion over the OpenCode, OpenAI, Anthropic and Gemini APIs."""

import os
import uuid
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx
import msgspec

from researcher.config import (
    CONVERSATION_TOKEN,
    ApiFormat,
    EndpointSettings,
    ModelSettings,
    Settings,
    credentials_path,
)
from researcher.credentials import CredentialError, load_credentials
from researcher.errors import UserError
from researcher.http import HTTPClient, Quota

ANTHROPIC_VERSION = "2026-10-21"
USER_AGENT = f"researcher/{version('researcher')}"

_ANTHROPIC_COMPLETE = ("end_turn", "stop_sequence")

ModelRole = Literal["flash", "evaluator"]


class ModelError(UserError):
    """A model request could not produce visible text."""


class Message(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One plain text turn supplied to a model."""

    role: Literal["user", "assistant"]
    text: str


class ModelClient:
    """Non-streaming text completion through the configured endpoint profiles.

    The caller owns the HTTP client and the request timeout policy. Quotas live
    here so that every role and conversation configured into the same scope
    observes the same cooldowns. ``credentials_path`` overrides the default
    location of the credentials file that holds named API keys.
    """

    def __init__(
        self,
        http: HTTPClient,
        settings: Settings,
        *,
        credentials_path: Path | None = None,
    ) -> None:
        self._http = http
        self._settings = settings
        self._credentials_path = credentials_path
        self._quotas: dict[tuple[str, str], Quota] = {}

    async def complete(
        self,
        role: ModelRole,
        messages: Sequence[Message],
        *,
        system: str | None = None,
    ) -> str:
        """Send one independent completion request and return its visible text."""
        if role not in ("flash", "evaluator"):
            raise ModelError("model role must be flash or evaluator")
        if not messages:
            raise ModelError("no messages were supplied")
        model = (
            self._settings.flash_model
            if role == "flash"
            else self._settings.evaluator_model
        )
        endpoint = self._settings.endpoints[model.endpoint]
        url, body = _request(endpoint, model, messages, system)
        headers = _headers(endpoint, self._api_key(endpoint))
        try:
            target = httpx.URL(url)
            if target.is_relative_url or target.userinfo:
                raise ModelError(
                    f"endpoint {model.endpoint!r} requires an absolute URL without "
                    "embedded credentials"
                )
            response = await self._http.request(
                "POST",
                url,
                quota=self._quota(model.endpoint, endpoint),
                replay_safe=False,
                headers=headers,
                json=body,
            )
        except TimeoutError:
            raise ModelError(
                f"{role} model request to endpoint {model.endpoint!r} timed out"
            ) from None
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeEncodeError) as exc:
            raise ModelError(
                f"{role} model request to endpoint {model.endpoint!r} failed: "
                f"{type(exc).__name__}"
            ) from None
        if response.status_code != 200:
            raise ModelError(
                f"{role} model request to endpoint {model.endpoint!r} failed with "
                f"HTTP status {response.status_code}"
            )
        text = _reply_text(endpoint.api, response.content, role, model.endpoint)
        if not text.strip():
            raise ModelError(
                f"{role} model reply from endpoint {model.endpoint!r} contained no "
                "visible text"
            )
        return text

    def _api_key(self, endpoint: EndpointSettings) -> str:
        """Resolve one endpoint's key, reading the credentials file if needed.

        A set environment variable wins even when its value is unusable, so an
        override that cannot be used is reported rather than replaced by the
        named key.
        """
        if endpoint.api_key_env is not None:
            value = os.environ.get(endpoint.api_key_env)
            if value is not None:
                if not value or not _is_visible(value):
                    raise ModelError(
                        f"environment variable {endpoint.api_key_env} does not "
                        "hold an API key"
                    )
                return value
        if endpoint.credential is None:
            raise ModelError(f"environment variable {endpoint.api_key_env} is not set")
        path = self._credentials_path
        if path is None:
            path = credentials_path()
        if path is None:
            raise ModelError(
                f"credential {endpoint.credential!r} needs a credentials file: "
                "set XDG_CONFIG_HOME or select a configuration file"
            )
        try:
            credentials = load_credentials(path)
        except CredentialError as exc:
            raise ModelError(str(exc)) from None
        value = credentials.get(endpoint.credential)
        if value is None:
            raise ModelError(
                f"credentials file {path} has no credential {endpoint.credential!r}"
            )
        if not value or not _is_visible(value):
            raise ModelError(
                f"credential {endpoint.credential!r} in {path} does not hold an API key"
            )
        return value

    def _quota(self, name: str, endpoint: EndpointSettings) -> Quota:
        """Return the quota shared by the endpoint's documented scope."""
        key = (
            ("scope", endpoint.quota_scope)
            if endpoint.quota_scope is not None
            else ("endpoint", name)
        )
        quota = self._quotas.get(key)
        if quota is None:
            quota = self._quotas[key] = Quota()
        return quota


def _is_visible(value: str) -> bool:
    return all(0x21 <= ord(character) <= 0x7E for character in value)


def _request(
    endpoint: EndpointSettings,
    model: ModelSettings,
    messages: Sequence[Message],
    system: str | None,
) -> tuple[str, dict[str, Any]]:
    """Build the provider URL and JSON body for one completion."""
    base = endpoint.base_url.rstrip("/")
    if endpoint.api == "anthropic":
        body: dict[str, Any] = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.text} for m in messages],
            "max_tokens": model.max_output_tokens,
        }
        if system is not None:
            body["system"] = system
        return f"{base}/messages", body
    if endpoint.api == "gemini":
        body = {
            "contents": [
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [{"text": m.text}],
                }
                for m in messages
            ]
        }
        if system is not None:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if model.max_output_tokens is not None:
            body["generationConfig"] = {"maxOutputTokens": model.max_output_tokens}
        name = quote(model.model_id.removeprefix("models/"), safe="")
        return f"{base}/models/{name}:generateContent", body
    body = {
        "model": model.model_id,
        "messages": _openai_messages(messages, system),
        "stream": False,
    }
    if model.max_output_tokens is not None:
        body["max_completion_tokens"] = model.max_output_tokens
    return f"{base}/chat/completions", body


def _openai_messages(
    messages: Sequence[Message], system: str | None
) -> list[dict[str, str]]:
    turns = [{"role": m.role, "content": m.text} for m in messages]
    if system is None:
        return turns
    return [{"role": "system", "content": system}, *turns]


def _headers(endpoint: EndpointSettings, api_key: str) -> dict[str, str]:
    """Compose custom headers, the application identity and the credential."""
    conversation_id = uuid.uuid4().hex
    headers = {
        name.lower(): value.replace(CONVERSATION_TOKEN, conversation_id)
        for name, value in endpoint.headers.items()
    }
    headers.setdefault("user-agent", USER_AGENT)
    if endpoint.api == "anthropic":
        headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
    mode = endpoint.auth_mode
    if mode == "bearer":
        headers["authorization"] = f"Bearer {api_key}"
    else:
        headers[mode] = api_key
    return headers


def _reply_text(api: ApiFormat, content: bytes, role: ModelRole, endpoint: str) -> str:
    try:
        reply = msgspec.json.decode(content)
    except msgspec.DecodeError:
        raise _reply_error(role, endpoint, "was not valid JSON") from None
    if api == "anthropic":
        return _anthropic_text(reply, role, endpoint)
    if api == "gemini":
        return _gemini_text(reply, role, endpoint)
    return _openai_text(reply, role, endpoint)


def _reply_error(role: ModelRole, endpoint: str, detail: str) -> ModelError:
    return ModelError(f"{role} model reply from endpoint {endpoint!r} {detail}")


def _openai_text(reply: Any, role: ModelRole, endpoint: str) -> str:
    choices = reply.get("choices") if isinstance(reply, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise _reply_error(role, endpoint, "was malformed: expected one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _reply_error(role, endpoint, "was malformed: choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _reply_error(role, endpoint, "was malformed: missing message")
    if message.get("refusal") not in (None, ""):
        raise _reply_error(role, endpoint, "was refused by the model")
    finish = choice.get("finish_reason")
    if finish == "content_filter":
        raise _reply_error(role, endpoint, "was blocked by the content filter")
    if finish != "stop":
        raise _reply_error(role, endpoint, "was incomplete: the model did not finish")
    text = message.get("content")
    if not isinstance(text, str):
        raise _reply_error(role, endpoint, "was malformed: message content is not text")
    return text


def _anthropic_text(reply: Any, role: ModelRole, endpoint: str) -> str:
    if not isinstance(reply, dict):
        raise _reply_error(role, endpoint, "was malformed: expected an object")
    stop = reply.get("stop_reason")
    if stop == "refusal":
        raise _reply_error(role, endpoint, "was refused by the model")
    if stop not in _ANTHROPIC_COMPLETE:
        raise _reply_error(role, endpoint, "was incomplete: the turn did not finish")
    blocks = reply.get("content")
    if not isinstance(blocks, list):
        raise _reply_error(role, endpoint, "was malformed: missing content blocks")
    texts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise _reply_error(
                role, endpoint, "was malformed: content block is not an object"
            )
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            raise _reply_error(role, endpoint, "was malformed: text block has no text")
        texts.append(text)
    return "".join(texts)


def _gemini_text(reply: Any, role: ModelRole, endpoint: str) -> str:
    if not isinstance(reply, dict):
        raise _reply_error(role, endpoint, "was malformed: expected an object")
    feedback = reply.get("promptFeedback")
    candidates = reply.get("candidates")
    if isinstance(feedback, dict):
        if feedback.get("blockReason") not in (None, "BLOCK_REASON_UNSPECIFIED"):
            raise _reply_error(role, endpoint, "was blocked before generation")
        if candidates is None or candidates == []:
            raise _reply_error(role, endpoint, "contained no candidate for the prompt")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise _reply_error(role, endpoint, "was malformed: expected one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise _reply_error(role, endpoint, "was malformed: candidate is not an object")
    finish = candidate.get("finishReason")
    if finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
        raise _reply_error(role, endpoint, "was blocked during generation")
    if finish != "STOP":
        raise _reply_error(
            role, endpoint, "was incomplete: the candidate did not finish"
        )
    content = candidate.get("content")
    if not isinstance(content, dict):
        raise _reply_error(role, endpoint, "was malformed: missing candidate content")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise _reply_error(role, endpoint, "was malformed: missing candidate parts")
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise _reply_error(role, endpoint, "was malformed: part is not an object")
        if part.get("thought") is True:
            continue
        text = part.get("text")
        if text is None:
            continue
        if not isinstance(text, str):
            raise _reply_error(role, endpoint, "was malformed: part text is not text")
        texts.append(text)
    return "".join(texts)
