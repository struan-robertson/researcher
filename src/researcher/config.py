"""Effective research settings from built-in defaults, TOML and CLI overrides."""

import math
import os
import re
from pathlib import Path
from typing import Annotated, Literal

import msgspec

from researcher.errors import UserError

DEFAULT_MAX_PAPERS = 25
DEFAULT_TIME_LIMIT = "1h"
DEFAULT_ENDPOINT = "opencode"
DEFAULT_QUOTA_SCOPE = "opencode-go"
DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_CREDENTIAL = "opencode"
DEFAULT_API_KEY_ENV = "OPENCODE_API_KEY"
CREDENTIALS_FILE = "credentials.toml"
DEFAULT_FLASH_MODEL_ID = "deepseek-v4.1-flash"
DEFAULT_EVALUATOR_MODEL_ID = "mimo-v2.6-pro"

ApiFormat = Literal["openai", "anthropic", "gemini"]
AuthMode = Literal["bearer", "x-api-key", "x-goog-api-key"]

_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0}
_DURATION = re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[smh])")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "content-type",
        "content-length",
        "transfer-encoding",
        "host",
    }
)
_DEFAULT_AUTH: dict[str, AuthMode] = {
    "openai": "bearer",
    "anthropic": "x-api-key",
    "gemini": "x-goog-api-key",
}
CONVERSATION_TOKEN = "{conversation_id}"


class ConfigError(UserError):
    """A configuration file or setting is invalid."""


def parse_duration(value: str) -> float:
    """Convert a duration such as ``90m`` or ``0.5h`` to positive seconds."""
    match = _DURATION.fullmatch(value)
    if match is not None:
        seconds = float(match["value"]) * _UNIT_SECONDS[match["unit"]]
        if math.isfinite(seconds) and seconds > 0:
            return seconds
    raise ConfigError(
        f"invalid duration {value!r}: expected a positive number with unit "
        "s, m or h, such as 0.5h"
    )


def _validate_headers(headers: dict[str, str]) -> None:
    """Prevent configuration from overriding or silently replacing headers."""
    seen: set[str] = set()
    for name in headers:
        folded = name.lower()
        if folded in _RESERVED_HEADERS:
            raise ConfigError(f"endpoint header {name!r} is managed by the client")
        if folded in seen:
            raise ConfigError(f"endpoint header {name!r} repeats another header")
        seen.add(folded)


class EndpointSettings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One model API endpoint profile, independent of the research roles.

    ``credential`` names a key in the credentials file, and ``api_key_env``
    names an environment variable that takes precedence when set, so a profile
    needs at least one of them.
    """

    api: ApiFormat
    base_url: str
    credential: str | None = None
    api_key_env: str | None = None
    auth: AuthMode | None = None
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    quota_scope: str | None = None

    def __post_init__(self) -> None:
        if self.credential is None and self.api_key_env is None:
            raise ConfigError("endpoint must set credential, api_key_env or both")
        if self.credential is not None and not self.credential.strip():
            raise ConfigError("endpoint credential must not be blank")
        if (
            self.api_key_env is not None
            and _ENVIRONMENT_NAME.fullmatch(self.api_key_env) is None
        ):
            raise ConfigError("endpoint api_key_env must name an environment variable")
        if self.quota_scope is not None and not self.quota_scope.strip():
            raise ConfigError("endpoint quota_scope must not be blank")
        _validate_headers(self.headers)

    @property
    def auth_mode(self) -> AuthMode:
        return self.auth if self.auth is not None else _DEFAULT_AUTH[self.api]


class ModelSettings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One model role: its endpoint profile, model ID and output limit."""

    endpoint: str
    model_id: str
    max_output_tokens: Annotated[int, msgspec.Meta(gt=0)] | None = None

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ConfigError("model endpoint must not be empty")
        if not self.model_id.strip():
            raise ConfigError("model_id must not be empty")


def _default_endpoints() -> dict[str, EndpointSettings]:
    return {
        DEFAULT_ENDPOINT: EndpointSettings(
            api="openai",
            base_url=DEFAULT_BASE_URL,
            credential=DEFAULT_CREDENTIAL,
            api_key_env=DEFAULT_API_KEY_ENV,
            headers={"x-opencode-session": f"ses_{CONVERSATION_TOKEN}"},
            quota_scope=DEFAULT_QUOTA_SCOPE,
        )
    }


class Settings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Research settings as decoded from TOML and overridden on the command line."""

    max_papers: Annotated[int, msgspec.Meta(gt=0)] = DEFAULT_MAX_PAPERS
    time_limit: str = DEFAULT_TIME_LIMIT
    endpoints: dict[str, EndpointSettings] = msgspec.field(
        default_factory=_default_endpoints
    )
    flash_model: ModelSettings = msgspec.field(
        default_factory=lambda: ModelSettings(
            endpoint=DEFAULT_ENDPOINT, model_id=DEFAULT_FLASH_MODEL_ID
        )
    )
    evaluator_model: ModelSettings = msgspec.field(
        default_factory=lambda: ModelSettings(
            endpoint=DEFAULT_ENDPOINT, model_id=DEFAULT_EVALUATOR_MODEL_ID
        )
    )

    def __post_init__(self) -> None:
        parse_duration(self.time_limit)
        _validate_models(self.endpoints, self.flash_model, self.evaluator_model)

    @property
    def time_limit_seconds(self) -> float:
        return parse_duration(self.time_limit)


def _validate_models(
    endpoints: dict[str, EndpointSettings],
    flash_model: ModelSettings,
    evaluator_model: ModelSettings,
) -> None:
    """Check role references and the output limit the selected wire API needs."""
    if not endpoints:
        raise ConfigError("at least one endpoint must be configured")
    for label, model in (
        ("flash_model", flash_model),
        ("evaluator_model", evaluator_model),
    ):
        endpoint = endpoints.get(model.endpoint)
        if endpoint is None:
            raise ConfigError(
                f"{label} refers to endpoint {model.endpoint!r}, which is not "
                "configured"
            )
        if endpoint.api == "anthropic" and model.max_output_tokens is None:
            raise ConfigError(
                f"{label} requires max_output_tokens for the Anthropic API"
            )
        if (
            endpoint.api == "gemini"
            and not model.model_id.removeprefix("models/").strip()
        ):
            raise ConfigError(f"{label} model_id must name a Gemini model")


def default_config_path() -> Path | None:
    base = os.environ.get("XDG_CONFIG_HOME")
    return Path(base) / "researcher" / "config.toml" if base else None


def credentials_path(config_path: Path | None = None) -> Path | None:
    """Locate the credentials file beside the selected or default configuration."""
    if config_path is not None:
        return config_path.parent / CREDENTIALS_FILE
    path = default_config_path()
    return None if path is None else path.parent / CREDENTIALS_FILE


def resolve_settings(
    config_path: Path | None = None,
    *,
    max_papers: int | None = None,
    time_limit: str | None = None,
) -> Settings:
    """Combine built-in defaults, an optional TOML file and explicit overrides.

    ``config_path`` names an explicit file, which must exist. Without it, only
    an existing configuration file under XDG_CONFIG_HOME is considered.
    Overrides pass through the same decoding as the file, so they cannot bypass
    the setting types or constraints.
    """
    if config_path is not None:
        settings = load_settings_file(config_path)
    else:
        path = default_config_path()
        settings = (
            load_settings_file(path)
            if path is not None and path.exists()
            else Settings()
        )
    if max_papers is None and time_limit is None:
        return settings
    values = msgspec.structs.asdict(settings)
    if max_papers is not None:
        values["max_papers"] = max_papers
    if time_limit is not None:
        values["time_limit"] = time_limit
    try:
        return msgspec.convert(values, Settings)
    except (msgspec.ValidationError, ConfigError) as exc:
        raise ConfigError(f"invalid command-line settings: {exc}") from exc


def load_settings_file(path: Path) -> Settings:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ConfigError(
            f"cannot read configuration file {path}: {exc.strerror or exc}"
        ) from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"configuration file is not valid UTF-8: {path}") from exc
    try:
        return msgspec.toml.decode(text, type=Settings)
    except (msgspec.DecodeError, ConfigError) as exc:
        raise ConfigError(f"invalid configuration file {path}: {exc}") from exc
