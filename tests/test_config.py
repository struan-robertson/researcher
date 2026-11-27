import textwrap
from pathlib import Path

import pytest

from researcher.config import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_CREDENTIAL,
    DEFAULT_ENDPOINT,
    DEFAULT_EVALUATOR_MODEL_ID,
    DEFAULT_FLASH_MODEL_ID,
    DEFAULT_MAX_PAPERS,
    DEFAULT_TIME_LIMIT,
    ConfigError,
    EndpointSettings,
    ModelSettings,
    Settings,
    credentials_path,
    default_config_path,
    parse_duration,
    resolve_settings,
)


@pytest.fixture
def empty_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_home = tmp_path / "config-home"
    config_home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    return config_home


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text('max_papers = 10\ntime_limit = "30m"\n', encoding="utf-8")
    return path


def test_builtin_defaults(empty_xdg: Path) -> None:
    settings = resolve_settings()
    assert settings.max_papers == 25
    assert settings.time_limit == "1h"
    assert settings.time_limit_seconds == 3600.0
    assert DEFAULT_MAX_PAPERS == 25
    assert DEFAULT_TIME_LIMIT == "1h"


def test_missing_default_config_uses_defaults(empty_xdg: Path) -> None:
    assert not default_config_path().exists()
    assert resolve_settings().max_papers == DEFAULT_MAX_PAPERS


def test_explicit_missing_config_is_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent.toml"
    with pytest.raises(ConfigError, match="configuration file not found"):
        resolve_settings(missing)


def test_default_config_file_is_loaded(empty_xdg: Path) -> None:
    path = empty_xdg / "researcher" / "config.toml"
    path.parent.mkdir()
    path.write_text('max_papers = 7\ntime_limit = "45m"\n', encoding="utf-8")
    settings = resolve_settings()
    assert settings.max_papers == 7
    assert settings.time_limit_seconds == 2700.0


def test_explicit_config_file_is_loaded(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    settings = resolve_settings(config_file)
    assert settings == Settings(max_papers=10, time_limit="30m")
    assert settings.time_limit_seconds == 1800.0


def test_explicit_config_replaces_default_config(
    empty_xdg: Path, config_file: Path
) -> None:
    default = empty_xdg / "researcher" / "config.toml"
    default.parent.mkdir()
    default.write_text("max_papers = 3\n", encoding="utf-8")
    assert resolve_settings(config_file).max_papers == 10


def test_cli_overrides_beat_config_file(config_file: Path) -> None:
    settings = resolve_settings(config_file, max_papers=40, time_limit="1m")
    assert settings == Settings(max_papers=40, time_limit="1m")
    assert settings.time_limit_seconds == 60.0


def test_single_cli_override_keeps_file_value(config_file: Path) -> None:
    settings = resolve_settings(config_file, max_papers=40)
    assert settings == Settings(max_papers=40, time_limit="30m")


def test_cli_overrides_beat_defaults(empty_xdg: Path) -> None:
    settings = resolve_settings(max_papers=5, time_limit="30s")
    assert settings == Settings(max_papers=5, time_limit="30s")


@pytest.mark.parametrize("max_papers", [0, -3])
def test_non_positive_cli_max_papers_is_rejected(
    config_file: Path, max_papers: int
) -> None:
    with pytest.raises(ConfigError, match="invalid command-line settings"):
        resolve_settings(config_file, max_papers=max_papers)


@pytest.mark.parametrize("time_limit", ["0h", "-1h", "1d", "soon"])
def test_invalid_cli_time_limit_is_rejected(config_file: Path, time_limit: str) -> None:
    with pytest.raises(ConfigError, match="invalid command-line settings"):
        resolve_settings(config_file, time_limit=time_limit)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("max_papers = 5\nretries = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown field `retries`"):
        resolve_settings(path)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[provider]\nmodel = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown field `provider`"):
        resolve_settings(path)


def test_malformed_toml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("max_papers = \n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid configuration file"):
        resolve_settings(path)


def test_non_utf8_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b"max_papers = \xff\n")
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        resolve_settings(path)


@pytest.mark.parametrize("value", [0, -3])
def test_non_positive_max_papers_is_rejected(tmp_path: Path, value: int) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"max_papers = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Expected `int` >= 1"):
        resolve_settings(path)


@pytest.mark.parametrize(
    ("value", "expected"), [("true", "bool"), ('"25"', "str"), ("2.5", "float")]
)
def test_wrong_max_papers_type_is_rejected(
    tmp_path: Path, value: str, expected: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"max_papers = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=f"Expected `int`, got `{expected}`"):
        resolve_settings(path)


def test_wrong_time_limit_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("time_limit = 60\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Expected `str`, got `int`"):
        resolve_settings(path)


def test_invalid_config_duration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('time_limit = "0h"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid duration"):
        resolve_settings(path)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("1h", 3600.0),
        ("0.5h", 1800.0),
        ("90m", 5400.0),
        ("1.5m", 90.0),
        ("45s", 45.0),
        ("0.25s", 0.25),
    ],
)
def test_duration_strings(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize(
    "text", ["", "h", "1", "0h", "0.0s", "-1h", "1H", "1d", "1 h", " 1h", "infh"]
)
def test_invalid_duration_strings(text: str) -> None:
    with pytest.raises(ConfigError, match="invalid duration"):
        parse_duration(text)


def test_default_config_path_uses_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert default_config_path() == tmp_path / "xdg/researcher/config.toml"


@pytest.mark.parametrize("config_home", [None, ""])
def test_no_config_is_loaded_without_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_home: str | None
) -> None:
    if config_home is None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", config_home)
    monkeypatch.setenv("HOME", str(tmp_path))
    home_config = tmp_path / ".config/researcher/config.toml"
    home_config.parent.mkdir(parents=True)
    home_config.write_text("max_papers = 3\n", encoding="utf-8")
    assert default_config_path() is None
    assert resolve_settings() == Settings()


def test_credentials_path_follows_an_explicit_configuration(tmp_path: Path) -> None:
    assert credentials_path(tmp_path / "custom.toml") == tmp_path / "credentials.toml"


def test_credentials_path_follows_the_default_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert credentials_path() == tmp_path / "xdg/researcher/credentials.toml"
    assert not default_config_path().exists()


def test_credentials_path_requires_a_configuration_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert credentials_path() is None


def test_unreadable_config_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read configuration file"):
        resolve_settings(tmp_path)


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def test_default_endpoint_and_model_roles(empty_xdg: Path) -> None:
    settings = resolve_settings()
    endpoint = settings.endpoints[DEFAULT_ENDPOINT]
    assert DEFAULT_ENDPOINT == "opencode"
    assert endpoint.api == "openai"
    assert endpoint.base_url == DEFAULT_BASE_URL
    assert endpoint.base_url == "https://opencode.ai/inference/go/openai/v1"
    assert endpoint.credential == DEFAULT_CREDENTIAL
    assert endpoint.credential == "opencode"
    assert endpoint.api_key_env == DEFAULT_API_KEY_ENV
    assert endpoint.api_key_env == "OPENCODE_API_KEY"
    assert endpoint.auth is None
    assert endpoint.auth_mode == "bearer"
    assert endpoint.headers == {"x-opencode-session": "ses_{conversation_id}"}
    assert endpoint.quota_scope == "opencode-go"
    assert settings.flash_model == ModelSettings("opencode", DEFAULT_FLASH_MODEL_ID)
    assert settings.flash_model.model_id == "deepseek-v4.1-flash"
    assert settings.evaluator_model.model_id == DEFAULT_EVALUATOR_MODEL_ID
    assert settings.evaluator_model.model_id == "mimo-v2.6-pro"
    assert settings.evaluator_model.max_output_tokens is None


@pytest.mark.parametrize(
    ("api", "mode"),
    [
        ("openai", "bearer"),
        ("anthropic", "x-api-key"),
        ("gemini", "x-goog-api-key"),
    ],
)
def test_default_auth_mode_follows_the_api(api: str, mode: str) -> None:
    endpoint = EndpointSettings(
        api=api, base_url="https://relay.test/v1", api_key_env="RELAY_KEY"
    )
    assert endpoint.auth_mode == mode


def test_auth_mode_override() -> None:
    endpoint = EndpointSettings(
        api="openai",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
        auth="x-goog-api-key",
    )
    assert endpoint.auth_mode == "x-goog-api-key"


def test_conversation_header_placeholder_and_static_headers() -> None:
    endpoint = EndpointSettings(
        api="openai",
        base_url="https://relay.test/v1",
        api_key_env="RELAY_KEY",
        headers={"x-session": "ses_{conversation_id}", "x-static": "fixed"},
    )
    assert endpoint.headers["x-session"] == "ses_{conversation_id}"


def test_configuration_loads_without_resolving_api_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RESEARCHER_UNSET_KEY", raising=False)
    path = _write_config(
        tmp_path,
        """
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        api_key_env = "RESEARCHER_UNSET_KEY"

        [flash_model]
        endpoint = "relay"
        model_id = "relay-flash"

        [evaluator_model]
        endpoint = "relay"
        model_id = "relay-eval"
        """,
    )
    settings = resolve_settings(path)
    assert settings.endpoints["relay"].api_key_env == "RESEARCHER_UNSET_KEY"


def test_explicit_endpoint_map_replaces_the_defaults(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.relay]
        api = "anthropic"
        base_url = "https://relay.test/v1"
        api_key_env = "RELAY_KEY"
        auth = "bearer"
        quota_scope = "shared"

        [endpoints.relay.headers]
        anthropic-version = "2025-01-01"
        x-conversation = "id_{conversation_id}"

        [flash_model]
        endpoint = "relay"
        model_id = "relay-flash"
        max_output_tokens = 256

        [evaluator_model]
        endpoint = "relay"
        model_id = "relay-eval"
        max_output_tokens = 4096
        """,
    )
    settings = resolve_settings(path)
    assert set(settings.endpoints) == {"relay"}
    endpoint = settings.endpoints["relay"]
    assert endpoint.auth_mode == "bearer"
    assert endpoint.quota_scope == "shared"
    assert endpoint.headers == {
        "anthropic-version": "2025-01-01",
        "x-conversation": "id_{conversation_id}",
    }
    assert settings.flash_model.max_output_tokens == 256
    assert settings.evaluator_model.max_output_tokens == 4096


def test_replaced_endpoint_map_must_configure_the_roles(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        api_key_env = "RELAY_KEY"
        """,
    )
    with pytest.raises(ConfigError, match="refers to endpoint 'opencode'"):
        resolve_settings(path)


def test_cli_overrides_preserve_endpoint_configuration(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        api_key_env = "RELAY_KEY"

        [flash_model]
        endpoint = "relay"
        model_id = "relay-flash"

        [evaluator_model]
        endpoint = "relay"
        model_id = "relay-eval"
        max_output_tokens = 64
        """,
    )
    settings = resolve_settings(path, max_papers=40)
    assert settings.max_papers == 40
    assert settings.endpoints["relay"].base_url == "https://relay.test/v1"
    assert settings.flash_model.model_id == "relay-flash"
    assert settings.evaluator_model.max_output_tokens == 64


def test_unknown_endpoint_field_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        api_key_env = "RELAY_KEY"
        timeout = 30
        """,
    )
    with pytest.raises(ConfigError, match="unknown field `timeout`"):
        resolve_settings(path)


def test_unknown_model_field_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [flash_model]
        endpoint = "opencode"
        model_id = "fast"
        temperature = 0.2
        """,
    )
    with pytest.raises(ConfigError, match="unknown field `temperature`"):
        resolve_settings(path)


@pytest.mark.parametrize(
    "text",
    [
        'api = "compatible"',
        'api = "openai"\nauth = "basic"',
    ],
)
def test_invalid_api_or_auth_value_is_rejected(tmp_path: Path, text: str) -> None:
    path = _write_config(
        tmp_path,
        f"""
        [endpoints.relay]
        {text}
        base_url = "https://relay.test/v1"
        api_key_env = "RELAY_KEY"
        """,
    )
    with pytest.raises(ConfigError, match="Invalid enum value"):
        resolve_settings(path)


@pytest.mark.parametrize(
    "base_url", ["http://127.0.0.1:8080/v1", "https://relay.test:bad/v1"]
)
def test_config_keeps_endpoint_url_as_supplied(tmp_path: Path, base_url: str) -> None:
    path = _write_config(
        tmp_path,
        f"""
        [endpoints.opencode]
        api = "openai"
        base_url = "{base_url}"
        api_key_env = "RELAY_KEY"
        """,
    )
    assert resolve_settings(path).endpoints["opencode"].base_url == base_url


@pytest.mark.parametrize("api_key_env", ["", "sk-live-secret", "KEY WITH SPACE"])
def test_endpoint_requires_an_environment_variable_name(
    tmp_path: Path, api_key_env: str
) -> None:
    path = _write_config(
        tmp_path,
        f"""
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        api_key_env = "{api_key_env}"
        """,
    )
    with pytest.raises(ConfigError, match="environment") as caught:
        resolve_settings(path)
    assert "sk-live-secret" not in str(caught.value)


def test_endpoint_may_use_only_a_named_credential(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        credential = "relay"

        [flash_model]
        endpoint = "relay"
        model_id = "relay-flash"

        [evaluator_model]
        endpoint = "relay"
        model_id = "relay-eval"
        """,
    )
    endpoint = resolve_settings(path).endpoints["relay"]
    assert endpoint.credential == "relay"
    assert endpoint.api_key_env is None


def test_endpoint_may_use_only_an_environment_variable() -> None:
    endpoint = EndpointSettings(
        api="openai", base_url="https://relay.test/v1", api_key_env="RELAY_KEY"
    )
    assert endpoint.credential is None


def test_endpoint_requires_a_credential_source(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        """,
    )
    with pytest.raises(ConfigError, match="credential"):
        resolve_settings(path)


@pytest.mark.parametrize("credential", ["", "   "])
def test_credential_name_cannot_be_blank(credential: str) -> None:
    with pytest.raises(ConfigError, match="credential must not be blank"):
        EndpointSettings(
            api="openai", base_url="https://relay.test/v1", credential=credential
        )


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        ({"Authorization": "Bearer secret"}, "managed by the client"),
        ({"content-type": "text/plain"}, "managed by the client"),
        ({"content-length": "0"}, "managed by the client"),
        ({"transfer-encoding": "chunked"}, "managed by the client"),
        ({"Host": "other.test"}, "managed by the client"),
        ({"X-Test": "1", "x-test": "2"}, "repeats another header"),
    ],
)
def test_invalid_endpoint_headers_are_rejected(
    tmp_path: Path, headers: dict[str, str], detail: str
) -> None:
    lines = "\n".join(f'"{name}" = "{value}"' for name, value in headers.items())
    path = _write_config(
        tmp_path,
        f"""
        [endpoints.relay]
        api = "openai"
        base_url = "https://relay.test/v1"
        api_key_env = "RELAY_KEY"

        [endpoints.relay.headers]
        {lines}
        """,
    )
    with pytest.raises(ConfigError, match=detail):
        resolve_settings(path)


@pytest.mark.parametrize("quota_scope", ["", "   "])
def test_quota_scope_cannot_be_blank(quota_scope: str) -> None:
    with pytest.raises(ConfigError, match="quota_scope must not be blank"):
        EndpointSettings(
            api="openai",
            base_url="https://relay.test/v1",
            api_key_env="RELAY_KEY",
            quota_scope=quota_scope,
        )


def test_anthropic_role_requires_an_output_limit(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.claude]
        api = "anthropic"
        base_url = "https://api.anthropic.com/v1"
        api_key_env = "ANTHROPIC_API_KEY"

        [flash_model]
        endpoint = "claude"
        model_id = "claude-flash"

        [evaluator_model]
        endpoint = "claude"
        model_id = "claude-eval"
        max_output_tokens = 4096
        """,
    )
    with pytest.raises(ConfigError, match="flash_model requires max_output_tokens"):
        resolve_settings(path)


def test_non_positive_output_limit_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [flash_model]
        endpoint = "opencode"
        model_id = "fast"
        max_output_tokens = 0
        """,
    )
    with pytest.raises(ConfigError, match="Expected `int` >= 1"):
        resolve_settings(path)


@pytest.mark.parametrize(
    "model",
    [
        '[flash_model]\nendpoint = "opencode"\nmodel_id = ""',
        '[flash_model]\nendpoint = ""\nmodel_id = "fast"',
    ],
)
def test_empty_model_reference_is_rejected(tmp_path: Path, model: str) -> None:
    with pytest.raises(ConfigError, match="must not be empty"):
        resolve_settings(_write_config(tmp_path, model))


def test_gemini_model_id_must_name_a_model(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        [endpoints.gemini]
        api = "gemini"
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        api_key_env = "GEMINI_API_KEY"

        [flash_model]
        endpoint = "gemini"
        model_id = "models/"

        [evaluator_model]
        endpoint = "gemini"
        model_id = "models/gemini-4-argon"
        """,
    )
    with pytest.raises(ConfigError, match="must name a Gemini model"):
        resolve_settings(path)
