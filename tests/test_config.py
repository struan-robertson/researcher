from pathlib import Path

import pytest

from researcher.config import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_TIME_LIMIT,
    ConfigError,
    Settings,
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


def test_unreadable_config_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read configuration file"):
        resolve_settings(tmp_path)
