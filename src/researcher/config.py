"""Effective research settings from built-in defaults, TOML and CLI overrides."""

import math
import os
import re
from pathlib import Path
from typing import Annotated

import msgspec

from researcher.errors import UserError

DEFAULT_MAX_PAPERS = 25
DEFAULT_TIME_LIMIT = "1h"

_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0}
_DURATION = re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[smh])")


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


class Settings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Research settings as decoded from TOML and overridden on the command line."""

    max_papers: Annotated[int, msgspec.Meta(gt=0)] = DEFAULT_MAX_PAPERS
    time_limit: str = DEFAULT_TIME_LIMIT

    def __post_init__(self) -> None:
        parse_duration(self.time_limit)

    @property
    def time_limit_seconds(self) -> float:
        return parse_duration(self.time_limit)


def default_config_path() -> Path | None:
    base = os.environ.get("XDG_CONFIG_HOME")
    return Path(base) / "researcher" / "config.toml" if base else None


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
