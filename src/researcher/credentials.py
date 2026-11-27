"""Named API keys read from a credentials file separate from configuration."""

import os
import stat
from pathlib import Path

import msgspec

from researcher.errors import UserError


class CredentialError(UserError):
    """A credentials file is missing, unsafe or unreadable."""


def load_credentials(path: Path) -> dict[str, str]:
    """Read the named keys from a flat TOML table of strings.

    Require a regular file and check POSIX ownership and permissions on the
    same descriptor used to read it.
    """
    try:
        with path.open("rb") as handle:
            _check_private_file(path, os.fstat(handle.fileno()))
            data = handle.read()
    except FileNotFoundError:
        raise CredentialError(f"credentials file not found: {path}") from None
    except OSError:
        raise CredentialError(f"cannot read credentials file: {path}") from None
    try:
        return msgspec.toml.decode(data.decode("utf-8"), type=dict[str, str])
    except UnicodeDecodeError, msgspec.DecodeError:
        raise CredentialError(
            f"invalid credentials file {path}: expected a flat table of named "
            'string values, such as opencode = "key"'
        ) from None


def _check_private_file(path: Path, info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise CredentialError(f"credentials file is not a regular file: {path}")
    if os.name != "posix":
        return
    if info.st_uid != os.geteuid():
        raise CredentialError(
            f"credentials file {path} is not owned by the current user"
        )
    if info.st_mode & 0o077:
        raise CredentialError(
            f"credentials file {path} is accessible to other users: restrict it "
            "to owner-only permissions, such as chmod 600"
        )
