import os
from pathlib import Path

import pytest

from researcher.credentials import CredentialError, load_credentials

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file ownership")


def _write(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def test_loads_the_named_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "credentials.toml", 'opencode = "go-key"\nrelay = "r-key"\n'
    )
    assert load_credentials(path) == {"opencode": "go-key", "relay": "r-key"}


def test_empty_file_has_no_credentials(tmp_path: Path) -> None:
    assert load_credentials(_write(tmp_path / "credentials.toml", "")) == {}


def test_missing_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "credentials.toml"
    with pytest.raises(CredentialError, match="credentials file not found"):
        load_credentials(path)


def test_directory_is_not_a_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "credentials.toml"
    path.mkdir()
    with pytest.raises(CredentialError, match="not a regular file|cannot read"):
        load_credentials(path)


def test_descriptor_error_is_reported_without_raw_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "credentials.toml", 'opencode = "go-key"\n')

    def fail(descriptor: int) -> os.stat_result:
        raise OSError("private details")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fstat", fail)
        with pytest.raises(CredentialError, match="cannot read") as caught:
            load_credentials(path)
    assert "private details" not in str(caught.value)
    assert caught.value.__suppress_context__


@posix_only
@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660, 0o644])
def test_group_or_other_permissions_are_rejected(tmp_path: Path, mode: int) -> None:
    path = _write(tmp_path / "credentials.toml", 'opencode = "go-key"\n', mode)
    with pytest.raises(CredentialError, match="accessible to other users"):
        load_credentials(path)


@posix_only
@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_owner_only_permissions_are_accepted(tmp_path: Path, mode: int) -> None:
    path = _write(tmp_path / "credentials.toml", 'opencode = "go-key"\n', mode)
    assert load_credentials(path) == {"opencode": "go-key"}


@posix_only
def test_file_must_be_owned_by_the_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "credentials.toml", 'opencode = "go-key"\n')
    monkeypatch.setattr(os, "geteuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(CredentialError, match="not owned by the current user"):
        load_credentials(path)


@pytest.mark.parametrize(
    "content",
    [
        '[opencode]\nkey = "sk-secret"\n',
        'opencode = "sk-secret"\nrelay = 7\n',
        'opencode = "sk-secret"\nrelay = ',
        b'opencode = "sk-secret"\n\xff\n',
    ],
)
def test_invalid_file_is_reported_without_its_contents(
    tmp_path: Path, content: str | bytes
) -> None:
    path = tmp_path / "credentials.toml"
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    path.chmod(0o600)
    with pytest.raises(CredentialError, match="invalid credentials file") as caught:
        load_credentials(path)
    assert str(path) in str(caught.value)
    assert "sk-secret" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('opencode = "go-key"\nrelay = ', "invalid credentials file"),
        ("opencode = 7\n", "invalid credentials file"),
    ],
)
def test_parser_errors_are_not_chained(
    tmp_path: Path, content: str, match: str
) -> None:
    path = _write(tmp_path / "credentials.toml", content)
    with pytest.raises(CredentialError, match=match) as caught:
        load_credentials(path)
    assert caught.value.__cause__ is None
