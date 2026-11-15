import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from researcher.cli import main


def test_no_arguments_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    output = capsys.readouterr()
    assert "usage: researcher" in output.out
    assert "not implemented yet" in output.out
    assert output.err == ""


def test_help_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: researcher" in capsys.readouterr().out


def test_version_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"researcher {version('researcher')}\n"


def test_unknown_option_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--unknown"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("entrypoint", ["console", "module"])
def test_installed_entrypoints(entrypoint: str, tmp_path: Path) -> None:
    if entrypoint == "console":
        command = [str(Path(sys.executable).with_name("researcher")), "--version"]
    else:
        command = [sys.executable, "-m", "researcher", "--version"]
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=True, timeout=10
    )
    assert result.stdout == f"researcher {version('researcher')}\n"
    assert result.stderr == ""
