import hashlib
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from researcher import cli
from researcher.cli import main
from researcher.config import Settings
from researcher.question import Question


@pytest.fixture(autouse=True)
def isolated_config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))


@pytest.fixture
def brief(tmp_path: Path) -> Path:
    path = tmp_path / "brief.md"
    path.write_bytes(b"# Question\n\nWhich methods handle sparse rewards?\n")
    return path


def brief_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_arguments_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    output = capsys.readouterr()
    assert "usage: researcher" in output.out
    assert "not implemented yet" in output.out
    assert output.err == ""


def test_options_without_question_are_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invocations = [
        ["--config", str(tmp_path / "absent.toml")],
        ["--max-papers", "0"],
    ]
    for arguments in invocations:
        with pytest.raises(SystemExit) as exc:
            main(arguments)
        assert exc.value.code == 2
        output = capsys.readouterr()
        assert "the following arguments are required: QUESTION" in output.err
        assert "configuration file not found" not in output.err
        assert output.out == ""


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
        main(["brief.md", "--unknown"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_brief_file_reports_preparation_summary(
    brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(brief)]) == 0
    output = capsys.readouterr()
    assert output.out == (
        f"question ID: {brief_id(brief)}\n"
        "max papers: 25\n"
        "time limit: 1h\n"
        "The research workflow is not implemented yet.\n"
    )
    assert output.err == ""


def test_directory_input_reports_same_id(
    brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = brief.parent / "run"
    output_dir.mkdir()
    (output_dir / "question.md").write_bytes(brief.read_bytes())
    assert main([str(output_dir)]) == 0
    assert f"question ID: {brief_id(brief)}" in capsys.readouterr().out


def test_cli_overrides_are_effective_and_leave_id_unchanged(
    brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(brief), "--max-papers", "4", "--time-limit", "0.5h"]) == 0
    output = capsys.readouterr()
    assert f"question ID: {brief_id(brief)}" in output.out
    assert "max papers: 4" in output.out
    assert "time limit: 0.5h" in output.out


def test_config_file_settings_are_applied(
    tmp_path: Path, brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('max_papers = 3\ntime_limit = "10m"\n', encoding="utf-8")
    assert main(["--config", str(config), str(brief)]) == 0
    output = capsys.readouterr()
    assert "max papers: 3" in output.out
    assert "time limit: 10m" in output.out


def test_default_config_file_is_applied(
    tmp_path: Path,
    brief: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_home = tmp_path / "xdg"
    config = config_home / "researcher" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("max_papers = 9\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    assert main([str(brief)]) == 0
    assert "max papers: 9" in capsys.readouterr().out


def test_cli_overrides_config_file(
    tmp_path: Path, brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("max_papers = 3\n", encoding="utf-8")
    assert main(["--config", str(config), str(brief), "--max-papers", "8"]) == 0
    assert "max papers: 8" in capsys.readouterr().out


@pytest.mark.parametrize("options", [["--help"], ["--version"]])
def test_help_and_version_do_not_load_config(
    tmp_path: Path, options: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.toml"
    with pytest.raises(SystemExit) as exc:
        main([*options, "--config", str(missing)])
    assert exc.value.code == 0
    assert capsys.readouterr().err == ""


def test_missing_explicit_config_is_reported(
    tmp_path: Path, brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.toml"
    assert main(["--config", str(missing), str(brief)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "researcher: error:" in output.err
    assert "configuration file not found" in output.err
    assert "Traceback" not in output.err


def test_malformed_config_is_reported(
    tmp_path: Path, brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("max_papers = \n", encoding="utf-8")
    assert main(["--config", str(config), str(brief)]) == 2
    assert "invalid configuration file" in capsys.readouterr().err


def test_invalid_config_setting_is_reported(
    tmp_path: Path, brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("max_papers = 0\n", encoding="utf-8")
    assert main(["--config", str(config), str(brief)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert str(config) in output.err
    assert "$.max_papers" in output.err
    assert "Traceback" not in output.err


def test_missing_question_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "absent.md")]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "question file not found" in output.err


def test_directory_without_question_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    assert main([str(output_dir)]) == 2
    assert "no question.md in directory" in capsys.readouterr().err


def test_empty_question_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.md"
    path.write_bytes(b" \n")
    assert main([str(path)]) == 2
    assert "question file is empty" in capsys.readouterr().err


def test_invalid_utf8_question_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "binary.md"
    path.write_bytes(b"\xff\xfe")
    assert main([str(path)]) == 2
    assert "not valid UTF-8" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-2"])
def test_non_positive_max_papers_is_reported(
    brief: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(brief), "--max-papers", value]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "invalid command-line settings" in output.err
    assert "$.max_papers" in output.err
    assert "Traceback" not in output.err


def test_non_numeric_max_papers_option_is_rejected(
    brief: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main([str(brief), "--max-papers", "many"])
    assert exc.value.code == 2
    assert "--max-papers" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0h", "-1h", "1d", "soon"])
def test_invalid_time_limit_is_reported(
    brief: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(brief), "--time-limit", value]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "invalid command-line settings" in output.err
    assert "invalid duration" in output.err
    assert "Traceback" not in output.err


def test_loaded_inputs_are_available_to_the_engine(
    brief: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[tuple[Question, Settings]] = []

    def record(question: Question, settings: Settings) -> int:
        received.append((question, settings))
        return 0

    monkeypatch.setattr(cli, "run", record)
    assert main([str(brief), "--max-papers", "2", "--time-limit", "30s"]) == 0
    [(question, settings)] = received
    assert question.question_id == brief_id(brief)
    assert question.text == brief.read_bytes().decode("utf-8")
    assert settings == Settings(max_papers=2, time_limit="30s")
    assert settings.time_limit_seconds == 30.0


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


@pytest.mark.parametrize("entrypoint", ["console", "module"])
def test_entrypoints_load_a_brief(entrypoint: str, tmp_path: Path) -> None:
    brief = tmp_path / "brief.md"
    brief.write_text("# Question\n\nContent.\n", encoding="utf-8")
    if entrypoint == "console":
        command = [str(Path(sys.executable).with_name("researcher"))]
    else:
        command = [sys.executable, "-m", "researcher"]
    result = subprocess.run(
        [*command, str(brief)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env={**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "xdg")},
    )
    assert result.stdout.startswith(f"question ID: {brief_id(brief)}\n")
    assert "The research workflow is not implemented yet." in result.stdout
    assert result.stderr == ""
