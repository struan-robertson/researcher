import hashlib
from pathlib import Path

import pytest

from researcher.question import QuestionError, load_question


def write_brief(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def test_question_id_hashes_exact_bytes(tmp_path: Path) -> None:
    content = b"# Question\n\nWhich methods handle sparse rewards?\n"
    path = write_brief(tmp_path / "brief.md", content)
    question = load_question(path)
    assert question.question_id == hashlib.sha256(content).hexdigest()
    assert question.text == content.decode("utf-8")
    assert question.source == path


def test_trailing_newline_changes_id(tmp_path: Path) -> None:
    bare = load_question(write_brief(tmp_path / "bare.md", b"Question"))
    newline = load_question(write_brief(tmp_path / "newline.md", b"Question\n"))
    assert bare.question_id != newline.question_id
    assert bare.text == "Question"
    assert newline.text == "Question\n"


def test_line_ending_changes_id(tmp_path: Path) -> None:
    lf = load_question(write_brief(tmp_path / "lf.md", b"a\nb\n"))
    crlf = load_question(write_brief(tmp_path / "crlf.md", b"a\r\nb\r\n"))
    assert lf.question_id != crlf.question_id
    assert crlf.text == "a\r\nb\r\n"


def test_changed_content_changes_id(tmp_path: Path) -> None:
    path = write_brief(tmp_path / "brief.md", b"First question\n")
    first = load_question(path)
    path.write_bytes(b"Second question\n")
    second = load_question(path)
    assert first.question_id != second.question_id


def test_moved_file_keeps_id(tmp_path: Path) -> None:
    source = write_brief(tmp_path / "brief.md", b"Same content\n")
    before = load_question(source)
    moved = source.rename(tmp_path / "renamed.md")
    assert load_question(moved).question_id == before.question_id


def test_directory_input_uses_question_md(tmp_path: Path) -> None:
    content = b"# Brief\n\nContent.\n"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    brief = write_brief(output_dir / "question.md", content)
    from_directory = load_question(output_dir)
    from_file = load_question(brief)
    assert from_directory.question_id == from_file.question_id
    assert from_directory.source == brief
    assert from_directory.text == content.decode("utf-8")


def test_missing_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QuestionError, match="question file not found"):
        load_question(tmp_path / "absent.md")


def test_directory_without_question_is_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    with pytest.raises(QuestionError, match="no question.md in directory"):
        load_question(output_dir)


@pytest.mark.parametrize("content", [b"", b"  \n\t\n"])
def test_empty_brief_is_rejected(tmp_path: Path, content: bytes) -> None:
    with pytest.raises(QuestionError, match="question file is empty"):
        load_question(write_brief(tmp_path / "brief.md", content))


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = write_brief(tmp_path / "brief.md", b"Question \xff\n")
    with pytest.raises(QuestionError, match="not valid UTF-8"):
        load_question(path)


def test_loading_does_not_modify_source(tmp_path: Path) -> None:
    content = b"# Question\n\nUnchanged.\n"
    path = write_brief(tmp_path / "brief.md", content)
    before = path.stat().st_mtime_ns
    load_question(path)
    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == before
