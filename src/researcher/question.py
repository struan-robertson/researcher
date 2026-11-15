"""Load a research brief from a Markdown file or a prior output directory."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from researcher.errors import UserError

QUESTION_FILENAME = "question.md"


class QuestionError(UserError):
    """The supplied research brief is missing or unreadable."""


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    source: Path


def load_question(target: Path) -> Question:
    """Read the brief named by a Markdown file or an output directory.

    The identity is the SHA-256 digest of the file bytes, so paths and
    configuration overrides cannot change it.
    """
    if target.is_dir():
        path = target / QUESTION_FILENAME
        if not path.is_file():
            raise QuestionError(f"no {QUESTION_FILENAME} in directory: {target}")
    elif target.is_file():
        path = target
    else:
        raise QuestionError(f"question file not found: {target}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise QuestionError(
            f"cannot read question file {path}: {exc.strerror or exc}"
        ) from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QuestionError(f"question file is not valid UTF-8: {path}") from exc
    if not text.strip():
        raise QuestionError(f"question file is empty: {path}")
    return Question(
        question_id=hashlib.sha256(data).hexdigest(), text=text, source=path
    )
