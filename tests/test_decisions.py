import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from researcher.library import (
    ArtefactRole,
    AssessmentDecision,
    DecisionCounts,
    EnrichmentLevel,
    Identifier,
    Library,
    ScreeningDecision,
)
from researcher.question import Question, load_question

BRIEF = (
    "# Question\n"
    "\n"
    "Which methods reduce naïve sparsity?\n"
    "\n"
    "  - primary criterion: held-out accuracy\n"
    "  - secondary criterion:\n"
    "      nested under an indented item\n"
    "\n"
    "Meaningful trailing spaces.  \n"
    "\n"
).encode()


@pytest.fixture
def library(tmp_path: Path) -> Iterator[Library]:
    with Library(tmp_path / "library") as library:
        yield library


def write_brief(path: Path, data: bytes = BRIEF) -> Question:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return load_question(path)


def decision_rows(
    connection: sqlite3.Connection, table: str, question_id: str, paper_id: str
) -> int:
    row = connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE question_id = ? AND paper_id = ?",
        (question_id, paper_id),
    ).fetchone()
    return row[0]


def test_identical_brief_bytes_share_identity_and_decisions(
    library: Library, tmp_path: Path
) -> None:
    first = write_brief(tmp_path / "first.md")
    second = write_brief(tmp_path / "results" / "question.md")
    assert first.source != second.source
    assert first.question_id == second.question_id
    library.record_question(first)
    library.record_question(second)
    assert library.question_text(first.question_id) == BRIEF.decode("utf-8")

    paper_id = library.add_paper(title="Sparse coding")
    library.record_screening(
        first.question_id,
        paper_id,
        True,
        model_id="flash",
        prompt_version="screening-1",
    )
    library.record_assessment(
        first.question_id,
        paper_id,
        True,
        model_id="evaluator",
        prompt_version="assessment-1",
        explanation="Directly measures the criterion.",
    )
    assert library.get_screening(first.question_id, paper_id) == ScreeningDecision(
        question_id=first.question_id,
        paper_id=paper_id,
        passed=True,
        model_id="flash",
        prompt_version="screening-1",
    )
    assert library.get_assessment(first.question_id, paper_id) == AssessmentDecision(
        question_id=first.question_id,
        paper_id=paper_id,
        surface=True,
        model_id="evaluator",
        prompt_version="assessment-1",
        explanation="Directly measures the criterion.",
        pdf_filename=None,
    )
    assert library.get_screening(second.question_id, paper_id) == library.get_screening(
        first.question_id, paper_id
    )


def test_changed_brief_reverses_decisions_without_losing_paper_state(
    library: Library, tmp_path: Path
) -> None:
    paper_id = library.add_paper(
        title="Retained work", abstract="An abstract.", authors=["Ada Lovelace"]
    )
    library.add_identifier(paper_id, "doi", "10.1000/retained")
    library.record_source(
        paper_id, "openalex", {"id": "W1"}, pdf_locations=["https://example/1.pdf"]
    )
    artefact_id = library.add_artefact_bytes(
        paper_id, b"%PDF-1.7 retained", role=ArtefactRole.PDF
    )
    library.promote(paper_id, EnrichmentLevel.READING_READY)
    original = write_brief(tmp_path / "original.md")
    changed = write_brief(tmp_path / "changed.md", BRIEF + b"Only graphical models.\n")
    assert original.question_id != changed.question_id
    library.record_question(original)
    library.record_question(changed)
    assert library.get_screening(original.question_id, paper_id) is None
    assert library.get_assessment(original.question_id, paper_id) is None
    assert library.get_assessment(changed.question_id, paper_id) is None

    library.record_screening(
        original.question_id,
        paper_id,
        True,
        model_id="flash",
        prompt_version="screening-1",
    )
    library.record_assessment(
        original.question_id,
        paper_id,
        True,
        model_id="evaluator",
        prompt_version="assessment-1",
        explanation="Within the original scope.",
    )
    library.record_screening(
        changed.question_id,
        paper_id,
        False,
        model_id="flash",
        prompt_version="screening-1",
    )
    library.record_assessment(
        changed.question_id,
        paper_id,
        False,
        model_id="evaluator",
        prompt_version="assessment-1",
    )

    approved = library.get_assessment(original.question_id, paper_id)
    rejected = library.get_assessment(changed.question_id, paper_id)
    assert approved is not None and approved.surface is True
    assert rejected is not None and rejected.surface is False
    assert (rejected.explanation, rejected.pdf_filename) == (None, None)
    assert library.get_screening(original.question_id, paper_id).passed is True
    assert library.get_screening(changed.question_id, paper_id).passed is False

    paper = library.get_paper(paper_id)
    assert paper is not None
    assert paper.title == "Retained work"
    assert paper.enrichment_level == EnrichmentLevel.READING_READY
    assert library.identifiers(paper_id) == (Identifier("doi", "10.1000/retained"),)
    assert library.paper_sources(paper_id)[0].pdf_locations == (
        "https://example/1.pdf",
    )
    assert library.paper_artefacts(paper_id)[0].artefact_id == artefact_id
    assert library.artefact_path(artefact_id).read_bytes() == b"%PDF-1.7 retained"


def test_decisions_and_provenance_survive_reopen(tmp_path: Path) -> None:
    root = tmp_path / "library"
    brief = write_brief(tmp_path / "question.md")
    other = write_brief(tmp_path / "other.md", b"# Another question\n")
    with Library(root) as library:
        library.record_question(brief)
        library.record_question(other)
        rejected = library.add_paper(title="Rejected")
        approved = library.add_paper(title="Approved")
        library.record_screening(
            brief.question_id,
            rejected,
            False,
            model_id="flash-a",
            prompt_version="screening-1",
        )
        library.record_screening(
            brief.question_id,
            approved,
            True,
            model_id="flash-a",
            prompt_version="screening-1",
        )
        library.record_assessment(
            brief.question_id,
            rejected,
            False,
            model_id="evaluator-a",
            prompt_version="assessment-1",
        )
        library.record_assessment(
            brief.question_id,
            approved,
            True,
            model_id="evaluator-a",
            prompt_version="assessment-1",
            explanation="Useful for this question.",
        )
        library.record_assessment(
            other.question_id,
            approved,
            True,
            model_id="evaluator-b",
            prompt_version="assessment-2",
            explanation="Useful for another question.",
            pdf_filename="useful-paper.pdf",
        )
        # Provenance is not part of the cache key, so later versions are ignored.
        library.record_screening(
            brief.question_id,
            rejected,
            True,
            model_id="flash-b",
            prompt_version="screening-2",
        )
        library.record_assessment(
            brief.question_id,
            approved,
            False,
            model_id="evaluator-b",
            prompt_version="assessment-2",
        )
    with Library(root) as reopened:
        assert reopened.question_text(brief.question_id) == BRIEF.decode("utf-8")
        assert reopened.question_text(other.question_id) == "# Another question\n"
        assert reopened.question_text("missing") is None
        assert reopened.get_screening(brief.question_id, rejected) == ScreeningDecision(
            question_id=brief.question_id,
            paper_id=rejected,
            passed=False,
            model_id="flash-a",
            prompt_version="screening-1",
        )
        assert reopened.get_assessment(
            brief.question_id, rejected
        ) == AssessmentDecision(
            question_id=brief.question_id,
            paper_id=rejected,
            surface=False,
            model_id="evaluator-a",
            prompt_version="assessment-1",
            explanation=None,
            pdf_filename=None,
        )
        assert reopened.get_assessment(
            brief.question_id, approved
        ) == AssessmentDecision(
            question_id=brief.question_id,
            paper_id=approved,
            surface=True,
            model_id="evaluator-a",
            prompt_version="assessment-1",
            explanation="Useful for this question.",
            pdf_filename=None,
        )
        optional_filename = reopened.get_assessment(other.question_id, approved)
        assert optional_filename is not None
        assert optional_filename.pdf_filename == "useful-paper.pdf"
        # An assessment does not depend on a screening record for the paper.
        assert reopened.get_screening(other.question_id, approved) is None
        assert reopened.decision_counts(approved) == DecisionCounts(
            screened=1,
            passed_screening=1,
            assessed=2,
            approved_assessment=2,
        )


def test_repeat_writes_keep_the_first_results(library: Library, tmp_path: Path) -> None:
    question = write_brief(tmp_path / "question.md")
    library.record_question(question)
    paper_id = library.add_paper(title="Cached")
    library.record_screening(
        question.question_id,
        paper_id,
        False,
        model_id="flash",
        prompt_version="screening-1",
    )
    library.record_assessment(
        question.question_id,
        paper_id,
        False,
        model_id="evaluator",
        prompt_version="assessment-1",
    )
    library.record_screening(
        question.question_id,
        paper_id,
        True,
        model_id="flash",
        prompt_version="screening-2",
    )
    library.record_assessment(
        question.question_id,
        paper_id,
        True,
        model_id="evaluator",
        prompt_version="assessment-2",
        explanation="A second attempt.",
        pdf_filename="second-attempt.pdf",
    )
    screening = library.get_screening(question.question_id, paper_id)
    assert screening is not None
    assert (screening.passed, screening.prompt_version) == (False, "screening-1")
    assert library.get_assessment(question.question_id, paper_id) == AssessmentDecision(
        question_id=question.question_id,
        paper_id=paper_id,
        surface=False,
        model_id="evaluator",
        prompt_version="assessment-1",
        explanation=None,
        pdf_filename=None,
    )
    assert library.decision_counts(paper_id) == DecisionCounts(
        screened=1,
        passed_screening=0,
        assessed=1,
        approved_assessment=0,
    )


def test_counts_cover_negatives_and_several_questions_per_paper(
    library: Library, tmp_path: Path
) -> None:
    questions = (
        write_brief(tmp_path / "first.md"),
        write_brief(tmp_path / "second.md", b"# Second question\n"),
        write_brief(tmp_path / "third.md", b"# Third question\n"),
    )
    for question in questions:
        library.record_question(question)
    kept = library.add_paper(title="Kept")
    rejected = library.add_paper(title="Rejected")
    untouched = library.add_paper(title="Untouched")
    first, second, third = (question.question_id for question in questions)

    library.record_screening(first, kept, True, model_id="m", prompt_version="v")
    library.record_assessment(
        first,
        kept,
        True,
        model_id="m",
        prompt_version="v",
        explanation="Surfaced once.",
    )
    library.record_screening(first, rejected, False, model_id="m", prompt_version="v")
    library.record_assessment(first, rejected, False, model_id="m", prompt_version="v")
    library.record_screening(second, kept, True, model_id="m", prompt_version="v")
    library.record_assessment(second, kept, False, model_id="m", prompt_version="v")
    library.record_screening(second, rejected, True, model_id="m", prompt_version="v")
    library.record_assessment(
        second,
        rejected,
        True,
        model_id="m",
        prompt_version="v",
        explanation="Surfaced twice.",
        pdf_filename="rejected-paper.pdf",
    )
    library.record_screening(third, rejected, True, model_id="m", prompt_version="v")

    assert library.decision_counts(kept) == DecisionCounts(
        screened=2,
        passed_screening=2,
        assessed=2,
        approved_assessment=1,
    )
    assert library.decision_counts(rejected) == DecisionCounts(
        screened=3,
        passed_screening=2,
        assessed=2,
        approved_assessment=1,
    )
    assert library.decision_counts(untouched) == DecisionCounts(
        screened=0,
        passed_screening=0,
        assessed=0,
        approved_assessment=0,
    )
    assert library.decision_counts("unknown") == DecisionCounts(
        screened=0,
        passed_screening=0,
        assessed=0,
        approved_assessment=0,
    )


def test_decision_tables_index_papers_for_counts(library: Library) -> None:
    connection = sqlite3.connect(library.database_path)
    indexed: dict[str, list[str]] = {}
    try:
        for table in ("screening_decisions", "assessment_decisions"):
            index = f"{table}_paper"
            names = {
                row[1] for row in connection.execute(f"PRAGMA index_list('{table}')")
            }
            assert index in names
            indexed[index] = [
                row[2] for row in connection.execute(f"PRAGMA index_info('{index}')")
            ]
    finally:
        connection.close()
    assert indexed == {
        "screening_decisions_paper": ["paper_id"],
        "assessment_decisions_paper": ["paper_id"],
    }


def test_decision_constraints_apply_to_direct_database_writes(
    library: Library, tmp_path: Path
) -> None:
    question = write_brief(tmp_path / "question.md")
    library.record_question(question)
    paper_id = library.add_paper(title="Direct")
    connection = sqlite3.connect(library.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    screening = (
        "INSERT INTO screening_decisions (question_id, paper_id, passed,"
        " model_id, prompt_version) VALUES (?, ?, ?, 'm', 'v')"
    )
    assessment = (
        "INSERT INTO assessment_decisions (question_id, paper_id, surface,"
        " model_id, prompt_version, explanation, pdf_filename)"
        " VALUES (?, ?, ?, 'm', 'v', ?, ?)"
    )
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(screening, ("missing", paper_id, 1))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(screening, (question.question_id, "missing", 1))
        for passed in (2, -1, None):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(screening, (question.question_id, paper_id, passed))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO screening_decisions (question_id, paper_id, passed,"
                " model_id, prompt_version) VALUES (?, ?, 1, NULL, 'v')",
                (question.question_id, paper_id),
            )
        connection.execute(screening, (question.question_id, paper_id, 1))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(screening, (question.question_id, paper_id, 0))

        bad_assessments = (
            (1, None, None),
            (1, None, "suggested.pdf"),
            (1, "", None),
            (1, "  \n\t ", None),
            (0, "Rejected with an explanation.", None),
            (0, None, "rejected.pdf"),
            (0, "", ""),
            (2, "Out of range.", None),
            (None, "Missing outcome.", None),
        )
        for surface, explanation, pdf_filename in bad_assessments:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    assessment,
                    (
                        question.question_id,
                        paper_id,
                        surface,
                        explanation,
                        pdf_filename,
                    ),
                )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(assessment, ("missing", paper_id, 1, "Why.", None))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                assessment,
                (question.question_id, "missing", 1, "Why.", None),
            )
        connection.execute(assessment, (question.question_id, paper_id, 0, None, None))
        assert (
            decision_rows(
                connection, "assessment_decisions", question.question_id, paper_id
            )
            == 1
        )
        connection.execute(
            "DELETE FROM assessment_decisions WHERE question_id = ? AND paper_id = ?",
            (question.question_id, paper_id),
        )
        connection.execute(
            assessment, (question.question_id, paper_id, 1, "Justified.", None)
        )
        connection.execute(
            "DELETE FROM assessment_decisions WHERE question_id = ? AND paper_id = ?",
            (question.question_id, paper_id),
        )
        connection.execute(
            assessment,
            (question.question_id, paper_id, 1, "Justified.", "suggested.pdf"),
        )
    finally:
        connection.rollback()
        connection.close()
    assert library.get_screening(question.question_id, paper_id) is None
    assert library.get_assessment(question.question_id, paper_id) is None


def test_deleting_question_or_paper_cascades_decisions(
    library: Library, tmp_path: Path
) -> None:
    question = write_brief(tmp_path / "question.md")
    library.record_question(question)
    paper_id = library.add_paper(title="Cascade")
    library.record_screening(
        question.question_id, paper_id, True, model_id="m", prompt_version="v"
    )
    library.record_assessment(
        question.question_id,
        paper_id,
        True,
        model_id="m",
        prompt_version="v",
        explanation="Surfaced.",
    )
    with sqlite3.connect(library.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
        assert (
            decision_rows(
                connection, "screening_decisions", question.question_id, paper_id
            )
            == 0
        )
        assert (
            decision_rows(
                connection, "assessment_decisions", question.question_id, paper_id
            )
            == 0
        )

    other_paper = library.add_paper(title="Cascade again")
    library.record_screening(
        question.question_id, other_paper, True, model_id="m", prompt_version="v"
    )
    library.record_assessment(
        question.question_id,
        other_paper,
        True,
        model_id="m",
        prompt_version="v",
        explanation="Surfaced.",
    )
    with sqlite3.connect(library.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "DELETE FROM questions WHERE question_id = ?", (question.question_id,)
        )
        assert (
            decision_rows(
                connection, "screening_decisions", question.question_id, other_paper
            )
            == 0
        )
        assert (
            decision_rows(
                connection, "assessment_decisions", question.question_id, other_paper
            )
            == 0
        )
    assert library.question_text(question.question_id) is None
