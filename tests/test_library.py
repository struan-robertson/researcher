import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import pytest

from researcher import artefacts
from researcher.library import (
    DATABASE_FILENAME,
    DEFAULT_MEDIA_TYPE,
    SCHEMA_VERSION,
    ArtefactRole,
    EnrichmentLevel,
    Identifier,
    Library,
    LibraryError,
    Paper,
    PaperSource,
)


@pytest.fixture
def library(tmp_path: Path) -> Iterator[Library]:
    with Library(tmp_path / "library") as library:
        yield library


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_default_library_uses_xdg_data_not_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_home = tmp_path / "data"
    cache_home = tmp_path / "cache"
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.chdir(working)
    expected = data_home / "researcher"
    with Library() as library:
        assert library.root == expected
        assert library.database_path == expected / DATABASE_FILENAME
        paper_id = library.add_paper(title="Stored outside the working directory")
        artefact_id = library.add_artefact_bytes(
            paper_id, b"paper text", role=ArtefactRole.EXTRACTED_TEXT
        )
        assert library.artefact_path(artefact_id).parent == expected / "artefacts"
    with Library() as reopened:
        assert reopened.get_paper(paper_id) is not None
        assert reopened.artefact_path(artefact_id).read_bytes() == b"paper text"
    assert list(working.iterdir()) == []
    assert not cache_home.exists()


@pytest.mark.parametrize("data_home", [None, "", "relative/data"])
def test_default_library_uses_standard_xdg_data_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data_home: str | None
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.setenv("HOME", str(home))
    if data_home is None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_DATA_HOME", data_home)
    monkeypatch.chdir(working)
    with Library() as library:
        assert library.root == home / ".local/share/researcher"
        assert library.database_path.is_file()
    assert list(working.iterdir()) == []


def test_explicit_library_root_overrides_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    explicit = tmp_path / "explicit"
    with Library(explicit) as library:
        assert library.root == explicit
        assert library.database_path == explicit / DATABASE_FILENAME
    assert not data_home.exists()


def test_minimal_paper_record_is_sparse(library: Library) -> None:
    paper_id = library.add_paper()
    assert library.get_paper(paper_id) == Paper(
        paper_id=paper_id,
        title=None,
        abstract=None,
        authors=(),
        published=None,
        enrichment_level=EnrichmentLevel.DISCOVERY,
    )


def test_metadata_round_trip(library: Library) -> None:
    paper_id = library.add_paper(
        title="Sparse rewards",
        abstract="We study sparse rewards.",
        authors=("Ada Lovelace", "Alan Turing"),
        published="2024-05-01",
    )
    paper = library.get_paper(paper_id)
    assert paper is not None
    assert paper.title == "Sparse rewards"
    assert paper.abstract == "We study sparse rewards."
    assert paper.authors == ("Ada Lovelace", "Alan Turing")
    assert paper.published == "2024-05-01"


def test_paper_id_survives_metadata_updates(library: Library) -> None:
    paper_id = library.add_paper(title="Working title")
    library.update_paper(paper_id, title="Published title", authors=["A. Author"])
    paper = library.get_paper(paper_id)
    assert paper is not None
    assert paper.paper_id == paper_id
    assert paper.title == "Published title"
    assert paper.authors == ("A. Author",)


def test_update_preserves_unsupplied_fields(library: Library) -> None:
    paper_id = library.add_paper(
        title="Title", abstract="Abstract", authors=["A"], published="2020"
    )
    library.update_paper(paper_id, abstract="Revised abstract")
    paper = library.get_paper(paper_id)
    assert paper is not None
    assert (paper.title, paper.authors, paper.published) == ("Title", ("A",), "2020")
    assert paper.abstract == "Revised abstract"


def test_update_treats_empty_strings_as_supplied_values(library: Library) -> None:
    paper_id = library.add_paper(
        title="Title", abstract="Abstract", authors=["A"], published="2020"
    )
    library.update_paper(paper_id, title="", abstract="", published="")
    paper = library.get_paper(paper_id)
    assert paper is not None
    assert (paper.title, paper.abstract, paper.published) == ("", "", "")
    assert paper.authors == ("A",)
    library.update_paper(paper_id, abstract="Revised")
    paper = library.get_paper(paper_id)
    assert paper is not None
    assert (paper.title, paper.abstract, paper.published) == ("", "Revised", "")


def test_update_replaces_and_clears_authors(library: Library) -> None:
    paper_id = library.add_paper(authors=["Ada Lovelace", "Alan Turing"])
    library.update_paper(paper_id, authors=["Grace Hopper"])
    assert library.get_paper(paper_id).authors == ("Grace Hopper",)
    library.update_paper(paper_id, title="Renamed")
    assert library.get_paper(paper_id).authors == ("Grace Hopper",)
    library.update_paper(paper_id, authors=[])
    assert library.get_paper(paper_id).authors == ()


def test_author_rows_keep_order_repeats_and_blank_names(library: Library) -> None:
    first = library.add_paper(title="First", authors=["Ada Lovelace", "Ada Lovelace"])
    second = library.add_paper(title="Second", authors=["Ada Lovelace", "Alan Turing"])
    library.update_paper(first, authors=["Alan Turing", "Ada Lovelace", ""])
    assert library.get_paper(first).authors == ("Alan Turing", "Ada Lovelace", "")
    assert library.get_paper(second).authors == ("Ada Lovelace", "Alan Turing")


def test_exact_name_selection_returns_every_matching_paper(library: Library) -> None:
    first = library.add_paper(title="First", authors=["Ada Lovelace", "Alan Turing"])
    second = library.add_paper(title="Second", authors=["Ada Lovelace"])
    unrelated = library.add_paper(title="Third", authors=["Grace Hopper"])
    connection = sqlite3.connect(library.database_path)
    try:
        rows = connection.execute(
            "SELECT paper_id FROM paper_authors WHERE name = ?", ("Ada Lovelace",)
        ).fetchall()
    finally:
        connection.close()
    selected = [row[0] for row in rows]
    assert sorted(selected) == sorted([first, second])
    assert unrelated not in selected


def test_add_paper_rejects_a_bare_string_without_writing(library: Library) -> None:
    with pytest.raises(TypeError, match="authors"):
        library.add_paper(title="Stringly", authors="Ada Lovelace")
    connection = sqlite3.connect(library.database_path)
    try:
        assert connection.execute("SELECT count(*) FROM papers").fetchone()[0] == 0
    finally:
        connection.close()


def test_update_paper_rejects_a_bare_string_without_writing(library: Library) -> None:
    paper_id = library.add_paper(authors=["Ada Lovelace", "Alan Turing"])
    with pytest.raises(TypeError, match="authors"):
        library.update_paper(paper_id, title="Renamed", authors="Grace Hopper")
    paper = library.get_paper(paper_id)
    assert paper is not None
    assert paper.title is None
    assert paper.authors == ("Ada Lovelace", "Alan Turing")


def test_failed_author_insert_leaves_no_paper(library: Library) -> None:
    # The NULL name fails the column constraint after the paper row is written.
    with pytest.raises(sqlite3.IntegrityError):
        library.add_paper(title="Broken", authors=["Ada Lovelace", None])
    connection = sqlite3.connect(library.database_path)
    try:
        assert connection.execute("SELECT count(*) FROM papers").fetchone()[0] == 0
    finally:
        connection.close()


def test_failed_author_update_restores_metadata_and_rows(library: Library) -> None:
    paper_id = library.add_paper(
        title="Original", abstract="Original abstract", authors=["Ada Lovelace"]
    )
    with pytest.raises(sqlite3.IntegrityError):
        library.update_paper(
            paper_id,
            title="Renamed",
            abstract="Renamed abstract",
            authors=["Alan Turing", None],
        )
    paper = library.get_paper(paper_id)
    assert paper is not None
    assert (paper.title, paper.abstract) == ("Original", "Original abstract")
    assert paper.authors == ("Ada Lovelace",)


def test_update_preserves_sources_and_artefacts(library: Library) -> None:
    paper_id = library.add_paper(title="Title")
    library.record_source(paper_id, "openalex", {"id": "W1"})
    artefact_id = library.add_artefact_bytes(
        paper_id, b"supplementary", role=ArtefactRole.SUPPLEMENTARY
    )
    library.update_paper(paper_id, title="Renamed")
    [source] = library.paper_sources(paper_id)
    assert source.metadata == {"id": "W1"}
    [link] = library.paper_artefacts(paper_id)
    assert link.artefact_id == artefact_id


@pytest.mark.parametrize("fields", [{}, {"title": "Title"}])
def test_update_missing_paper_is_reported(
    library: Library, fields: dict[str, str]
) -> None:
    with pytest.raises(LibraryError, match="paper not found"):
        library.update_paper("missing", **fields)


class InterleavingLibrary(Library):
    """A library whose first child-row read can trigger an outside write."""

    interleaved = False

    def update_on_author_read(self, update: Callable[[], None]) -> None:
        def authorizer(
            action: int, table: str | None, _column: str | None, *_args: object
        ) -> int:
            if (
                action == sqlite3.SQLITE_READ
                and table == "paper_authors"
                and not self.interleaved
            ):
                self.interleaved = True
                update()
            return sqlite3.SQLITE_OK

        self._connection.set_authorizer(authorizer)

    def update_on_location_read(self, update: Callable[[], None]) -> None:
        """Trigger an outside source rewrite when the locations are first read."""

        def authorizer(
            action: int, table: str | None, _column: str | None, *_args: object
        ) -> int:
            if (
                action == sqlite3.SQLITE_READ
                and table == "paper_pdf_locations"
                and not self.interleaved
            ):
                self.interleaved = True
                update()
            return sqlite3.SQLITE_OK

        self._connection.set_authorizer(authorizer)


def test_get_paper_reads_metadata_and_authors_coherently(tmp_path: Path) -> None:
    root = tmp_path / "library"
    with Library(root) as writer, InterleavingLibrary(root) as library:
        paper_id = library.add_paper(title="Original", authors=["Ada Lovelace"])
        library.update_on_author_read(
            lambda: writer.update_paper(
                paper_id, title="Renamed", authors=["Alan Turing", "Ada Lovelace"]
            )
        )
        paper = library.get_paper(paper_id)
    assert library.interleaved
    assert paper is not None
    assert (paper.title, paper.authors) == ("Renamed", ("Alan Turing", "Ada Lovelace"))


def test_identifier_lookup_reads_paper_and_authors_coherently(tmp_path: Path) -> None:
    root = tmp_path / "library"
    with Library(root) as writer, InterleavingLibrary(root) as library:
        paper_id = library.add_paper(title="Original", authors=["Ada Lovelace"])
        library.add_identifier(paper_id, "doi", "10.1000/coherent")
        library.update_on_author_read(
            lambda: writer.update_paper(
                paper_id, title="Renamed", authors=["Alan Turing", "Ada Lovelace"]
            )
        )
        paper = library.find_paper_by_identifier("doi", "10.1000/coherent")
    assert library.interleaved
    assert paper is not None
    assert (paper.title, paper.authors) == ("Renamed", ("Alan Turing", "Ada Lovelace"))


def test_paper_sources_reads_metadata_and_locations_coherently(tmp_path: Path) -> None:
    root = tmp_path / "library"
    with Library(root) as writer, InterleavingLibrary(root) as library:
        paper_id = library.add_paper(title="Sourced")
        library.record_source(
            paper_id,
            "arxiv",
            {"version": "v1"},
            pdf_locations=["https://arxiv.example/1v1.pdf"],
        )
        library.update_on_location_read(
            lambda: writer.record_source(
                paper_id,
                "arxiv",
                {"version": "v2"},
                pdf_locations=["https://arxiv.example/1v2.pdf"],
            )
        )
        [source] = library.paper_sources(paper_id)
    assert library.interleaved
    assert source.metadata == {"version": "v2"}
    assert source.pdf_locations == ("https://arxiv.example/1v2.pdf",)


def test_identifier_lookup_and_listing(library: Library) -> None:
    paper_id = library.add_paper(
        title="Identified", authors=["Ada Lovelace", "Alan Turing"]
    )
    library.add_identifier(paper_id, "doi", "10.1000/example")
    library.add_identifier(paper_id, "arxiv", "2401.00001")
    found = library.find_paper_by_identifier("doi", "10.1000/example")
    assert found is not None
    assert found.paper_id == paper_id
    assert found.authors == ("Ada Lovelace", "Alan Turing")
    assert library.find_paper_by_identifier("arxiv", "2401.00001").paper_id == paper_id
    assert library.find_paper_by_identifier("doi", "10.1000/other") is None
    assert library.identifiers(paper_id) == (
        Identifier(namespace="arxiv", value="2401.00001"),
        Identifier(namespace="doi", value="10.1000/example"),
    )


def test_identifier_cannot_attach_to_two_papers(library: Library) -> None:
    first = library.add_paper(title="First")
    second = library.add_paper(title="Second")
    library.add_identifier(first, "doi", "10.1000/duplicate")
    with pytest.raises(sqlite3.IntegrityError):
        library.add_identifier(second, "doi", "10.1000/duplicate")
    found = library.find_paper_by_identifier("doi", "10.1000/duplicate")
    assert found is not None
    assert found.paper_id == first
    assert library.identifiers(second) == ()


def test_identifier_for_missing_paper_violates_foreign_key(library: Library) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        library.add_identifier("missing", "doi", "10.1000/missing")


def test_paper_source_shape_and_provider_key(library: Library) -> None:
    paper_id = library.add_paper(title="Sourced")
    library.record_source(paper_id, "openalex", {"id": "W1"})
    [source] = library.paper_sources(paper_id)
    assert [field.name for field in fields(PaperSource)] == [
        "paper_id",
        "provider",
        "metadata",
        "pdf_locations",
        "url",
        "retrieved_at",
    ]
    assert (source.paper_id, source.provider) == (paper_id, "openalex")
    assert source.metadata == {"id": "W1"}
    assert source.pdf_locations == ()
    assert source.url is None
    assert source.retrieved_at
    assert library.paper_sources("missing") == ()


def test_paper_sources_are_retained_per_provider(library: Library) -> None:
    paper_id = library.add_paper(title="Sourced")
    library.record_source(
        paper_id,
        "openalex",
        {"id": "W1", "counts": {"cited_by": 3}},
        url="https://openalex.example/W1",
    )
    library.record_source(paper_id, "arxiv", {"id": "2401.00001"})
    library.record_source(paper_id, "openalex", {"id": "W1", "counts": {"cited_by": 4}})
    records = {record.provider: record for record in library.paper_sources(paper_id)}
    assert all(record.paper_id == paper_id for record in records.values())
    assert records["openalex"].metadata == {"id": "W1", "counts": {"cited_by": 4}}
    assert records["openalex"].url == "https://openalex.example/W1"
    assert records["arxiv"].metadata == {"id": "2401.00001"}
    assert records["arxiv"].url is None
    assert all(record.pdf_locations == () for record in records.values())
    assert all(record.retrieved_at for record in records.values())


def test_pdf_locations_change_only_when_supplied(library: Library) -> None:
    paper_id = library.add_paper(title="Located")
    arxiv_locations = (
        "https://arxiv.example/pdf/2401.00001v2",
        "https://arxiv.example/pdf/2401.00001v1",
    )
    openalex_locations = ("https://openalex.example/W1.pdf",)
    library.record_source(
        paper_id, "arxiv", {"id": "2401.00001"}, pdf_locations=arxiv_locations
    )
    library.record_source(
        paper_id, "openalex", {"id": "W1"}, pdf_locations=openalex_locations
    )
    library.record_source(paper_id, "arxiv", {"id": "2401.00001", "version": "v3"})
    records = {record.provider: record for record in library.paper_sources(paper_id)}
    assert records["arxiv"].pdf_locations == arxiv_locations
    assert records["openalex"].pdf_locations == openalex_locations

    library.record_source(
        paper_id,
        "arxiv",
        {"id": "2401.00001"},
        pdf_locations=["https://mirror.example/2401.00001.pdf"],
    )
    records = {record.provider: record for record in library.paper_sources(paper_id)}
    assert records["arxiv"].pdf_locations == ("https://mirror.example/2401.00001.pdf",)
    assert records["openalex"].pdf_locations == openalex_locations

    library.record_source(paper_id, "arxiv", {"id": "2401.00001"}, pdf_locations=[])
    records = {record.provider: record for record in library.paper_sources(paper_id)}
    assert records["arxiv"].pdf_locations == ()
    assert records["openalex"].pdf_locations == openalex_locations


@pytest.mark.parametrize("existing", [False, True])
def test_pdf_locations_reject_a_bare_string_without_changing_sources(
    library: Library, existing: bool
) -> None:
    paper_id = library.add_paper(title="Located")
    if existing:
        library.record_source(
            paper_id,
            "arxiv",
            {"version": "v1"},
            pdf_locations=["https://arxiv.example/paper.pdf"],
        )
    before = library.paper_sources(paper_id)
    with pytest.raises(TypeError, match="pdf_locations"):
        library.record_source(
            paper_id,
            "arxiv",
            {"version": "v2"},
            pdf_locations="https://arxiv.example/paper.pdf",
        )
    assert library.paper_sources(paper_id) == before


def test_record_source_for_missing_paper_violates_foreign_key(
    library: Library,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        library.record_source("missing", "openalex", {"id": "W1"})


def test_one_pdf_and_one_extracted_text_coexist(library: Library) -> None:
    paper_id = library.add_paper(title="Main pair")
    pdf = library.add_artefact_bytes(
        paper_id, b"%PDF-1.7", role=ArtefactRole.PDF, media_type="application/pdf"
    )
    text = library.add_artefact_bytes(
        paper_id, b"extracted text", role=ArtefactRole.EXTRACTED_TEXT
    )
    links = library.paper_artefacts(paper_id)
    assert {(link.artefact_id, link.role) for link in links} == {
        (pdf, ArtefactRole.PDF),
        (text, ArtefactRole.EXTRACTED_TEXT),
    }
    assert {link.source for link in links} == {None}


@pytest.mark.parametrize(
    ("role", "media_type"),
    [
        (ArtefactRole.PDF, "application/pdf"),
        (ArtefactRole.EXTRACTED_TEXT, "text/plain"),
    ],
)
def test_a_paper_holds_one_artefact_per_main_role(
    library: Library, role: ArtefactRole, media_type: str
) -> None:
    paper_id = library.add_paper(title="One main artefact")
    recorded = library.add_artefact_bytes(
        paper_id, b"first", role=role, media_type=media_type, source="arxiv"
    )
    with pytest.raises(
        sqlite3.IntegrityError, match="paper_artefacts.paper_id, paper_artefacts.role"
    ):
        library.add_artefact_bytes(
            paper_id, b"second", role=role, media_type=media_type, source="openalex"
        )
    [link] = library.paper_artefacts(paper_id)
    assert (link.artefact_id, link.source) == (recorded, "arxiv")
    assert link.path.read_bytes() == b"first"
    # The rejected artefact leaves no record but may leave its completed blob.
    assert library.get_artefact(digest(b"second")) is None
    assert library.artefact_path(digest(b"second")).read_bytes() == b"second"


def test_repeated_main_artefact_keeps_one_link_and_first_provenance(
    library: Library,
) -> None:
    paper_id = library.add_paper(title="Repeated main artefact")
    first = library.add_artefact_bytes(
        paper_id,
        b"%PDF-1.7",
        role=ArtefactRole.PDF,
        source="arxiv",
        url="https://arxiv.example/pdf/1",
    )
    second = library.add_artefact_bytes(
        paper_id,
        b"%PDF-1.7",
        role=ArtefactRole.PDF,
        source="openalex",
        url="https://openalex.example/pdf/1",
    )
    assert first == second
    [link] = library.paper_artefacts(paper_id)
    assert (link.source, link.url) == ("arxiv", "https://arxiv.example/pdf/1")


@pytest.mark.parametrize("from_file", [False, True])
def test_invalid_role_does_not_store_an_artefact(
    library: Library, tmp_path: Path, from_file: bool
) -> None:
    paper_id = library.add_paper(title="Invalid role")
    data = b"unretained input"
    source = tmp_path / "source.txt"
    source.write_bytes(data)
    with pytest.raises(ValueError, match="not-a-role"):
        if from_file:
            library.add_artefact_file(paper_id, source, role="not-a-role")
        else:
            library.add_artefact_bytes(paper_id, data, role="not-a-role")
    assert library.paper_artefacts(paper_id) == ()
    assert library.get_artefact(digest(data)) is None
    assert list((library.root / "artefacts").iterdir()) == []
    assert source.read_bytes() == data


def test_supplementary_artefacts_are_unlimited(library: Library) -> None:
    paper_id = library.add_paper(title="Supplementary set")
    stored = [
        library.add_artefact_bytes(
            paper_id,
            f"review {index}".encode(),
            role=ArtefactRole.SUPPLEMENTARY,
            source="openreview",
        )
        for index in range(3)
    ]
    assert [link.artefact_id for link in library.paper_artefacts(paper_id)] == sorted(
        stored
    )


def test_main_artefact_limits_are_per_paper(library: Library) -> None:
    first, second = (library.add_paper(title=title) for title in ("First", "Second"))
    first_pdf = library.add_artefact_bytes(
        first, b"%PDF-1.7 first", role=ArtefactRole.PDF
    )
    second_pdf = library.add_artefact_bytes(
        second, b"%PDF-1.7 second", role=ArtefactRole.PDF
    )
    with pytest.raises(
        sqlite3.IntegrityError, match="paper_artefacts.paper_id, paper_artefacts.role"
    ):
        library.add_artefact_bytes(first, b"%PDF-1.7 second", role=ArtefactRole.PDF)
    [link] = library.paper_artefacts(first)
    assert link.artefact_id == first_pdf
    [link] = library.paper_artefacts(second)
    assert link.artefact_id == second_pdf


def test_direct_writes_enforce_one_main_artefact_per_paper(library: Library) -> None:
    paper_id = library.add_paper(title="Direct")
    other = library.add_paper(title="Other")
    recorded = library.add_artefact_bytes(
        paper_id, b"%PDF-1.7 recorded", role=ArtefactRole.PDF
    )
    spare = library.add_artefact_bytes(other, b"%PDF-1.7 spare", role=ArtefactRole.PDF)
    connection = sqlite3.connect(library.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="paper_artefacts.paper_id, paper_artefacts.role",
        ):
            connection.execute(
                "INSERT INTO paper_artefacts (paper_id, artefact_id, role,"
                " created_at) VALUES (?, ?, ?, ?)",
                (paper_id, spare, ArtefactRole.PDF.value, "now"),
            )
        for artefact_id in (recorded, spare):
            connection.execute(
                "INSERT INTO paper_artefacts (paper_id, artefact_id, role,"
                " created_at) VALUES (?, ?, ?, ?)",
                (paper_id, artefact_id, ArtefactRole.SUPPLEMENTARY.value, "now"),
            )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM paper_artefacts WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()[0]
            == 3
        )
    finally:
        connection.rollback()
        connection.close()


def test_supplementary_artefact_records_provenance(library: Library) -> None:
    paper_id = library.add_paper(title="Supplementary")
    artefact_id = library.add_artefact_bytes(
        paper_id,
        b"<html>reviews</html>",
        role=ArtefactRole.SUPPLEMENTARY,
        source="openreview",
        url="https://openreview.example/forum?id=1",
    )
    [link] = library.paper_artefacts(paper_id)
    assert link.artefact_id == artefact_id
    assert (link.source, link.url) == (
        "openreview",
        "https://openreview.example/forum?id=1",
    )
    assert link.path.read_bytes() == b"<html>reviews</html>"


def test_shared_content_is_stored_once(library: Library) -> None:
    data = b"%PDF-1.7 shared"
    first = library.add_paper(title="First")
    second = library.add_paper(title="Second")
    first_artefact = library.add_artefact_bytes(first, data, role=ArtefactRole.PDF)
    second_artefact = library.add_artefact_bytes(second, data, role=ArtefactRole.PDF)
    assert first_artefact == second_artefact == digest(data)
    artefact = library.get_artefact(first_artefact)
    assert artefact is not None
    assert artefact.byte_size == len(data)
    assert [path.name for path in artefact.path.parent.iterdir()] == [digest(data)]
    assert library.paper_artefacts(first)[0].artefact_id == first_artefact
    assert library.paper_artefacts(second)[0].artefact_id == second_artefact


def test_imported_file_keeps_source_and_uses_managed_path(
    library: Library, tmp_path: Path
) -> None:
    paper_id = library.add_paper(title="Imported")
    source = tmp_path / "Downloads" / "final final.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7 imported")
    before = source.stat()
    artefact_id = library.add_artefact_file(
        paper_id, source, role=ArtefactRole.PDF, media_type="application/pdf"
    )
    after = source.stat()
    assert source.read_bytes() == b"%PDF-1.7 imported"
    assert after.st_mtime_ns == before.st_mtime_ns
    path = library.artefact_path(artefact_id)
    assert path.name == digest(b"%PDF-1.7 imported")
    assert path.parent == library.root / "artefacts"
    assert "final final" not in str(path)


@pytest.mark.parametrize(
    "artefact_id", ["", "../../etc/passwd", "0" * 63, "0" * 65, "A" * 64]
)
def test_artefact_path_rejects_non_digests(library: Library, artefact_id: str) -> None:
    with pytest.raises(ValueError, match="not a SHA-256 artefact ID"):
        library.artefact_path(artefact_id)


@pytest.mark.parametrize("root", [Path("library"), Path("sub/../library")])
def test_import_with_a_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    data = b"%PDF-1.7 relative"
    source = tmp_path / "download.pdf"
    source.write_bytes(data)
    with Library(root) as library:
        paper_id = library.add_paper(title="Relative")
        artefact_id = library.add_artefact_file(paper_id, source, role=ArtefactRole.PDF)
        assert library.artefact_path(artefact_id).read_bytes() == data
        assert library.paper_artefacts(paper_id)[0].artefact_id == artefact_id


def test_failed_link_rolls_back_the_artefact_record(library: Library) -> None:
    data = b"%PDF-1.7 unlinked"
    with pytest.raises(sqlite3.IntegrityError):
        library.add_artefact_bytes("missing", data, role=ArtefactRole.PDF)
    assert library.get_artefact(digest(data)) is None
    # A completed but unreferenced blob is acceptable; no temporary file remains.
    assert [
        path.name for path in library.artefact_path(digest(data)).parent.iterdir()
    ] == [digest(data)]


def test_failed_write_removes_the_artefact_record(
    library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_id = library.add_paper(title="Interrupted")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(artefacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        library.add_artefact_bytes(paper_id, b"%PDF-1.7", role=ArtefactRole.PDF)
    assert library.get_artefact(digest(b"%PDF-1.7")) is None
    assert list((library.root / "artefacts").iterdir()) == []


def test_partial_import_copy_leaves_no_artefact(
    library: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_id = library.add_paper(title="Partial import")
    data = b"%PDF-1.7 partial"
    source = tmp_path / "download.pdf"
    source.write_bytes(data)

    def fail_after_partial_copy(source_path: Path, destination: Path) -> None:
        Path(destination).write_bytes(data[:4])
        raise OSError("simulated copy failure")

    monkeypatch.setattr(artefacts.shutil, "copyfile", fail_after_partial_copy)
    with pytest.raises(OSError, match="simulated"):
        library.add_artefact_file(paper_id, source, role=ArtefactRole.PDF)
    assert library.get_artefact(digest(data)) is None
    assert library.paper_artefacts(paper_id) == ()
    assert list((library.root / "artefacts").iterdir()) == []
    assert source.read_bytes() == data


def test_citations_are_idempotent_and_traverse_both_directions(
    library: Library,
) -> None:
    citing, cited, other = (library.add_paper(title=name) for name in ("A", "B", "C"))
    library.add_citation(citing, cited, source="openalex", url="https://example/A")
    library.add_citation(citing, other)
    library.add_citation(citing, cited, source="openalex", url="https://example/A")
    assert len(library.cited_papers(citing)) == 2
    assert set(library.cited_papers(citing)) == {cited, other}
    assert library.citing_papers(cited) == (citing,)
    assert library.citing_papers(other) == (citing,)
    assert library.citing_papers(citing) == ()
    assert library.cited_papers(cited) == ()


def test_citation_for_missing_paper_violates_foreign_key(library: Library) -> None:
    paper_id = library.add_paper(title="Present")
    with pytest.raises(sqlite3.IntegrityError):
        library.add_citation("missing", paper_id)
    with pytest.raises(sqlite3.IntegrityError):
        library.add_citation(paper_id, "missing")


@pytest.mark.parametrize("level", EnrichmentLevel)
def test_promotion_is_monotonic(library: Library, level: EnrichmentLevel) -> None:
    paper_id = library.add_paper(title="Levelled")
    assert library.promote(paper_id, level) == level
    assert library.get_paper(paper_id).enrichment_level == level
    levels = tuple(EnrichmentLevel)
    for lower in levels[: levels.index(level)]:
        assert library.promote(paper_id, lower) == level
        assert library.get_paper(paper_id).enrichment_level == level


def test_promote_missing_paper_is_reported(library: Library) -> None:
    with pytest.raises(LibraryError, match="paper not found"):
        library.promote("missing", EnrichmentLevel.SCREENING_READY)


def test_state_survives_close_and_reopen(tmp_path: Path) -> None:
    root = tmp_path / "library"
    with Library(root) as library:
        paper_id = library.add_paper(
            title="Persisted", authors=["Alan Turing", "Ada Lovelace"]
        )
        library.add_identifier(paper_id, "doi", "10.1000/persisted")
        library.record_source(
            paper_id, "openalex", {"id": "W1"}, pdf_locations=["https://example/1.pdf"]
        )
        artefact_id = library.add_artefact_bytes(
            paper_id,
            b"%PDF-1.7 persisted",
            role=ArtefactRole.PDF,
            source="arxiv",
            url="https://arxiv.example/pdf/1v1",
        )
        citing = library.add_paper(title="Citing")
        library.add_citation(citing, paper_id)
        library.promote(paper_id, EnrichmentLevel.READING_READY)
    with Library(root) as reopened:
        paper = reopened.get_paper(paper_id)
        assert paper is not None
        assert paper.title == "Persisted"
        assert paper.enrichment_level == EnrichmentLevel.READING_READY
        assert reopened.get_paper(paper_id).authors == ("Alan Turing", "Ada Lovelace")
        found = reopened.find_paper_by_identifier("doi", "10.1000/persisted")
        assert found is not None
        assert found.authors == ("Alan Turing", "Ada Lovelace")
        [source] = reopened.paper_sources(paper_id)
        assert source.metadata == {"id": "W1"}
        assert source.pdf_locations == ("https://example/1.pdf",)
        [link] = reopened.paper_artefacts(paper_id)
        assert link.artefact_id == artefact_id
        assert (link.role, link.source, link.url) == (
            ArtefactRole.PDF,
            "arxiv",
            "https://arxiv.example/pdf/1v1",
        )
        assert reopened.artefact_path(artefact_id).read_bytes() == b"%PDF-1.7 persisted"
        assert reopened.citing_papers(paper_id) == (citing,)


def test_paper_retains_sources_locations_and_discussion_across_reopen(
    tmp_path: Path,
) -> None:
    """One paper keeps provider metadata, its main pair and supplementary discussion."""
    root = tmp_path / "library"
    review_body = "SENTINEL review body paragraph. " * 25
    discussion = json.dumps(
        {
            "forum": "forum-1",
            "notes": [
                {
                    "id": f"note-{index}",
                    "invitation": "Synthetic/-/Official_Review",
                    "content": {"review": review_body, "rating": index % 10},
                }
                for index in range(10)
            ],
        }
    ).encode()
    pdf_bytes = b"%PDF-1.7 selected main document"
    text_bytes = b"Extracted text for the selected main document."
    arxiv_locations = [
        "https://arxiv.example/pdf/2401.00001v2",
        "https://arxiv.example/pdf/2401.00001v1",
        "https://mirror.example/papers/2401.00001.pdf",
    ]
    openalex_locations = [
        "https://openalex.example/W000000001.pdf",
        "https://repository.example/bitstream/2401.00001.pdf",
    ]
    with Library(root) as library:
        paper_id = library.add_paper(
            title="Synthetic provider paper",
            abstract="A synthetic paper used to exercise storage conformance.",
            authors=("Ada Lovelace",),
            published="2024-06-01",
        )
        library.add_identifier(paper_id, "arxiv", "2401.00001")
        library.add_identifier(paper_id, "openreview", "forum-1")
        library.record_source(
            paper_id,
            "arxiv",
            {"id": "2401.00001", "version": "v2"},
            pdf_locations=arxiv_locations,
            url="https://arxiv.example/abs/2401.00001",
        )
        library.record_source(
            paper_id,
            "openalex",
            {"id": "W000000001"},
            pdf_locations=openalex_locations,
            url="https://openalex.example/W000000001",
        )
        library.record_source(
            paper_id,
            "openreview",
            {
                "forum": "forum-1",
                "submission_number": 17,
                "revision": "v3",
                "decision": "Accept (poster)",
                "venue": "Synthetic Conference 2024",
            },
            url="https://openreview.example/forum?id=forum-1",
        )
        pdf_id = library.add_artefact_bytes(
            paper_id,
            pdf_bytes,
            role=ArtefactRole.PDF,
            media_type="application/pdf",
            source="arxiv",
            url=arxiv_locations[0],
        )
        text_id = library.add_artefact_bytes(
            paper_id,
            text_bytes,
            role=ArtefactRole.EXTRACTED_TEXT,
            media_type="text/plain",
            source="arxiv",
            url=arxiv_locations[0],
        )
        discussion_id = library.add_artefact_bytes(
            paper_id,
            discussion,
            role=ArtefactRole.SUPPLEMENTARY,
            media_type="application/json",
            source="openreview",
            url="https://openreview.example/forum?id=forum-1",
        )
    with Library(root) as reopened:
        paper = reopened.get_paper(paper_id)
        assert paper is not None
        assert paper.title == "Synthetic provider paper"
        assert set(reopened.identifiers(paper_id)) == {
            Identifier(namespace="arxiv", value="2401.00001"),
            Identifier(namespace="openreview", value="forum-1"),
        }
        records = {
            record.provider: record for record in reopened.paper_sources(paper_id)
        }
        assert set(records) == {"arxiv", "openalex", "openreview"}
        assert {record.paper_id for record in records.values()} == {paper_id}
        assert records["arxiv"].metadata == {"id": "2401.00001", "version": "v2"}
        assert records["arxiv"].pdf_locations == tuple(arxiv_locations)
        assert records["arxiv"].url == "https://arxiv.example/abs/2401.00001"
        assert records["openalex"].metadata == {"id": "W000000001"}
        assert records["openalex"].pdf_locations == tuple(openalex_locations)
        assert records["openreview"].pdf_locations == ()
        assert records["openreview"].metadata == {
            "forum": "forum-1",
            "submission_number": 17,
            "revision": "v3",
            "decision": "Accept (poster)",
            "venue": "Synthetic Conference 2024",
        }
        assert (
            records["openreview"].url == "https://openreview.example/forum?id=forum-1"
        )

        links = {link.role: link for link in reopened.paper_artefacts(paper_id)}
        assert set(links) == set(ArtefactRole)
        assert links[ArtefactRole.PDF].artefact_id == pdf_id
        assert links[ArtefactRole.EXTRACTED_TEXT].artefact_id == text_id
        assert links[ArtefactRole.SUPPLEMENTARY].artefact_id == discussion_id
        assert links[ArtefactRole.PDF].path.read_bytes() == pdf_bytes
        assert links[ArtefactRole.EXTRACTED_TEXT].path.read_bytes() == text_bytes
        assert links[ArtefactRole.SUPPLEMENTARY].path.read_bytes() == discussion
        assert len(discussion) > 1000
        assert (links[ArtefactRole.PDF].source, links[ArtefactRole.PDF].url) == (
            "arxiv",
            arxiv_locations[0],
        )
        assert (
            links[ArtefactRole.SUPPLEMENTARY].source,
            links[ArtefactRole.SUPPLEMENTARY].url,
        ) == ("openreview", "https://openreview.example/forum?id=forum-1")
        assert b"SENTINEL" in links[ArtefactRole.SUPPLEMENTARY].path.read_bytes()

        # Candidate PDF locations stay in provider records, not the artefact store.
        stored_urls = {link.url for link in links.values()}
        unused = (set(arxiv_locations) | set(openalex_locations)) - stored_urls
        assert len(unused) == 4
        assert len(list((root / "artefacts").iterdir())) == 3

        # Full review bodies live only in the supplementary payload.
        compact = json.dumps([record.metadata for record in records.values()])
        assert "SENTINEL" not in compact


def test_reopening_keeps_the_schema_version(tmp_path: Path) -> None:
    root = tmp_path / "library"
    with Library(root):
        pass
    with Library(root):
        pass
    connection = sqlite3.connect(root / DATABASE_FILENAME)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        connection.close()


def test_unsupported_schema_version_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "library"
    with Library(root):
        pass
    connection = sqlite3.connect(root / DATABASE_FILENAME)
    try:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(LibraryError, match="schema version"):
        Library(root)


def test_constraints_apply_to_direct_database_writes(library: Library) -> None:
    paper_id = library.add_paper(title="Direct")
    library.add_identifier(paper_id, "doi", "10.1000/direct")
    connection = sqlite3.connect(library.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO external_identifiers (namespace, value, paper_id,"
                " created_at) VALUES (?, ?, ?, ?)",
                ("doi", "10.1000/direct", paper_id, "now"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_artefacts (paper_id, artefact_id, role,"
                " created_at) VALUES (?, ?, ?, ?)",
                (paper_id, "0" * 64, ArtefactRole.SUPPLEMENTARY.value, "now"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO papers (paper_id, enrichment_level, created_at,"
                " updated_at) VALUES (?, ?, ?, ?)",
                ("paper", "unknown", "now", "now"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_sources (paper_id, provider, metadata, url,"
                " retrieved_at) VALUES (?, ?, ?, ?, ?)",
                (paper_id, "openalex", "not json", None, "now"),
            )
    finally:
        connection.rollback()
        connection.close()


def test_source_is_unique_per_paper_and_provider(library: Library) -> None:
    paper_id = library.add_paper(title="Direct")
    library.record_source(paper_id, "arxiv", {"id": "2401.00001"})
    connection = sqlite3.connect(library.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_sources (paper_id, provider, metadata,"
                " retrieved_at) VALUES (?, ?, ?, ?)",
                (paper_id, "arxiv", "{}", "now"),
            )
    finally:
        connection.close()
    [source] = library.paper_sources(paper_id)
    assert source.metadata == {"id": "2401.00001"}


def test_pdf_locations_are_ordered_child_rows(library: Library) -> None:
    paper_id = library.add_paper(title="Located")
    locations = (
        "https://arxiv.example/pdf/2401.00001v2",
        "https://mirror.example/2401.00001.pdf",
        "https://arxiv.example/pdf/2401.00001v2",
        "",
    )
    library.record_source(
        paper_id, "arxiv", {"id": "2401.00001"}, pdf_locations=locations
    )
    connection = sqlite3.connect(library.database_path)
    try:
        rows = connection.execute(
            "SELECT position, url FROM paper_pdf_locations"
            " WHERE paper_id = ? AND provider = ? ORDER BY position",
            (paper_id, "arxiv"),
        ).fetchall()
        source_columns = {
            row[1] for row in connection.execute("PRAGMA table_info('paper_sources')")
        }
    finally:
        connection.close()
    assert rows == [
        (0, locations[0]),
        (1, locations[1]),
        (2, locations[2]),
        (3, ""),
    ]
    assert "pdf_locations" not in source_columns
    [source] = library.paper_sources(paper_id)
    assert source.pdf_locations == locations


def test_source_location_schema_uses_a_composite_key_and_cascade(
    library: Library,
) -> None:
    connection = sqlite3.connect(library.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        columns = {
            row[1]: (row[2], row[3], row[5])
            for row in connection.execute("PRAGMA table_info('paper_pdf_locations')")
        }
        foreign_keys = {
            (row[2], row[3], row[4], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list('paper_pdf_locations')"
            )
        }
        source_columns = {
            row[1] for row in connection.execute("PRAGMA table_info('paper_sources')")
        }
    finally:
        connection.close()
    assert columns == {
        "paper_id": ("TEXT", 1, 1),
        "provider": ("TEXT", 1, 2),
        "position": ("INTEGER", 1, 3),
        "url": ("TEXT", 1, 0),
    }
    assert foreign_keys == {
        ("paper_sources", "paper_id", "paper_id", "CASCADE"),
        ("paper_sources", "provider", "provider", "CASCADE"),
    }
    assert "pdf_locations" not in source_columns


def test_source_location_constraints_apply_to_direct_writes(library: Library) -> None:
    paper_id = library.add_paper(title="Direct")
    connection = sqlite3.connect(library.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.execute(
            "INSERT INTO paper_sources (paper_id, provider, metadata, retrieved_at)"
            " VALUES (?, 'arxiv', '{}', 'now')",
            (paper_id,),
        )
        connection.execute(
            "INSERT INTO paper_pdf_locations (paper_id, provider, position, url)"
            " VALUES (?, 'arxiv', 0, 'https://arxiv.example/1.pdf')",
            (paper_id,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_pdf_locations (paper_id, provider, position, url)"
                " VALUES (?, 'arxiv', -1, 'https://arxiv.example/negative.pdf')",
                (paper_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_pdf_locations (paper_id, provider, position, url)"
                " VALUES (?, 'arxiv', 1, NULL)",
                (paper_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_pdf_locations (paper_id, provider, position, url)"
                " VALUES (?, 'arxiv', 0, 'https://arxiv.example/duplicate.pdf')",
                (paper_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_pdf_locations (paper_id, provider, position, url)"
                " VALUES ('missing', 'arxiv', 0, 'https://arxiv.example/orphan.pdf')"
            )
        connection.execute(
            "DELETE FROM paper_sources WHERE paper_id = ? AND provider = 'arxiv'",
            (paper_id,),
        )
        remaining = connection.execute(
            "SELECT COUNT(*) FROM paper_pdf_locations WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()[0]
    finally:
        connection.rollback()
        connection.close()
    assert remaining == 0
    assert library.paper_sources(paper_id) == ()


def test_deleting_paper_cascades_sources_and_locations(library: Library) -> None:
    paper_id = library.add_paper(title="Cascade")
    library.record_source(
        paper_id, "arxiv", {}, pdf_locations=["https://arxiv.example/1.pdf"]
    )
    with sqlite3.connect(library.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM paper_sources WHERE paper_id = ?", (paper_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM paper_pdf_locations WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()[0]
            == 0
        )


def test_failed_location_insert_leaves_no_source_rows(library: Library) -> None:
    paper_id = library.add_paper(title="Failed")
    with pytest.raises(sqlite3.IntegrityError):
        library.record_source(
            paper_id,
            "arxiv",
            {"id": "2401.00001"},
            pdf_locations=["https://arxiv.example/1.pdf", None],
        )
    assert library.paper_sources(paper_id) == ()
    connection = sqlite3.connect(library.database_path)
    try:
        sources = connection.execute(
            "SELECT COUNT(*) FROM paper_sources WHERE paper_id = ?", (paper_id,)
        ).fetchone()[0]
        locations = connection.execute(
            "SELECT COUNT(*) FROM paper_pdf_locations WHERE paper_id = ?", (paper_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert (sources, locations) == (0, 0)


def test_failed_location_update_restores_source_and_locations(
    library: Library,
) -> None:
    paper_id = library.add_paper(title="Failed update")
    library.record_source(
        paper_id,
        "arxiv",
        {"id": "2401.00001", "version": "v1"},
        pdf_locations=[
            "https://arxiv.example/1v1.pdf",
            "https://mirror.example/2401.00001.pdf",
        ],
        url="https://arxiv.example/abs/2401.00001v1",
    )
    [before] = library.paper_sources(paper_id)
    with pytest.raises(sqlite3.IntegrityError):
        library.record_source(
            paper_id,
            "arxiv",
            {"id": "2401.00001", "version": "v2"},
            pdf_locations=["https://arxiv.example/1v2.pdf", None],
            url="https://arxiv.example/abs/2401.00001v2",
        )
    [after] = library.paper_sources(paper_id)
    assert after.metadata == before.metadata == {"id": "2401.00001", "version": "v1"}
    assert after.url == before.url == "https://arxiv.example/abs/2401.00001v1"
    assert after.retrieved_at == before.retrieved_at
    assert (
        after.pdf_locations
        == before.pdf_locations
        == (
            "https://arxiv.example/1v1.pdf",
            "https://mirror.example/2401.00001.pdf",
        )
    )


def test_artefact_directory_sync_precedes_database_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_sync = artefacts._fsync_directory

    def record_sync(path: Path) -> None:
        events.append("fsync-directory")
        real_sync(path)

    monkeypatch.setattr(artefacts, "_fsync_directory", record_sync)

    class RecordingLibrary(Library):
        @contextmanager
        def _transaction(self) -> Iterator[sqlite3.Connection]:
            events.append("begin")
            with super()._transaction() as connection:
                yield connection
            events.append("commit")

    with RecordingLibrary(tmp_path / "library") as library:
        paper_id = library.add_paper()
        library.add_artefact_bytes(paper_id, b"%PDF-1.7 ordered", role=ArtefactRole.PDF)

    assert events[-3:] == ["fsync-directory", "begin", "commit"]


def test_directory_sync_failure_prevents_database_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_sync(path: Path) -> None:
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(artefacts, "_fsync_directory", fail_sync)
    data = b"%PDF-1.7 unsynced"
    with Library(tmp_path / "library") as library:
        paper_id = library.add_paper(title="Unsynced")
        with pytest.raises(OSError, match="simulated"):
            library.add_artefact_bytes(paper_id, data, role=ArtefactRole.PDF)
        assert library.get_artefact(digest(data)) is None
        assert library.paper_artefacts(paper_id) == ()
        assert [path.name for path in (library.root / "artefacts").iterdir()] == [
            digest(data)
        ]


def test_media_type_can_upgrade_from_the_default(library: Library) -> None:
    data = b"%PDF-1.7 upgraded"
    first = library.add_paper(title="First")
    second = library.add_paper(title="Second")
    artefact_id = library.add_artefact_bytes(
        first, data, role=ArtefactRole.SUPPLEMENTARY
    )
    artefact = library.get_artefact(artefact_id)
    assert artefact is not None
    assert artefact.media_type == DEFAULT_MEDIA_TYPE
    library.add_artefact_bytes(
        second, data, role=ArtefactRole.PDF, media_type="application/pdf"
    )
    artefact = library.get_artefact(artefact_id)
    assert artefact is not None
    assert artefact.media_type == "application/pdf"
    [link] = library.paper_artefacts(second)
    assert link.media_type == "application/pdf"


def test_default_media_type_does_not_downgrade(library: Library) -> None:
    data = b"%PDF-1.7 specific"
    first = library.add_paper(title="First")
    second = library.add_paper(title="Second")
    artefact_id = library.add_artefact_bytes(
        first, data, role=ArtefactRole.PDF, media_type="application/pdf"
    )
    library.add_artefact_bytes(second, data, role=ArtefactRole.SUPPLEMENTARY)
    artefact = library.get_artefact(artefact_id)
    assert artefact is not None
    assert artefact.media_type == "application/pdf"


def test_conflicting_specific_media_types_are_rejected(library: Library) -> None:
    data = b"%PDF-1.7 conflicting"
    first = library.add_paper(title="First")
    second = library.add_paper(title="Second")
    artefact_id = library.add_artefact_bytes(
        first, data, role=ArtefactRole.PDF, media_type="application/pdf"
    )
    with pytest.raises(LibraryError, match="already registered"):
        library.add_artefact_bytes(
            second, data, role=ArtefactRole.PDF, media_type="text/plain"
        )
    artefact = library.get_artefact(artefact_id)
    assert artefact is not None
    assert artefact.media_type == "application/pdf"
    assert library.paper_artefacts(second) == ()
    library.add_artefact_bytes(second, data, role=ArtefactRole.SUPPLEMENTARY)
    assert len(library.paper_artefacts(second)) == 1


def test_repeated_artefact_link_keeps_first_provenance(library: Library) -> None:
    paper_id = library.add_paper(title="Provenance")
    data = b"<html>reviews</html>"
    library.add_artefact_bytes(
        paper_id,
        data,
        role=ArtefactRole.SUPPLEMENTARY,
        source="openreview",
        url="https://openreview.example/first",
    )
    library.add_artefact_bytes(
        paper_id,
        data,
        role=ArtefactRole.SUPPLEMENTARY,
        source="snapshot",
        url="https://example/second",
    )
    [link] = library.paper_artefacts(paper_id)
    assert (link.source, link.url) == (
        "openreview",
        "https://openreview.example/first",
    )


def test_paper_author_constraints_apply_to_direct_writes(library: Library) -> None:
    paper_id = library.add_paper()
    connection = sqlite3.connect(library.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_authors (paper_id, position, name)"
                " VALUES ('missing', 0, 'Ada Lovelace')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_authors (paper_id, position, name)"
                " VALUES (?, -1, 'Ada Lovelace')",
                (paper_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_authors (paper_id, position, name)"
                " VALUES (?, 0, NULL)",
                (paper_id,),
            )
        connection.execute(
            "INSERT INTO paper_authors (paper_id, position, name)"
            " VALUES (?, 0, 'Ada Lovelace')",
            (paper_id,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO paper_authors (paper_id, position, name)"
                " VALUES (?, 0, 'Alan Turing')",
                (paper_id,),
            )
    finally:
        connection.rollback()
        connection.close()
    assert library.get_paper(paper_id).authors == ()


def test_author_schema_indexes_names_and_drops_the_json_column(
    library: Library,
) -> None:
    connection = sqlite3.connect(library.database_path)
    try:
        paper_columns = {
            row[1] for row in connection.execute("PRAGMA table_info('papers')")
        }
        indexes = {
            row[1]: row
            for row in connection.execute("PRAGMA index_list('paper_authors')")
        }
        indexed_columns = [
            row[2]
            for row in connection.execute("PRAGMA index_info('paper_authors_name')")
        ]
    finally:
        connection.close()
    assert "authors" not in paper_columns
    assert indexes["paper_authors_name"][2] == 0
    assert indexed_columns == ["name"]


def test_self_citation_is_rejected(library: Library) -> None:
    paper_id = library.add_paper(title="Self")
    with pytest.raises(sqlite3.IntegrityError):
        library.add_citation(paper_id, paper_id)
    connection = sqlite3.connect(library.database_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO citations (citing_paper_id, cited_paper_id,"
                " discovered_at) VALUES (?, ?, ?)",
                (paper_id, paper_id, "now"),
            )
    finally:
        connection.close()
    assert library.citing_papers(paper_id) == ()
    assert library.cited_papers(paper_id) == ()
