"""Persistent paper library: SQLite records and managed artefact files."""

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import groupby
from pathlib import Path
from typing import Any, Self

from platformdirs import user_data_path

from researcher.artefacts import ArtefactStore, Blob

SCHEMA_VERSION = 1
DATABASE_FILENAME = "library.sqlite3"
ARTEFACT_DIRECTORY = "artefacts"
DEFAULT_MEDIA_TYPE = "application/octet-stream"


class LibraryError(Exception):
    """The library cannot satisfy the request."""


class EnrichmentLevel(StrEnum):
    """The highest completeness level a paper has reached."""

    DISCOVERY = "discovery"
    SCREENING_READY = "screening_ready"
    ASSESSMENT_READY = "assessment_ready"
    READING_READY = "reading_ready"


class ArtefactRole(StrEnum):
    """How a retained artefact is used as an input."""

    PDF = "pdf"
    EXTRACTED_TEXT = "extracted_text"
    SUPPLEMENTARY = "supplementary"


# A paper selects one main PDF and reuses its extracted text; anything else is
# supplementary. The database enforces the limit on the two main roles.
_MAIN_ROLES = (ArtefactRole.PDF, ArtefactRole.EXTRACTED_TEXT)


@dataclass(frozen=True)
class Paper:
    """Shared identity and selected metadata for a bibliographic work."""

    paper_id: str
    title: str | None
    abstract: str | None
    authors: tuple[str, ...]
    published: str | None
    enrichment_level: EnrichmentLevel


@dataclass(frozen=True)
class Identifier:
    namespace: str
    value: str


@dataclass(frozen=True)
class PaperSource:
    """One provider's retained record for a paper."""

    paper_id: str
    provider: str
    metadata: Mapping[str, Any]
    pdf_locations: tuple[str, ...]
    url: str | None
    retrieved_at: str


@dataclass(frozen=True)
class Artefact:
    artefact_id: str
    byte_size: int
    media_type: str
    path: Path


@dataclass(frozen=True)
class ArtefactLink:
    artefact_id: str
    role: ArtefactRole
    media_type: str
    byte_size: int
    path: Path
    source: str | None
    url: str | None


class Library:
    """A paper library in the user data directory or an explicit location.

    The directory holds the SQLite database and a content-addressed artefact
    store. Callers own acquisition and enrichment policies; this class only
    records their results.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else user_data_path("researcher", appauthor=False, ensure_exists=True)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / DATABASE_FILENAME
        self._artefacts = ArtefactStore(self.root / ARTEFACT_DIRECTORY)
        self._connection = sqlite3.connect(self.database_path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._prepare_schema()
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def add_paper(
        self,
        *,
        title: str | None = None,
        abstract: str | None = None,
        authors: Sequence[str] = (),
        published: str | None = None,
    ) -> str:
        """Store a paper and return its stable internal ID.

        Every field is optional so that citation-only records can be kept.
        Author names are stored as ordered rows for this paper alone.
        """
        names = _author_names(authors)
        paper_id = str(uuid.uuid4())
        now = _utc_now()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO papers (
                    paper_id, title, abstract, published, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (paper_id, title, abstract, published, now, now),
            )
            _replace_authors(connection, paper_id, names)
        return paper_id

    def get_paper(self, paper_id: str) -> Paper | None:
        rows = self._connection.execute(
            f"{_PAPER_SELECT} WHERE p.paper_id = ? ORDER BY a.position", (paper_id,)
        ).fetchall()
        return _paper(rows)

    def update_paper(
        self,
        paper_id: str,
        *,
        title: str | None = None,
        abstract: str | None = None,
        authors: Sequence[str] | None = None,
        published: str | None = None,
    ) -> None:
        """Set the supplied fields, leaving omitted fields unchanged.

        A ``None`` scalar is omitted, while an empty string is supplied and
        stored. Supplied author names replace the paper's rows in order; an
        empty sequence clears them. The paper ID does not depend on metadata,
        so it survives every update.
        """
        names = None if authors is None else _author_names(authors)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE papers SET
                    title = COALESCE(?, title),
                    abstract = COALESCE(?, abstract),
                    published = COALESCE(?, published),
                    updated_at = ?
                WHERE paper_id = ?
                """,
                (title, abstract, published, _utc_now(), paper_id),
            )
            if cursor.rowcount == 0:
                raise LibraryError(f"paper not found: {paper_id}")
            if names is not None:
                _replace_authors(connection, paper_id, names)

    def add_identifier(self, paper_id: str, namespace: str, value: str) -> None:
        """Bind a normalised external identifier to a paper.

        The identifier primary key is global, so binding an already recorded
        identifier raises ``sqlite3.IntegrityError``, as does a missing paper.
        """
        self._connection.execute(
            """
            INSERT INTO external_identifiers (namespace, value, paper_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (namespace, value, paper_id, _utc_now()),
        )

    def find_paper_by_identifier(self, namespace: str, value: str) -> Paper | None:
        rows = self._connection.execute(
            f"""
            {_PAPER_SELECT}
            JOIN external_identifiers AS i ON i.paper_id = p.paper_id
            WHERE i.namespace = ? AND i.value = ?
            ORDER BY a.position
            """,
            (namespace, value),
        ).fetchall()
        return _paper(rows)

    def identifiers(self, paper_id: str) -> tuple[Identifier, ...]:
        rows = self._connection.execute(
            "SELECT namespace, value FROM external_identifiers WHERE paper_id = ?"
            " ORDER BY namespace, value",
            (paper_id,),
        ).fetchall()
        return tuple(Identifier(row["namespace"], row["value"]) for row in rows)

    def record_source(
        self,
        paper_id: str,
        provider: str,
        metadata: Mapping[str, Any],
        *,
        pdf_locations: Sequence[str] | None = None,
        url: str | None = None,
    ) -> None:
        """Create or update one provider's source record for a paper.

        An update replaces the metadata and refreshes the retrieval time.
        Omitted URLs and PDF locations keep their existing values. An explicit
        location sequence replaces the old one, and an empty sequence clears it.
        """
        if isinstance(pdf_locations, str):
            raise TypeError("pdf_locations must be a sequence of URLs, not a string")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO paper_sources (
                    paper_id, provider, metadata, url, retrieved_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (paper_id, provider) DO UPDATE SET
                    metadata = excluded.metadata,
                    url = COALESCE(excluded.url, paper_sources.url),
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    paper_id,
                    provider,
                    json.dumps(dict(metadata), sort_keys=True),
                    url,
                    _utc_now(),
                ),
            )
            if pdf_locations is not None:
                connection.execute(
                    "DELETE FROM paper_pdf_locations"
                    " WHERE paper_id = ? AND provider = ?",
                    (paper_id, provider),
                )
                connection.executemany(
                    "INSERT INTO paper_pdf_locations"
                    " (paper_id, provider, position, url) VALUES (?, ?, ?, ?)",
                    (
                        (paper_id, provider, position, location)
                        for position, location in enumerate(pdf_locations)
                    ),
                )

    def paper_sources(self, paper_id: str) -> tuple[PaperSource, ...]:
        rows = self._connection.execute(
            """
            SELECT
                s.paper_id,
                s.provider,
                s.metadata,
                s.url,
                s.retrieved_at,
                l.url AS pdf_url
            FROM paper_sources AS s
            LEFT JOIN paper_pdf_locations AS l
                ON l.paper_id = s.paper_id AND l.provider = s.provider
            WHERE s.paper_id = ?
            ORDER BY s.provider, l.position
            """,
            (paper_id,),
        ).fetchall()
        sources: list[PaperSource] = []
        for provider, group in groupby(rows, key=lambda row: row["provider"]):
            provider_rows = list(group)
            first = provider_rows[0]
            sources.append(
                PaperSource(
                    paper_id=first["paper_id"],
                    provider=provider,
                    metadata=json.loads(first["metadata"]),
                    pdf_locations=tuple(
                        row["pdf_url"]
                        for row in provider_rows
                        if row["pdf_url"] is not None
                    ),
                    url=first["url"],
                    retrieved_at=first["retrieved_at"],
                )
            )
        return tuple(sources)

    def add_citation(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        *,
        source: str | None = None,
        url: str | None = None,
    ) -> None:
        """Record a directed edge; repeating an edge keeps the first provenance."""
        self._connection.execute(
            """
            INSERT INTO citations (
                citing_paper_id, cited_paper_id, source, url, discovered_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (citing_paper_id, cited_paper_id, source, url, _utc_now()),
        )

    def citing_papers(self, paper_id: str) -> tuple[str, ...]:
        """Return the IDs of papers that cite the given paper."""
        rows = self._connection.execute(
            "SELECT citing_paper_id FROM citations WHERE cited_paper_id = ?"
            " ORDER BY citing_paper_id",
            (paper_id,),
        ).fetchall()
        return tuple(row["citing_paper_id"] for row in rows)

    def cited_papers(self, paper_id: str) -> tuple[str, ...]:
        """Return the IDs of papers that the given paper cites."""
        rows = self._connection.execute(
            "SELECT cited_paper_id FROM citations WHERE citing_paper_id = ?"
            " ORDER BY cited_paper_id",
            (paper_id,),
        ).fetchall()
        return tuple(row["cited_paper_id"] for row in rows)

    def promote(self, paper_id: str, level: EnrichmentLevel) -> EnrichmentLevel:
        """Raise a paper's enrichment level, never lowering it.

        Returns the level in effect afterwards, which equals the request only if
        the paper advanced. The caller remains responsible for acquiring inputs.
        """
        target = EnrichmentLevel(level)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT enrichment_level FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            if row is None:
                raise LibraryError(f"paper not found: {paper_id}")
            current = EnrichmentLevel(row["enrichment_level"])
            if _level_rank(target) <= _level_rank(current):
                return current
            connection.execute(
                "UPDATE papers SET enrichment_level = ?, updated_at = ?"
                " WHERE paper_id = ?",
                (target.value, _utc_now(), paper_id),
            )
            return target

    def add_artefact_bytes(
        self,
        paper_id: str,
        data: bytes,
        *,
        role: ArtefactRole,
        media_type: str = DEFAULT_MEDIA_TYPE,
        source: str | None = None,
        url: str | None = None,
    ) -> str:
        """Store bytes and associate them with a paper.

        A paper holds at most one PDF and one extracted-text artefact, so
        supplying different bytes for an occupied main role raises
        ``sqlite3.IntegrityError``. Supplementary artefacts are unlimited.
        The completed file and its directory entry are synced before any
        database reference is committed. Repeating an association keeps its
        first source and URL. An existing blob keeps its media type unless the
        registered default is upgraded to a specific type; conflicting
        specific types raise ``LibraryError``.
        """
        role = ArtefactRole(role)
        return self._record_artefact(
            paper_id,
            self._artefacts.put_bytes(data),
            role,
            media_type,
            source,
            url,
        )

    def add_artefact_file(
        self,
        paper_id: str,
        path: Path,
        *,
        role: ArtefactRole,
        media_type: str = DEFAULT_MEDIA_TYPE,
        source: str | None = None,
        url: str | None = None,
    ) -> str:
        """Import a file by content, leaving the given file unchanged.

        Provenance, role, media-type and durability rules match
        ``add_artefact_bytes``.
        """
        role = ArtefactRole(role)
        return self._record_artefact(
            paper_id,
            self._artefacts.put_file(path),
            role,
            media_type,
            source,
            url,
        )

    def get_artefact(self, artefact_id: str) -> Artefact | None:
        row = self._connection.execute(
            "SELECT artefact_id, byte_size, media_type FROM artefacts"
            " WHERE artefact_id = ?",
            (artefact_id,),
        ).fetchone()
        if row is None:
            return None
        return Artefact(
            artefact_id=row["artefact_id"],
            byte_size=row["byte_size"],
            media_type=row["media_type"],
            path=self._artefacts.path(row["artefact_id"]),
        )

    def artefact_path(self, artefact_id: str) -> Path:
        """Return the managed path, rejecting anything that is not a digest."""
        return self._artefacts.path(artefact_id)

    def paper_artefacts(self, paper_id: str) -> tuple[ArtefactLink, ...]:
        rows = self._connection.execute(
            "SELECT a.artefact_id, a.byte_size, a.media_type, pa.role,"
            " pa.source, pa.url"
            " FROM paper_artefacts AS pa JOIN artefacts AS a USING (artefact_id)"
            " WHERE pa.paper_id = ? ORDER BY a.artefact_id, pa.role",
            (paper_id,),
        ).fetchall()
        return tuple(self._artefact_link(row) for row in rows)

    def _artefact_link(self, row: sqlite3.Row) -> ArtefactLink:
        return ArtefactLink(
            artefact_id=row["artefact_id"],
            role=ArtefactRole(row["role"]),
            media_type=row["media_type"],
            byte_size=row["byte_size"],
            path=self._artefacts.path(row["artefact_id"]),
            source=row["source"],
            url=row["url"],
        )

    def _record_artefact(
        self,
        paper_id: str,
        blob: Blob,
        role: ArtefactRole,
        media_type: str,
        source: str | None,
        url: str | None,
    ) -> str:
        now = _utc_now()
        with self._transaction() as connection:
            registered = connection.execute(
                "SELECT media_type FROM artefacts WHERE artefact_id = ?",
                (blob.artefact_id,),
            ).fetchone()
            if registered is None:
                connection.execute(
                    "INSERT INTO artefacts (artefact_id, byte_size, media_type,"
                    " created_at) VALUES (?, ?, ?, ?)",
                    (blob.artefact_id, blob.byte_size, media_type, now),
                )
            connection.execute(
                "INSERT INTO paper_artefacts (paper_id, artefact_id, role,"
                " created_at, source, url) VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (paper_id, artefact_id, role) DO NOTHING",
                (paper_id, blob.artefact_id, role.value, now, source, url),
            )
            if registered is not None:
                _reconcile_media_type(
                    connection, blob.artefact_id, registered["media_type"], media_type
                )
        return blob.artefact_id

    def _prepare_schema(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version != 0:
            raise LibraryError(
                f"library schema version {version} is not supported: "
                f"{self.database_path}"
            )
        with self._transaction() as connection:
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise


_PAPER_SELECT = """
    SELECT
        p.paper_id,
        p.title,
        p.abstract,
        p.published,
        p.enrichment_level,
        a.name
    FROM papers AS p
    LEFT JOIN paper_authors AS a ON a.paper_id = p.paper_id
"""


def _paper(rows: Sequence[sqlite3.Row]) -> Paper | None:
    """Build one paper from its rows, which pair metadata with ordered authors."""
    if not rows:
        return None
    first = rows[0]
    return Paper(
        paper_id=first["paper_id"],
        title=first["title"],
        abstract=first["abstract"],
        authors=tuple(row["name"] for row in rows if row["name"] is not None),
        published=first["published"],
        enrichment_level=EnrichmentLevel(first["enrichment_level"]),
    )


def _author_names(authors: Sequence[str]) -> list[str]:
    if isinstance(authors, str):
        raise TypeError("authors must be a sequence of names, not a string")
    return list(authors)


def _replace_authors(
    connection: sqlite3.Connection, paper_id: str, names: Sequence[str]
) -> None:
    connection.execute("DELETE FROM paper_authors WHERE paper_id = ?", (paper_id,))
    connection.executemany(
        "INSERT INTO paper_authors (paper_id, position, name) VALUES (?, ?, ?)",
        ((paper_id, position, name) for position, name in enumerate(names)),
    )


def _level_rank(level: EnrichmentLevel) -> int:
    return tuple(EnrichmentLevel).index(level)


def _reconcile_media_type(
    connection: sqlite3.Connection,
    artefact_id: str,
    registered: str,
    supplied: str,
) -> None:
    if supplied == registered or supplied == DEFAULT_MEDIA_TYPE:
        return
    if registered == DEFAULT_MEDIA_TYPE:
        connection.execute(
            "UPDATE artefacts SET media_type = ? WHERE artefact_id = ?",
            (supplied, artefact_id),
        )
        return
    raise LibraryError(f"artefact {artefact_id} is already registered as {registered}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _schema() -> tuple[str, ...]:
    levels = ", ".join(f"'{level.value}'" for level in EnrichmentLevel)
    roles = ", ".join(f"'{role.value}'" for role in ArtefactRole)
    main_roles = ", ".join(f"'{role.value}'" for role in _MAIN_ROLES)
    return (
        f"""
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            published TEXT,
            enrichment_level TEXT NOT NULL
                DEFAULT '{EnrichmentLevel.DISCOVERY.value}'
                CHECK (enrichment_level IN ({levels})),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE paper_authors (
            paper_id TEXT NOT NULL REFERENCES papers (paper_id)
                ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK (position >= 0),
            name TEXT NOT NULL,
            PRIMARY KEY (paper_id, position)
        )
        """,
        """
        CREATE INDEX paper_authors_name ON paper_authors (name)
        """,
        """
        CREATE TABLE external_identifiers (
            namespace TEXT NOT NULL,
            value TEXT NOT NULL,
            paper_id TEXT NOT NULL REFERENCES papers (paper_id)
                ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            PRIMARY KEY (namespace, value)
        )
        """,
        """
        CREATE INDEX external_identifiers_paper
            ON external_identifiers (paper_id)
        """,
        """
        CREATE TABLE paper_sources (
            paper_id TEXT NOT NULL REFERENCES papers (paper_id)
                ON DELETE CASCADE,
            provider TEXT NOT NULL,
            metadata TEXT NOT NULL CHECK (json_valid(metadata)),
            url TEXT,
            retrieved_at TEXT NOT NULL,
            PRIMARY KEY (paper_id, provider)
        )
        """,
        """
        CREATE TABLE paper_pdf_locations (
            paper_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            position INTEGER NOT NULL CHECK (position >= 0),
            url TEXT NOT NULL,
            PRIMARY KEY (paper_id, provider, position),
            FOREIGN KEY (paper_id, provider)
                REFERENCES paper_sources (paper_id, provider)
                ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE citations (
            citing_paper_id TEXT NOT NULL REFERENCES papers (paper_id)
                ON DELETE CASCADE,
            cited_paper_id TEXT NOT NULL REFERENCES papers (paper_id)
                ON DELETE CASCADE,
            source TEXT,
            url TEXT,
            discovered_at TEXT NOT NULL,
            PRIMARY KEY (citing_paper_id, cited_paper_id),
            CHECK (citing_paper_id <> cited_paper_id)
        )
        """,
        """
        CREATE INDEX citations_cited ON citations (cited_paper_id)
        """,
        """
        CREATE TABLE artefacts (
            artefact_id TEXT PRIMARY KEY,
            byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
            media_type TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE paper_artefacts (
            paper_id TEXT NOT NULL REFERENCES papers (paper_id)
                ON DELETE CASCADE,
            artefact_id TEXT NOT NULL REFERENCES artefacts (artefact_id),
            role TEXT NOT NULL CHECK (role IN ({roles})),
            created_at TEXT NOT NULL,
            source TEXT,
            url TEXT,
            PRIMARY KEY (paper_id, artefact_id, role)
        )
        """,
        f"""
        CREATE UNIQUE INDEX paper_artefacts_single_main_role
            ON paper_artefacts (paper_id, role)
            WHERE role IN ({main_roles})
        """,
    )


_SCHEMA = _schema()
