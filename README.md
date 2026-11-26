# Researcher

A Python CLI for ML literature discovery with a persistent local paper library.
The CLI loads and validates research briefs and configuration; it does not perform research yet.
A separate library API provides persistent paper records and managed artefacts.
The system design is in [design.org](design.org).

## Usage

```sh
uv run researcher question.md
uv run researcher ./my-topic/ --max-papers 40 --time-limit 2h
uv run researcher question.md --config ./config.toml
```

Directory input reads its `question.md`.
Question IDs are SHA-256 hashes of the exact UTF-8 file bytes, independent of the file path and configuration overrides.

Configuration is read from `$XDG_CONFIG_HOME/researcher/config.toml` only when `XDG_CONFIG_HOME` is set and non-empty.
There is no home-directory fallback.
An unset configuration directory or missing default file uses built-in defaults; an explicitly supplied `--config` file must exist.
CLI options override TOML settings, which override built-in defaults:

```toml
max_papers = 25
time_limit = "1h"
```

TOML is loaded into a typed `msgspec` schema, which rejects unknown settings and invalid field types.
`max_papers` must be a positive integer.
`time_limit` accepts a positive number followed by `s`, `m` or `h`, such as `45m` or `0.5h`.

## Library API

`Library()` uses `platformdirs` to select the per-user data directory containing `library.sqlite3` and `artefacts/<sha256>`.
On Linux, this is `$XDG_DATA_HOME/researcher`, or the standard XDG default `~/.local/share/researcher` when no valid data directory is set.
This data-directory fallback does not change configuration-file loading above.
An explicit `Library(path)` still selects a custom directory for tests or deliberate overrides.
The CLI does not open the library yet.

```python
from researcher.library import ArtefactRole, Library

with Library() as library:
    paper_id = library.add_paper(title="Example paper")
    library.record_source(
        paper_id,
        provider="arxiv",
        metadata={"version": "v2"},
        pdf_locations=["https://arxiv.example/pdf/examplev2"],
        url="https://arxiv.example/abs/example",
    )
    artefact_id = library.add_artefact_bytes(
        paper_id,
        b"Extracted paper text",
        role=ArtefactRole.EXTRACTED_TEXT,
        media_type="text/plain",
    )
    text = library.artefact_path(artefact_id).read_text(encoding="utf-8")
```

The library stores ordered author names, identifiers, paper sources and citation edges, with explicit monotonic enrichment promotion.
Author names use per-paper rows indexed for name-based lookup, without shared person identities.
`Paper.authors` remains an ordered tuple in returned values.
Each `PaperSource` belongs directly to a paper and combines provider metadata, explicit candidate PDF locations, a source record URL and retrieval time.
SQLite stores PDF locations as ordered child rows of the source; provider-specific metadata remains JSON because its contents are not used for SQL filtering.
Metadata-only sources have no location rows and return an empty location tuple.
Use `record_source` to write a source and `paper_sources(paper_id)` to read its records.
Source updates preserve PDF locations when the argument is omitted; an explicit empty list clears them.
Alternative PDF locations remain recorded even when they are not downloaded.
Compact metadata stays in SQLite; PDFs, extracted text and full OpenReview discussion snapshots use managed artefacts.
Artefacts belong directly to papers, with at most one main PDF and one extracted-text artefact per paper, plus multiple supplementary artefacts.
Acquisition selects one usable main PDF and uses other PDF locations only as fallbacks.
Identical artefact content shares one file through separate paper associations; importing a file does not modify the original.
Use `library.add_artefact_file()` to import a local file without changing the original.
The store manages temporary files internally before publishing artefacts under their content hashes.
The Python dataclasses are query results, while primary and foreign keys are defined in the SQLite schema.

## Development

Use system Python 3.15:

```sh
uv sync --python /usr/bin/python3.15
uv run researcher --help
uv run researcher --version
```

The package also supports `uv run python -m researcher --help`.

Run checks:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```
