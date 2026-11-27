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

## Model endpoints

The model client supports text-only, non-streaming OpenAI Chat Completions, Anthropic Messages and Gemini `generateContent`.
The CLI does not call it yet.
The default configuration is equivalent to:

```toml
[flash_model]
endpoint = "opencode"
model_id = "deepseek-v4.1-flash"

[evaluator_model]
endpoint = "opencode"
model_id = "mimo-v2.6-pro"

[endpoints.opencode]
api = "openai"
base_url = "https://opencode.ai/inference/go/openai/v1"
credential = "opencode"
api_key_env = "OPENCODE_API_KEY"
quota_scope = "opencode-go"

[endpoints.opencode.headers]
x-opencode-session = "ses_{conversation_id}"
```

The OpenCode defaults require a [Go subscription and a key with model request access](https://opencode.ai/docs/console/go/).
DeepSeek V4.1 Flash also requires **Global** under **Settings → Privacy → Regions** in the OpenCode workspace.
Researcher does not change that setting.

Put API keys in a separate `credentials.toml` next to the selected `config.toml`.
Without `--config`, the default location is `$XDG_CONFIG_HOME/researcher/credentials.toml`, even if `config.toml` does not exist.
For the OpenCode defaults, add this entry to `credentials.toml`:

```toml
opencode = "your-api-key"
```

On POSIX, create the file with owner-only permissions, such as `0600`, before adding the key.
The file contains plaintext keys; keep it out of version control.
Several endpoint profiles can refer to the same named credential.
If the optional `api_key_env` variable is set, it overrides the file key.
For automation, a profile can specify `api_key_env` without a named credential.

To use a different service, configure an endpoint and point one or both model roles at it.
Specifying `endpoints` replaces the default endpoint map, so include every endpoint that either role references.
The API format determines the path appended to `base_url` and the default authentication header:

| `api` | Native `base_url` | Appended path | Default `auth` |
| --- | --- | --- | --- |
| `openai` | `https://api.openai.com/v1` | `/chat/completions` | `bearer` |
| `anthropic` | `https://api.anthropic.com/v1` | `/messages` | `x-api-key` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta` | `/models/<model_id>:generateContent` | `x-goog-api-key` |

Local OpenAI-compatible servers such as llama.cpp can use `base_url = "http://127.0.0.1:8080/v1"`.
Use HTTPS for remote services so credentials and prompts are encrypted in transit.
Use `auth = "bearer"` where a gateway requires bearer authentication with a different API format.
Set `max_output_tokens` in a model role when a cap is needed; Anthropic Messages requires it.
Chat Completions sends an explicit cap as `max_completion_tokens`; check that your endpoint supports that field before setting it.
Anthropic Messages defaults to `anthropic-version: 2026-10-21`, which an endpoint can override in its headers.
Custom headers support `{conversation_id}`, which the client generates afresh for each model call.
Use the same `quota_scope` value on endpoint profiles that share a documented quota.
Automatic retries do not replay inference requests.

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

### Cached decisions

Use `record_question()` to store a loaded brief by its content hash and `question_text()` to retrieve it.
`record_screening()` and `record_assessment()` retain the first completed result for each brief and paper, including rejections.
`get_screening()` and `get_assessment()` return `None` when no result is stored.
Only approved assessments retain an explanation and can include a suggested PDF filename.
`decision_counts()` reports how many distinct questions checked and approved a paper at each of the two stages.

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
