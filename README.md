# Researcher

A Python CLI for ML literature discovery with a persistent local paper library.
The current implementation loads and validates research briefs and configuration; it does not perform research yet.
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
