# Researcher

A Python CLI for ML literature discovery with a persistent local paper library.
The current implementation is the project scaffold, not the research workflow.
The system design is in [design.org](design.org).

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
