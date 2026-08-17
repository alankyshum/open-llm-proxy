# Contributing

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

Run checks with:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

## Pull requests

Keep changes focused, add or update tests where appropriate, and describe
user-visible or security-relevant effects. Pull requests must pass CI and
receive review before merging.

## Supported platforms

macOS is the primary platform; Linux is supported. The launchd configuration
is macOS-only.
