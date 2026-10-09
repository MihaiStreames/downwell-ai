# Contributing

## Setup

```sh
uv sync
uvx prek install  # pre-commit hooks
```

Always run tools through `uv run`. Add dependencies with `uv add` (or `uv add --dev`), never by editing `pyproject.toml` by hand. Platform-only packages go in the `windows` or `linux` group.

## Checks

```sh
uv run ruff check --fix . && uv run ruff format .
uv run python -m ssort downwell_ai/ scripts/ tests/
uv run ty check
uvx taplo fmt  # every toml except pyproject.toml
```

All tool configuration lives in `pyproject.toml`, except taplo's `.taplo.toml`.

## Tests

```sh
uv run python -m pytest              # unit tests, no game needed
uv run python -m pytest -m smoke -s  # needs the patched game running inside a run
```

- Unit tests drive the environment with a fake game, so they never press keys or read memory
- Order tests in a file from simplest to most involved
- Test behavior and edge cases, skip anything that only exercises a library

## Changing what the game exports

The patch and the reader share one layout, `downwell_ai/env/memory/layout.py`: header fields, record fields, family flags and the objects listed in `downwell_ai/env/downwell.toml`. After changing either:

```sh
uv run python scripts/gen_export.py  # rewrites scripts/generated/ (gitignored)
```

Then re-inject `scripts/generated/InjectDQNExport.csx` into a clean `data.win`, start a run, and rerun `scripts/gen_ptrs.py` so `generated/game_ptrs.py` finds the patch's globals. Old checkpoints stop loading whenever the observation shape changes.

## Code style

- win32 and linux only: `if sys.platform == "win32"` / `if sys.platform == "linux"`, never `else`
- Import third-party classes by name (`from msgspec import Struct`) and call third-party functions through their module (`msgspec.json.decode`)
- Third-party exceptions never leave their module. Return `None` for transient failures, raise a `DownwellError` subclass for invariant violations
- Log with `loguru`: `trace` per frame, `debug` lifecycle, `warning` recoverable, `error` degraded, and never log one failure twice

## Commits

```
type(scope): short imperative summary

- what changed
- what changed
```

Imperative mood, lowercase after the colon, under 70 characters, no trailing dot.
