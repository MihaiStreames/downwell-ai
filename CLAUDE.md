# Python Project Rules

<!-- Generated from pydevtools.com, the Python Developer Tooling Handbook -->
<!-- Last verified against: uv 0.11.8, ruff 0.15.12, ty 0.0.35, pytest 9.0.3 -->
<!-- Full explanations: https://pydevtools.com/handbook/explanation/modern-python-project-setup-guide-for-ai-assistants/ -->

## Package management

This project uses `uv`. Do not use `pip`, `pip-tools`, `poetry`, or `conda`.

- Add runtime dependency: `uv add <package>` (writes to `[project.dependencies]`)
- Add dev dependency: `uv add --dev <package>` (writes to `[dependency-groups]` per PEP 735)
- Remove dependency: `uv remove <package>`
- Sync environment from lockfile: `uv sync`
- Regenerate lockfile from constraints: `uv lock`
- Upgrade locked versions: `uv lock --upgrade`
- Commit `uv.lock` to version control

### Platform-specific dependencies

There are none right now. If one is needed, add a `linux` or `windows` dependency group (`uv add --group windows <package>`) and sync with `uv sync --group windows`, not inline `sys_platform` markers.

Do not add `; sys_platform == ...` markers to `[project.dependencies]`. Put them in the `linux` or `windows` group instead.

## Running code

Always use `uv run` to execute Python code and tools. Never call `python`, `pytest`, `ruff`, or other tools directly. They may not resolve to the project's virtual environment.

- Run a script: `uv run python script.py`
- Run a module: `uv run python -m module_name`
- Run a tool: `uv run pytest`, `uv run ruff check .`
- One-off tool (not a project dependency): `uvx <tool>`

## Testing

- Framework: `pytest`
- Run tests: `uv run pytest`
- Smoke tests require the game running and are skipped by default (`-m not smoke`)
- Test files go in `tests/` at the project root
- Test file naming: `test_*.py`
- No `__init__.py` needed in `tests/`

## Linting and formatting

- Tool: `ruff` (handles both linting and formatting)
- Lint: `uv run ruff check .`
- Lint and auto-fix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Check formatting: `uv run ruff format --check .`
- Sort modules: `uv run ssort downwell_ai/ scripts/ tests/`
- Line length: 160 characters
- All `ruff` configuration lives in `pyproject.toml` under `[tool.ruff]`

## TOML formatting

- Tool: `taplo`
- Format: `uvx taplo fmt` (every `.toml` except `pyproject.toml`, which `pyproject-fmt` owns)
- Configuration lives in `.taplo.toml` at the repo root, the one exception to keeping tool config in `pyproject.toml`
- Pre-commit hook runs `taplo` automatically on staged `.toml` files

## Type checking

- Tool: `ty`
- Run: `uv run ty check`
- Configuration lives in `pyproject.toml` under `[tool.ty]`

## Tool configuration

All tool configuration lives in `pyproject.toml`. Do not create separate `ruff.toml`, `ty.toml`, or `pytest.ini` files. `.taplo.toml` is the one exception.

In `[tool.uv]`, always put `[[tool.uv.index]]` above `[tool.uv.sources]` (define names before referencing them).

## Pre-commit hooks

- Tool: `prek`
- Install hooks: `uvx prek install`
- Do not install `prek` with `pip`. Use `uvx`.

## Security

- Scan dependencies for vulnerabilities: `uv audit`
- GitHub Actions pins: `scripts/pin-actions.sh` (updates SHAs via `pinact` + audits via `zizmor`)
- Check only (for CI): `scripts/pin-actions.sh --check`
- Requires `pinact` and `zizmor` installed on the system

## CI

Three separate jobs in `.github/workflows/ci.yml`:

- `lint` -- `ruff check`, `ruff format`, `ty check`
- `test` -- `pytest` (non-smoke)
- `audit` -- `uv audit`

All GitHub Actions are SHA-pinned with trailing tag comments. Use `scripts/pin-actions.sh` to update them.

## Cross-platform

win32 and linux only. No macOS.

- Always `if sys.platform == "win32"` / `if sys.platform == "linux"` -- never `else`
- Platform-specific imports at module level inside their guard
- Platform-specific dependencies go in `[dependency-groups]` (`linux`, `windows`), not inline `sys_platform` markers

## Game patch and snapshot

- `downwell_ai/env/memory/layout.py` is the single source of truth for the snapshot buffer; `scripts/gen_export.py` generates the GML from it and `_state.parse_snapshot` reads with it
- `scripts/generated/` is build output and gitignored; `downwell_ai/env/memory/generated/game_ptrs.py` is committed and must be regenerated with `scripts/gen_ptrs.py` against the live patched game after every re-inject
- Never edit generated files by hand; change the layout or the generator instead
- Object ids, stomp rules, hazards and pickups live in `downwell_ai/env/downwell.toml`; stomp rules mirror each enemy's player-collision handler in the game code

## Exception hierarchy

```console
DownwellError (base)
├── FieldResolveError -- all pointer chains for a game field failed
├── StateLayoutError  -- snapshot record count impossible for layout.py (stale patch)
└── GameTimeoutError  -- game stopped advancing / player never spawned within the deadline
```

- `None` return for transient failures: game not running
- `FieldResolveError` for invariant violations in the hot path -- callers must handle, training stops
- Third-party exceptions (`xlib_error`, `ProcessNotFoundError`, etc.) never escape their module -- catch at the boundary

## Logging levels

- `trace` -- hot-path per-frame noise (pointer reads, memory reads)
- `debug` -- lifecycle events (attach, close, init)
- `warning` -- recoverable failures that self-resolve (game slow to release, will retry)
- `error` -- unexpected failures that degrade behavior but don't stop execution
- `success` -- meaningful milestones (first successful process attach)
- Never log the same failure at two layers -- log once at the site of the exception

## Training loop contract

- `attach() -> None` -- process not running; retry
- `FieldResolveError` / `StateLayoutError` raised from `read()` -- patch missing or stale; catch at loop level, clean shutdown
- `GameTimeoutError` raised from `step()` / `reset()` -- game stopped answering in lockstep; catch at loop level, clean shutdown

## What NOT to do

- Do not create or activate virtual environments manually. `uv` manages `.venv/` automatically.
- Do not install packages globally or with `pip install`.
- Do not create `requirements.txt` for dependency management. Use `pyproject.toml` and `uv.lock`.
- Do not run `python setup.py` commands.
- Do not add dependencies to `pyproject.toml` by hand. Use `uv add`.
- Do not create separate config files for `ruff`, `ty`, or `pytest`. Everything goes in `pyproject.toml`, except taplo's `.taplo.toml`.
- If you must edit `pyproject.toml` directly, write dev dependencies under `[dependency-groups]` (PEP 735), not the legacy `[tool.uv.dev-dependencies]` table.
