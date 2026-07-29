# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/janus_ts/`. Keep chemistry representation,
parsing, metrics, training, evaluation, and workflow logic in their existing
focused modules rather than expanding `cli.py`. Tests mirror those modules in
`tests/test_*.py`. Experiment settings are under `configs/`; the locked
Transition1x contract is documented in `docs/experiment-contract.md`.
Operational helpers belong in `scripts/`, while source patches belong in
`patches/`. Treat `artifacts/` as generated, content-addressed run state; do not
commit or hand-edit it.

## Build, Test, and Development Commands

- `uv sync --frozen` installs the Python 3.11 environment from `uv.lock`.
- `uv run janus-ts --help` lists the CLI and subcommands.
- `uv run pytest` runs the complete test suite.
- `uv run pytest tests/test_metrics.py -q` runs a focused test module.
- `uv run ruff check .` checks imports, correctness, and style.
- `uv run ruff format --check .` verifies formatting.
- `uv run janus-ts data audit --config configs/transition1x.yaml` validates
  prepared data without starting formal training.
- `uv run janus-ts train smoke --config configs/transition1x.yaml` runs CPU and
  GPU readiness gates. It is resource-intensive and requires the gu30 GPU lock.

Do not start `train run` or formal evaluation casually: both operate on the
locked experiment and expensive 27B-model artifacts.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations for public interfaces, and a
100-character line limit. Ruff enforces `E`, `F`, `I`, `UP`, `B`, and `SIM`
rules. Use `snake_case` for functions and modules, `PascalCase` for classes,
and `UPPER_SNAKE_CASE` for constants. Prefer small deterministic functions and
explicit provenance fields for experiment-critical behavior.

## Testing Guidelines

Pytest discovers `tests/test_*.py`; name tests `test_<observable_behavior>`.
Add a regression test for every bug fix and cover failure paths for gates,
checkpointing, parsing, and state transitions. No numeric coverage threshold is
configured, but changed behavior must be exercised. Run focused tests while
iterating, then the full suite and Ruff before submission.

## Commit & Pull Request Guidelines

History uses short, imperative subjects, optionally with prefixes such as
`fix:`. Keep each commit to one logical change. Pull requests should explain
the motivation, affected experiment contract or configuration, validation
commands and results, and any migration or checkpoint compatibility impact.
Do not include model weights, datasets, generated run outputs, credentials, or
machine-specific paths.

## Operational Safety

Raw chemistry datasets and the historical Chemformer checkout are read-only
inputs. Respect project GPU locks, never terminate unrelated processes, and
preserve fingerprinted checkpoints and completion markers during recovery.
