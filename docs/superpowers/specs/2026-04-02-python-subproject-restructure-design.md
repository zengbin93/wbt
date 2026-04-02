# Python Subproject Restructure Design

**Date:** 2026-04-02

## Goal

Restructure the repository so that all Python-facing assets live under `python/` as a self-contained subproject, while the repository root remains the Rust crate and top-level project entrypoint.

## Current State

- The Rust crate already lives at the repository root via `Cargo.toml` and `src/`.
- The Python package source already lives at `python/wbt/`.
- Python tests, docs, and README files are still split across the repository root.
- `pyproject.toml` currently sits at the repository root, which blurs the Python and Rust boundaries.
- Existing Python testing assets include machine-specific paths and are not yet shaped as portable project tests.

## Design Decision

Adopt a two-entrypoint repository layout:

- Repository root: Rust crate, shared license, and top-level navigation docs.
- `python/`: standalone Python project containing packaging metadata, docs, tests, and the `wbt` Python package.

This keeps the public Python import path unchanged while making Python tooling operate against a focused subproject.

## Target Structure

```text
wbt/
|-- Cargo.toml
|-- README.md
|-- README_CN.md
|-- LICENSE
|-- src/
|-- python/
|   |-- pyproject.toml
|   |-- README.md
|   |-- README_CN.md
|   |-- docs/
|   |-- tests/
|   `-- wbt/
`-- docs/
    `-- superpowers/
        |-- specs/
        `-- plans/
```

## Repository Responsibilities

### Root

- Keep `Cargo.toml`, Rust source, and Rust packaging metadata at the root.
- Keep a lightweight repository `README.md` that explains the split layout and points users to Rust and Python entrypoints.
- Keep a lightweight `README_CN.md` aligned with the repository overview.

### Python Subproject

- Move Python packaging metadata from the root into `python/pyproject.toml`.
- Move Python docs and usage guides into `python/docs/`.
- Move Python tests into `python/tests/`.
- Move detailed Python install and usage instructions into `python/README.md` and `python/README_CN.md`.

## Build and Packaging Approach

The Python subproject will continue to build the extension module with `maturin`, but it will reference the root Rust crate instead of duplicating Rust metadata.

Expected setup:

- `python/pyproject.toml` becomes the Python packaging source of truth.
- `tool.maturin.python-source = "."` so the package root inside the Python subproject is self-contained.
- `tool.maturin.module-name = "wbt._wbt"` remains unchanged to preserve imports.
- `tool.maturin.manifest-path = "../Cargo.toml"` points packaging back to the root Rust crate.

This allows Python development to happen from `python/` with `uv`, while keeping the Rust crate canonical at the root.

## Python Tooling Design

The Python subproject should become independently operable with `uv`:

- `uv sync --extra dev`
- `uv run maturin develop --release`
- `uv run pytest`
- `uv run ruff format .`
- `uv run ruff check . --fix`
- `uv run basedpyright`

`python/pyproject.toml` will include:

- Core runtime dependencies already needed by the package.
- A `dev` optional dependency group for `pytest`, `ruff`, `basedpyright`, and `maturin`.
- Ruff configuration following the `python-code-quality` guidance.
- Basedpyright configuration with strict type checking where practical.
- Pytest configuration rooted in the Python subproject.

## Test Migration Strategy

The current root-level `tests/test_alignment.py` is not portable in its present form because it depends on local absolute paths and external data files.

The migration should:

- Move Python tests under `python/tests/`.
- Convert portable package verification into proper pytest tests.
- Preserve the existing alignment script only if it still provides value, but place it in a clearly non-default location such as `python/tests/manual/` or `python/scripts/` and mark it as non-CI.
- Add at least one portable smoke test for the public Python API so the new Python project has a reliable baseline check.

## Documentation Strategy

### Root README files

- Keep them concise.
- Explain that the repository contains:
  - a Rust crate at the root
  - a Python package in `python/`
- Link to the detailed Python docs in `python/README.md`.

### Python README files

- Hold detailed install, development, and usage instructions for Python users.
- Document `uv` workflow and code quality commands.
- Reflect the new project layout so contributors know where to run Python commands.

### Python docs

- Move existing Python-specific design and usage material into `python/docs/`.
- Fix obvious naming issues such as `docs/desgin.md` if it is retained.

## Compatibility Expectations

- `import wbt` remains unchanged.
- The Python module name `wbt._wbt` remains unchanged.
- The root Rust crate metadata remains valid for Rust users.
- Existing repository links may need updates where they currently assume Python docs are at the root.

## Risks and Mitigations

### Risk: maturin path configuration breaks builds

Mitigation:

- Explicitly configure `manifest-path` in `python/pyproject.toml`.
- Verify with a local `uv`-based development install.

### Risk: moved documentation breaks package metadata

Mitigation:

- Update `project.readme` to the new Python README path inside the subproject.
- Keep root README files as stable navigation entrypoints.

### Risk: tests become weaker after removing environment-specific checks

Mitigation:

- Replace machine-local checks with portable smoke tests.
- Preserve heavyweight comparison scripts as manual verification tools when needed.

### Risk: repository contributors run commands from the wrong directory

Mitigation:

- Make root README and Python README explicit about command locations.
- Keep commands short and directory-specific.

## Implementation Boundaries

This restructuring should include:

- Python project layout cleanup
- Python docs and test relocation
- `uv` environment setup metadata
- Ruff and basedpyright configuration
- Verification commands for Python tooling

This restructuring should not include:

- Moving the Rust crate into a separate `rust/` directory
- Rewriting Rust internals unrelated to packaging
- Broad behavioral changes to the `wbt` Python API

## Validation Plan

Implementation will be considered successful when:

- The Python project can be set up from `python/` with `uv`.
- The extension can be built from the Python subproject against the root `Cargo.toml`.
- Python tests run from `python/`.
- Ruff and basedpyright run successfully from `python/`.
- Root documentation clearly explains the split project layout.
