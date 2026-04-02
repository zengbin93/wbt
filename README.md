# wbt

Position-weighted backtesting engine with a Rust core and a Python package.

[中文说明](README_CN.md)

## Repository Layout

- Rust crate: repository root
- Python package: `python/`

```text
wbt/
|-- Cargo.toml
|-- src/
`-- python/
    |-- pyproject.toml
    |-- README.md
    |-- tests/
    `-- wbt/
```

## Python

The Python package keeps the import path `import wbt`, but its packaging, docs, tests, and quality tooling now live under `python/`.

- User and contributor guide: `python/README.md`
- Chinese guide: `python/README_CN.md`

Quick start:

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
uv run pytest -v
```

## Rust

The repository root remains the Rust crate entrypoint.

```bash
cargo test
```

```toml
[dependencies]
wbt = "0.1"
```

## Development

- Rust checks run from the repository root.
- Python checks run from `python/`.
- CI validates both layers.

## License

[MIT](LICENSE)
