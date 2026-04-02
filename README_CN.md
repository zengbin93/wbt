# wbt

持仓权重回测引擎，底层为 Rust，Python 项目独立放在 `python/` 目录。

[English](README.md)

## 仓库结构

- Rust crate：仓库根目录
- Python 包：`python/`

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

Python 包仍然使用 `import wbt`，但打包、测试、文档和代码质量配置都集中在 `python/` 下。

- Python 使用说明：`python/README.md`
- 中文说明：`python/README_CN.md`

快速开始：

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
uv run pytest -v
```

## Rust

仓库根目录继续作为 Rust crate 的入口。

```bash
cargo test
```

```toml
[dependencies]
wbt = "0.1"
```

## 开发约定

- Rust 相关命令在仓库根目录执行
- Python 相关命令在 `python/` 目录执行
- CI 会同时检查这两部分

## License

[MIT](LICENSE)
