# wbt Python 包

`wbt` Rust 回测引擎的 Python 绑定项目。

[English](README.md)

## 目录结构

当前目录就是独立的 Python 子工程：

```text
python/
|-- pyproject.toml
|-- README.md
|-- docs/
|-- tests/
`-- wbt/
```

Rust crate 仍然位于上一层的 `../Cargo.toml`，通过 maturin 进行构建。

## 开发环境

要求：

- Rust 工具链
- Python 3.10+
- `uv`

命令：

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
```

## 代码质量检查

在 `python/` 目录执行：

```bash
uv run pytest -v
uv run ruff format --check .
uv run ruff check . --no-fix
uv run basedpyright
```

## 快速开始

```python
import pandas as pd
from wbt import WeightBacktest

dfw = pd.DataFrame(
    {
        "dt": ["2024-01-02 09:01:00", "2024-01-02 09:02:00"],
        "symbol": ["AAPL", "AAPL"],
        "weight": [0.5, 0.0],
        "price": [185.0, 186.0],
    }
)

wb = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts")
print(wb.stats)
```

## License

[MIT](../LICENSE)
