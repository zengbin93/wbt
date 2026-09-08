# D03：结果快照边界

Python `WeightBacktest.digits/fee_rate/weight_type/yearly_days` 为只读有效配置，直接来自 Rust。weight_type 沿用 #43 的严格校验，仅接受小写 `ts` 或 `cs`，非法值抛出 ValueError，不回退、不自动去除空白。合法 weight_type 下，`fee_rate=None` 的有效默认值仍为 `0.0002`。`symbols` 和 pandas 入口的 `dfw` 返回独立副本；结果表继续通过 Arrow 解码为独立 DataFrame。

`BacktestResult` 公共字段不可重新赋值或删除；嵌套字典为只读 Mapping、列表为 tuple、数组使用不可写底层缓冲区，懒计算字段同样冻结。`to_dict()` 与 JSON/MessagePack 仍返回普通、可独立修改的数据。需要编辑绘图数据时复制数组或使用导出字典。私有下划线属性不属于公共 API。

参数修改请重新构造 `WeightBacktest(data, ...)` 并调用 `to_result()`；不新增原地更新 API，从而保留旧快照。Rust 字段改为私有，使用同名 getter；缓存 getter 返回 `&DataFrame`，需要编辑的调用方应 `.clone()`。Rust 原有显式 `backtest(...)` 仍会在计算成功后统一替换 report、有效配置和全部缓存；只读借用由编译器防止跨重算修改。

兼容性：直接字段赋值、原地修改结果嵌套对象、依赖 list/dict 具体类型，以及 Rust 直接访问字段/可变缓存的代码需按上述方式迁移。只读取结果、绘图、HTML 和交换格式保持支持。
