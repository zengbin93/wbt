# D04：交换格式和指标名称

所有 `BacktestResult.to_dict/to_json/to_msgpack/dump_json/dump_msgpack` 及同名模块函数统一默认 `full=False`。过去 JSON/MessagePack 默认完整导出的调用方，应显式传 `full=True`。

JSON 返回 UTF-8 bytes，MessagePack 返回 bytes，dump 写文件，load 返回普通 dict payload。load 不恢复 BacktestResult：缺少回测源数据不能可靠恢复懒计算；绘图需要保留原实例或从原始权重重新回测。to_dict 返回无 envelope 的 payload。

新写出 envelope 为 `{"format":"wbt.backtest_result","format_version":2,"full":false,"payload":{...}}`。JSON 和 MessagePack 共用相同版本、结构和规范化逻辑。Rust `decode_wire/load_wire` 读取 MessagePack 并返回 serde_json::Value。

最小 payload（full=False）必需字段：

| 类型 | 字段 |
|---|---|
| string | start_date、end_date、weight_type |
| integer | symbol_count、yearly_days |
| array | dates、year_starts |
| object | curves、return_dist、monthly、symbol_returns、pairs_dist、stats、stats_by_side |

curves 必含多空、多头、空头、基准、超额，每条含 daily/cum/drawdown 数组，与 dates 等长。日期为 ISO 字符串；比例为原始小数，仅 *_pct 为百分比；非有限值规范化为 null。上述为结构最低要求，不代表完整财务数据有效性校验。

full=True 额外必含 curves_voladj、drawdowns、key_trades、verdict、verdict_recent、yearly_returns、rolling、segment_comparison，其中 drawdowns 为数组，其余为对象。full=False 不计算、也不输出这些字段。full 不意味着包含原始持仓或可还原回测对象。

版本迁移：读取器同时支持 v1 与 v2。v1 不强加新 schema、不推断 full，也不伪造缺失数据，原 payload 原样返回；v2 校验最小字段类型、曲线长度及 full 一致性。未知版本拒绝，未知附加字段保留。v1 文件继续可读，升级生成端后应同时升级读取端；旧读取端会拒绝 v2。修改或删除必需字段、改变单位或含义需升主格式版本；新增可选字段不需要升版本。旧 fixture 保留，并新增 Python 生成的 v2 fixture 供 Rust 交叉读取。

指标名集中在 `wbt.metrics`：以 Rust DailyPerformance 英文成员为机器名（如 annual_returns、sharpe_ratio），METRIC_LABELS 提供中文展示，to_machine_metrics 规范化已有中文字典，lookup_metric 兼容机器名/中文全名/历史简称。冲突时机器名优先，然后中文全名、历史简称；未知字段保留。stats 与 daily_performance 原有中文返回键不改，HTML 与 Plotly 表格共用映射，判定年度/近期字段标签也集中在该模块。
