# 多头超额的统一归一化口径

Rust 多头超额统计、策略判定与 Python 结果曲线共用 Rust 数值算子。以全样本总体标准差（ddof=0）乘 √yearly_days 得到年化波动率，分别将多头与基准归一至 target_vol 后相减。切片判定继续使用全样本尺度。

`to_result(target_vol=...)` 同时影响归一化曲线、history/recent 判定及年度超额指标。`long_alpha_stats` 保持固定 0.20 的既有属性接口；比较自定义 target_vol 的结果时需考虑这一差异。

无法归一化包括空/单值序列、非有限输入、年化波动率低于 1e-12、计算溢出。算子统一返回无结果；统计属性为兼容保留零指标，判定标记 alpha_degenerate 且派生指标为 null，结果曲线为 NaN（JSON/MessagePack 导出为 null）。非退化但多头与基准完全相抵时，零超额仍是有效序列。

这是对旧 Python 曲线 ddof=1、常数序列不缩放以及结果判定遗漏 target_vol 的修正，可能改变历史曲线和非默认 target_vol 下的判定。
