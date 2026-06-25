from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes, polars_to_arrow_bytes
from wbt._wbt import PyWeightBacktest, daily_performance

# Canonical field order for all stats output (from design doc)
STATS_FIELD_ORDER = [
    "绝对收益",
    "年化收益",
    "夏普比率",
    "卡玛比率",
    "新高占比",
    "单笔盈亏比",
    "单笔收益",
    "日胜率",
    "周胜率",
    "月胜率",
    "季胜率",
    "年胜率",
    "最大回撤",
    "年化波动率",
    "下行波动率",
    "新高间隔",
    "交易次数",
    "年化交易次数",
    "持仓K线数",
    "交易胜率",
    "多头占比",
    "空头占比",
    "品种数量",
    "开始日期",
    "结束日期",
]


def _reorder_stats(d: dict[str, object]) -> dict[str, object]:
    """Reorder dict keys to follow canonical stats field order."""
    ordered: dict[str, object] = {}
    for key in STATS_FIELD_ORDER:
        if key in d:
            ordered[key] = d[key]
    # Append any remaining keys not in the canonical order
    for key in d:
        if key not in ordered:
            ordered[key] = d[key]
    return ordered


def _to_date_key(value: object) -> int | None:
    """Convert date input to YYYYMMDD integer for Rust backend.

    Supports: str ("2020-01-01" or "20200101"), pd.Timestamp, datetime, int, None.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # "20200101" format
        if len(value) == 8 and value.isdigit():
            return int(value)
        # "2020-01-01" or ISO format
        return int(pd.Timestamp(value).strftime("%Y%m%d"))
    # pd.Timestamp, datetime, etc. — cast via str
    return int(pd.Timestamp(str(value)).strftime("%Y%m%d"))


WEIGH_DATA_TYPE = pd.DataFrame | pl.DataFrame | pl.LazyFrame | str | Path


def _arrow_or_empty(arrow_bytes: bytes) -> pd.DataFrame:
    """解码 Arrow 字节；空字节（无 pairs）返回空 DataFrame 而非抛错。"""
    if not arrow_bytes:
        return pd.DataFrame()
    return arrow_bytes_to_pd_df(arrow_bytes)


class WeightBacktest:
    """持仓权重回测

    飞书文档：https://s0cqcxuy3p.feishu.cn/wiki/Pf1fw1woQi4iJikbKJmcYToznxb
    """

    def __init__(
        self,
        data: WEIGH_DATA_TYPE,
        digits: int = 2,
        fee_rate: float = 0.0002,
        n_jobs: int = 1,
        weight_type: str = "ts",
        yearly_days: int = 252,
    ) -> None:
        """持仓权重回测

        :param data: 持仓权重数据，支持以下类型：
            - pd.DataFrame: columns = ['dt', 'symbol', 'weight', 'price']
            - polars.DataFrame / polars.LazyFrame: 同上列
            - str / Path: 文件路径（支持 .csv, .parquet, .feather, .arrow）

            DataFrame 数据样例如下：
            ===================  ========  ========  =======
            dt                   symbol      weight    price
            ===================  ========  ========  =======
            2019-01-02 09:01:00  DLi9001       0.5   961.695
            2019-01-02 09:02:00  DLi9001       0.25  960.72
            2019-01-02 09:03:00  DLi9001       0.25  962.669
            2019-01-02 09:04:00  DLi9001       0.25  960.72
            2019-01-02 09:05:00  DLi9001       0.25  961.695
            ===================  ========  ========  =======

        :param digits: int, 权重列保留小数位数
        :param fee_rate: float, default 0.0002，单边交易成本，包括手续费与冲击成本
        :param n_jobs: int, default 1，并行计算的线程数
        :param weight_type: str, default 'ts'，持仓权重类别，可选值：'ts'（时序策略）、'cs'（截面策略）
        :param yearly_days: int, default 252，年化交易日数量
        """
        self.digits = digits
        self.fee_rate = fee_rate
        self.weight_type = weight_type
        self.yearly_days = yearly_days

        # Type dispatch
        if isinstance(data, (str, Path)):
            # File path — delegate entirely to Rust
            self.dfw = None
            self._inner: PyWeightBacktest = PyWeightBacktest.from_file(
                str(data), digits, fee_rate, n_jobs, weight_type, yearly_days
            )
            self.symbols = self._inner.symbol_dict()
        else:
            # Try polars types first
            try:
                import polars as pl

                if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
                    self.dfw = None
                    arrow_data = polars_to_arrow_bytes(data)
                    self._inner = PyWeightBacktest.from_arrow(
                        arrow_data, digits, fee_rate, n_jobs, weight_type, yearly_days
                    )
                    self.symbols = self._inner.symbol_dict()
                    return
            except ImportError:
                pass

            # pd.DataFrame path
            dfw = data
            if dfw["weight"].dtype != "float":
                dfw["weight"] = dfw["weight"].astype(float)
            if dfw.isnull().sum().sum() > 0:
                raise ValueError(f"data 中存在空值，请先处理; 具体数据：\n{dfw[dfw.isnull().T.any().T]}")

            dfw = dfw[["dt", "symbol", "weight", "price"]].copy()
            dfw["weight"] = dfw["weight"].astype("float").round(digits)

            self.dfw = dfw.copy()
            self.symbols = list(dfw["symbol"].unique().tolist())

            arrow_data = pandas_to_arrow_bytes(dfw)
            self._inner = PyWeightBacktest.from_arrow(arrow_data, digits, fee_rate, n_jobs, weight_type, yearly_days)

    def get_top_symbols(self, n: int = 1, kind: str = "profit") -> list[str]:
        """获取回测赚钱/亏钱最多的前n个品种

        :param n: int, 前n个品种
        :param kind: str, 获取赚钱最多的品种，默认为"profit"，即获取赚钱最多的品种
        :return: list, 品种列表
        """
        assert kind in ["profit", "loss"], "kind 只能为 'profit' 或 'loss'"

        df = self.daily_return.copy()
        df.drop(columns=["total"], inplace=True)
        symbol_return = df.set_index("date").sum(axis=0)
        symbol_return = symbol_return.sort_values(ascending=kind != "profit")
        return symbol_return.head(n).index.tolist()

    @property
    def stats(self) -> dict:
        """回测绩效评价（多空综合）

        :return: dict, 包含三大类指标：

            收益指标：绝对收益、年化收益、夏普比率、卡玛比率、新高占比、单笔盈亏比、单笔收益、
                      日胜率、周胜率、月胜率、季胜率、年胜率

            风险指标：最大回撤、年化波动率、下行波动率、新高间隔

            特质指标：交易次数、年化交易次数、持仓K线数、交易胜率、多头占比、空头占比、品种数量

            样例如下：
            {'开始日期': '2017-01-03', '结束日期': '2023-07-31', '绝对收益': 0.6889,
            '年化收益': 0.0922, '夏普比率': 1.1931, '卡玛比率': 0.6715, '新高占比': 0.0861,
            '单笔盈亏比': 1.0568, '单笔收益': 25.59, '日胜率': 0.5436, '周胜率': 0.5769,
            '月胜率': 0.6250, '季胜率': 0.7500, '年胜率': 0.8333,
            '最大回撤': 0.1373, '年化波动率': 0.0773, '下行波动率': 0.0551, '新高间隔': 229.0,
            '交易次数': 120, '年化交易次数': 18.46, '持仓K线数': 972.81, '交易胜率': 0.3717,
            '多头占比': 0.5028, '空头占比': 0.4611, '品种数量': 9}
        """
        return _reorder_stats(self._inner.stats())

    @property
    def symbol_dict(self) -> list:
        """从 Rust 内部提取符号字典映射"""
        return self._inner.symbol_dict()

    def _map_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """按需将 Rust 内部的整数符号映射回字符串代码。"""
        if "symbol" in df.columns and pd.api.types.is_numeric_dtype(df["symbol"]):
            s_dict = dict(enumerate(self.symbol_dict))
            df["symbol"] = df["symbol"].map(s_dict)
        return df

    @property
    def daily_return(self) -> pd.DataFrame:
        """品种等权费后日收益率

        样例如下：
        ==========  ===========  ============  ===========  ============
        date            DLj9001      SQag9001     ZZSF9001         total
        ==========  ===========  ============  ===========  ============
        2017-01-03   0.0264417    0.00246216   -0.00180836   0.00903183
        2017-01-04  -0.0237968   -0.00226261    0.00659331  -0.00648869
        2017-01-05  -0.00247365   0.00568681   -0.00669249  -0.00115977
        2017-01-06  -0.0145385   -0.0103144    -0.0184913   -0.0144481
        2017-01-07   0           -0.000373236   0           -0.000124412
        ==========  ===========  ============  ===========  ============

        """
        return self._map_symbols(arrow_bytes_to_pd_df(self._inner.daily_return()))

    def yearly_return(self, min_days: int = 120) -> pd.DataFrame:
        """计算每个 symbol 的年度收益（按年复利）

        基于 daily_return 宽表按年分组，用 (1+r1)*(1+r2)*...*(1+rn) - 1 公式计算。
        策略整体收益用 symbol='total' 表示。

        :param min_days: int, default 120，每年最少交易日数量；不足的 (year, symbol) 会被跳过
        :return: pd.DataFrame，列为 ['year', 'symbol', 'return']，按 (year, symbol) 升序排序

            样例如下：
            ====  ========  ===========
            year  symbol         return
            ====  ========  ===========
            2017  DLj9001    0.1542
            2017  SQag9001  -0.0318
            2017  total      0.0612
            2018  DLj9001   -0.0205
            2018  SQag9001   0.0876
            2018  total      0.0336
            ====  ========  ===========
        """
        return arrow_bytes_to_pd_df(self._inner.yearly_return(min_days))

    @property
    def dailys(self) -> pd.DataFrame:
        """品种每日的交易信息

        columns = ['symbol', 'date', 'n1b', 'edge', 'return', 'cost', 'turnover',
                   'long_edge', 'short_edge', 'long_cost', 'short_cost',
                   'long_turnover', 'short_turnover', 'long_return', 'short_return']

        其中:
            date            交易日
            symbol          合约代码
            n1b             品种每日基准收益率
            edge            策略每日收益率
            return          策略每日收益率减去交易成本后的真实收益（= edge - cost）
            cost            交易成本
            turnover        当日的单边换手率
            long_edge       多头部分的策略收益
            short_edge      空头部分的策略收益
            long_return     多头部分的净收益（= long_edge - long_cost）
            short_return    空头部分的净收益（= short_edge - short_cost）
        """
        return self._map_symbols(arrow_bytes_to_pd_df(self._inner.dailys()))

    @property
    def alpha(self) -> pd.DataFrame:
        """策略超额收益

        样例数据如下：
        ==========  ============  ============  ============
        date                超额          策略          基准
        ==========  ============  ============  ============
        2017-01-03   0.0163507     0.00903183   -0.00731888
        2017-01-04  -0.0180457    -0.00648869    0.011557
        2017-01-05  -0.000717903  -0.00115977   -0.000441871
        2017-01-06  -0.00610561   -0.0144481    -0.00834245
        2017-01-07  -0.000867702  -0.000373236   0.000494466
        ==========  ============  ============  ============
        """
        return arrow_bytes_to_pd_df(self._inner.alpha())

    def _pivot_daily_return(self, values_col: str) -> pd.DataFrame:
        df = self.dailys.copy()
        dfv = pd.pivot_table(df, index="date", columns="symbol", values=values_col)
        if self.weight_type == "ts":
            dfv["total"] = dfv.mean(axis=1)
        elif self.weight_type == "cs":
            dfv["total"] = dfv.sum(axis=1)
        else:
            raise ValueError(f"weight_type {self.weight_type} not supported")
        return dfv.reset_index(drop=False)

    def _compute_stats(self, df: pd.DataFrame, column: str) -> dict:
        stats = daily_performance(df[column].to_numpy(), yearly_days=self.yearly_days)
        stats["开始日期"] = df["date"].min().strftime("%Y-%m-%d")
        stats["结束日期"] = df["date"].max().strftime("%Y-%m-%d")
        return stats

    @property
    def alpha_stats(self) -> dict:
        """策略超额收益统计"""
        return self._compute_stats(self.alpha, "超额")

    @property
    def bench_stats(self) -> dict:
        """基准收益统计"""
        return self._compute_stats(self.alpha, "基准")

    @property
    def long_daily_return(self):
        """多头每日收益率

        样例如下：
        ==========  ==========  ===========  ===========  ============
        date           DLj9001     SQag9001     ZZSF9001         total
        ==========  ==========  ===========  ===========  ============
        2017-01-03   0           0.00246216  -0.00180836   0.000217931
        2017-01-04   0          -0.00226261   0.00659331   0.00144357
        2017-01-05   0           0.00568681  -0.00669249  -0.000335226
        2017-01-06  -0.0100097  -0.00816301  -0.0184913   -0.0122213
        2017-01-07   0           0            0            0
        ==========  ==========  ===========  ===========  ============
        """
        return self._pivot_daily_return("long_return")

    @property
    def short_daily_return(self):
        """空头每日收益率"""
        return self._pivot_daily_return("short_return")

    @property
    def long_stats(self) -> dict:
        """多头收益统计（从 Rust 端计算）"""
        return _reorder_stats(self._inner.long_stats())

    @property
    def short_stats(self) -> dict:
        """空头收益统计（从 Rust 端计算）"""
        return _reorder_stats(self._inner.short_stats())

    def segment_stats(
        self,
        sdt: str | int | pd.Timestamp | None = None,
        edt: str | int | pd.Timestamp | None = None,
        kind: str = "多空",
    ) -> dict:
        """分段统计

        :param sdt: str | None, 开始日期，支持 "2020-01-01"、"20200101"、pd.Timestamp 格式，None 表示从头开始
        :param edt: str | None, 结束日期，同上格式，None 表示到末尾
        :param kind: str, "多空" | "多头" | "空头"
        :return: dict, 字段与 :attr:`stats` 同构（含 ``多头占比`` / ``空头占比`` / ``品种数量``）。
            ``多头占比`` / ``空头占比`` 遵循 ``kind`` 单边语义：
            ``kind="多头"`` 时 ``空头占比`` 恒为 0；``kind="空头"`` 时 ``多头占比`` 恒为 0；
            ``kind="多空"`` 时两者各自取 [sdt, edt] 范围内的实际权重行占比。
        """
        sdt_int = _to_date_key(sdt)
        edt_int = _to_date_key(edt)
        return _reorder_stats(self._inner.segment_stats(sdt_int, edt_int, kind))

    @property
    def long_alpha_stats(self) -> dict:
        """波动率调整后的多头超额收益统计"""
        return _reorder_stats(self._inner.long_alpha_stats())

    def is_good_strategy(
        self,
        mode: str = "history",
        target_vol: float = 0.20,
        max_dd_threshold: float = 0.20,
        min_year_days: int = 200,
        recent_days: int = 252,
        min_history_days: int = 60,
    ) -> dict[str, object]:
        """判定策略能不能搞。

        两种判定模式，业务口径与方案子文档一致：

        - ``mode="history"``：每个完整自然年（≥ ``min_year_days``）满足**三者之一**即合格——
          绝对收益 > 0 **或** 波动率归一多头超额 > 0 **或** 当年多头超额最大回撤 <
          ``max_dd_threshold``；所有完整自然年都合格才 ``is_good``。
          （回撤条件已从"跨全样本的独立硬门"下放为逐年计算、并入年度 OR；不再有全样本级别的
          回撤一票否决。）
        - ``mode="recent"``：尾部 ``recent_days`` 天满足**三者之一**即可——绝对收益 > 0 **或**
          波动率归一多头超额 > 0 **或** 近期多头超额最大回撤 < ``max_dd_threshold``；
          **且** 近期最大回撤严格小于"剔除 recent 窗口后"的历史最大回撤（唯一保留的硬门）。
          两段计算窗口完全错开。

        :param mode: 判定模式，``"history"`` 或 ``"recent"``。
        :param target_vol: 波动率归一目标年化波动率（默认 0.20）。
        :param max_dd_threshold: 多头超额最大回撤阈值（默认 0.20）。history 模式下为**逐年**
            回撤阈值，recent 模式下为近期窗口回撤阈值。
        :param min_year_days: 视为"完整自然年"所需的最少交易日数（默认 200）。
        :param recent_days: ``"recent"`` 模式取序列尾部的日数（默认 252）。
        :param min_history_days: ``"recent"`` 模式下剔除 recent 窗口后的历史段必须达到
            的最小长度，否则 ``history_window_empty=True`` 且 ``is_good=False``
            （默认 60；设为 0 关闭 floor）。

        **波动率归一化口径**：``compute_vol_adjusted_alpha`` 用**全样本** long/bench
        的年化标准差做归一化，整段共用同一组 scale。如果用户期望 "recent 窗口按自身
        vol 归一化"，需要单独跑一个截短样本的回测对象再调用本方法。

        **退化与错误**：

        - 输入序列含 NaN/Inf、long/bench 的年化波动率 ~= 0，归一化无法定义 → 返回 dict
          中 ``alpha_degenerate=True``，所有 alpha 派生字段为 ``None``，``is_good=False``。
        - 输入日期 / 序列长度不匹配、空输入、``recent_days=0``、``target_vol<=0`` 等用户
          错误 → 抛出 ``Exception``（Rust 端 ``WbtError::InvalidInput``）。

        :return: dict，顶层 key 用英文 snake_case，按字母序稳定排列。

            **history / recent 两个模式返回的 key 集合互斥**，按 ``mode`` 字段 dispatch；
            不要假设可以同时拿到两个模式的字段。

            **history 模式必含 key**：``alpha_degenerate``、``cond_yearly_passed``、
            ``complete_year_count``、``is_good``、``mode``、``reason``、``yearly_metrics``。
            ``yearly_metrics`` 每项含 ``year``、``abs_return``、``alpha_return``、
            ``alpha_max_drawdown``（当年超额回撤）、``days``、``is_complete_year``、
            ``year_passed``。全样本回撤硬门已取消，``history_alpha_max_drawdown`` /
            ``cond_history_dd_passed`` 不再返回。

            **recent 模式必含 key**：``alpha_degenerate``、``cond_recent_dd_passed``
            （现仅表示"近期回撤严格小于历史回撤"这一硬门）、``cond_recent_return_passed``
            （现已并入"近期回撤 < 阈值"的 OR 分支）、``history_alpha_max_drawdown_excl_recent``
            （退化或样本不足时为 ``None``）、``history_window_empty``、``is_good``、
            ``mode``、``reason``、``recent_abs_return``、``recent_actual_days``、
            ``recent_alpha_max_drawdown``（退化时为 ``None``）、``recent_alpha_return``
            （退化时为 ``None``）、``recent_end_date``、``recent_start_date``。

            ``is_good`` 类型为 ``bool``；``yearly_metrics`` 类型为 ``list[dict]``；
            ``reason`` 类型为 ``str``（成功时为空字符串）。
        """
        return self._inner.is_good_strategy(
            mode, target_vol, max_dd_threshold, min_year_days, recent_days, min_history_days
        )

    @property
    def pairs(self) -> pd.DataFrame:
        """所有交易对数据

        包含所有品种的买卖交易对信息，包含以下字段：
        - symbol: 品种代码
        - 开仓时间: 开仓时间
        - 平仓时间: 平仓时间
        - 方向: 交易方向（多头/空头）
        - 开仓价格: 开仓价格
        - 平仓价格: 平仓价格
        - 盈亏比例: 盈亏比例
        - 持仓K线数: 持仓K线数量
        """
        return arrow_bytes_to_pd_df(self._inner.pairs())

    @property
    def aggregated_pairs(self) -> pd.DataFrame:
        """聚合去重后的开平记录（按 symbol/开仓时间/平仓时间 聚合，记 count）

        columns = ['symbol', '交易方向', '开仓时间', '平仓时间', '开仓价格',
                   '平仓价格', '持仓K线数', '盈亏比例', 'count']
        其中 盈亏比例 单位为 BP（与 :attr:`pairs` 一致），count 为被合并的原始记录数。
        """
        return _arrow_or_empty(self._inner.aggregated_pairs())

    def key_trades(self, top: int = 3) -> pd.DataFrame:
        """每年最赚/最亏各 top 笔关键交易（在聚合结果上按平仓年份取极值）

        :param top: int, 每年每侧取的笔数，默认 3
        :return: pd.DataFrame，在 :attr:`aggregated_pairs` 列基础上增加 ['year', 'kind']，
            kind ∈ {'best', 'worst'}。
        """
        return _arrow_or_empty(self._inner.key_trades(top))

    def to_result(self, target_vol: float = 0.20):
        """组装绘图与审核页面统一输入 :class:`wbt.result.BacktestResult`。

        :param target_vol: float, 波动率归一目标年化波动率，默认 0.20
        """
        from wbt.result import BacktestResult

        return BacktestResult.from_backtest(self, target_vol=target_vol)

    def get_symbol_daily(self, symbol: str) -> pd.DataFrame:
        """获取某个合约的每日收益率

        :param symbol: str，合约代码
        :return: pd.DataFrame，品种每日收益率，
            columns = ['date', 'symbol', 'edge', 'return', 'cost', 'n1b',
                        'turnover', 'long_return', 'short_return', ...]
        """
        df = self.dailys
        return df[df["symbol"] == symbol].copy()

    def get_symbol_pairs(self, symbol: str) -> pd.DataFrame:
        """获取某个合约的开平交易记录

        :param symbol: str，合约代码
        :return: pd.DataFrame，品种的开平仓交易记录
        """
        df = self.pairs
        symbol_col = "标的代码" if "标的代码" in df.columns else "symbol"
        return df[df[symbol_col] == symbol].copy()


def backtest(
    data: WEIGH_DATA_TYPE,
    digits: int = 2,
    fee_rate: float = 0.0002,
    n_jobs: int = 1,
    weight_type: str = "ts",
    yearly_days: int = 252,
) -> WeightBacktest:
    """WeightBacktest 的便捷接口

    :param data: 持仓权重数据，支持以下类型：
        - pd.DataFrame: columns = ['dt', 'symbol', 'weight', 'price']
        - polars.DataFrame / polars.LazyFrame: 同上列
        - str / Path: 文件路径（支持 .csv, .parquet, .feather, .arrow)
    :param digits: int, 权重列保留小数位数
    :param fee_rate: float, default 0.0002, 单边交易成本, 包括手续费与冲击成本
    :param n_jobs: int, default 1, 并行计算的线程数
    :param weight_type: str, default 'ts'，持仓权重类别，可选值：'ts'（时序策略）、'cs'（截面策略）
    :param yearly_days: int, default 252, 年化交易日数量
    :return: WeightBacktest 对象
    """
    return WeightBacktest(
        data, digits=digits, fee_rate=fee_rate, n_jobs=n_jobs, weight_type=weight_type, yearly_days=yearly_days
    )
