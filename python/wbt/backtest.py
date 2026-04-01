import pandas as pd
from wbt._wbt import PyWeightBacktest, daily_performance
from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes


class WeightBacktest:
    """持仓权重回测

    飞书文档：https://s0cqcxuy3p.feishu.cn/wiki/Pf1fw1woQi4iJikbKJmcYToznxb
    """

    def __init__(
        self,
        dfw: pd.DataFrame,
        digits: int = 2,
        fee_rate: float = 0.0002,
        n_jobs: int = 1,
        weight_type: str = "ts",
        yearly_days: int = 252,
    ) -> None:
        """持仓权重回测

        :param dfw: pd.DataFrame, columns = ['dt', 'symbol', 'weight', 'price'], 持仓权重数据，其中

            dt      为K线结束时间，必须是连续的交易时间序列，不允许有时间断层
            symbol  为合约代码，
            weight  为K线结束时间对应的持仓权重，品种之间的权重是独立的，不会互相影响
            price   为结束时间对应的交易价格，可以是当前K线的收盘价，或者下一根K线的开盘价，或者未来N根K线的TWAP、VWAP等

            数据样例如下：
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
        if dfw['weight'].dtype != 'float':
            dfw['weight'] = dfw['weight'].astype(float)
        if dfw.isnull().sum().sum() > 0:
            raise ValueError(f"dfw 中存在空值，请先处理; 具体数据：\n{dfw[dfw.isnull().T.any().T]}")

        dfw = dfw[['dt', 'symbol', 'weight', 'price']].copy()
        dfw['weight'] = dfw['weight'].astype('float').round(digits)

        # 保存实例变量，与 Python 原版一致
        self.dfw = dfw.copy()
        self.digits = digits
        self.fee_rate = fee_rate
        self.weight_type = weight_type
        self.yearly_days = yearly_days
        self.symbols = list(dfw['symbol'].unique().tolist())

        data = pandas_to_arrow_bytes(dfw)
        self._inner: PyWeightBacktest = PyWeightBacktest.from_arrow(
            data,
            digits,
            fee_rate,
            n_jobs,
            weight_type,
            yearly_days,
        )

    def get_top_symbols(self, n: int = 1, kind: str = "profit") -> list:
        """获取回测赚钱/亏钱最多的前n个品种

        :param n: int, 前n个品种
        :param kind: str, 获取赚钱最多的品种，默认为"profit"，即获取赚钱最多的品种
        :return: list, 品种列表
        """
        assert kind in ["profit", "loss"], "kind 只能为 'profit' 或 'loss'"

        df = self.daily_return.copy()
        df.drop(columns=['total'], inplace=True)
        symbol_return = df.set_index('date').sum(axis=0)
        symbol_return = symbol_return.sort_values(ascending=not kind == "profit")
        return symbol_return.head(n).index.tolist()

    @property
    def stats(self) -> dict:
        """回测绩效评价

        :return: dict, 回测绩效评价, 样例如下：
            {'开始日期': '2017-01-03', '结束日期': '2023-07-31', '绝对收益': 0.6889, '年化': 0.0922,
            '夏普': 1.1931, '最大回撤': 0.1373, '卡玛': 0.6715, '日胜率': 0.5436, '日盈亏比': 1.0568,
            '日赢面': 0.1181, '年化波动率': 0.0773, '下行波动率': 0.0551, '非零覆盖': 0.9665,
            '盈亏平衡点': 0.9782, '新高间隔': 229.0, '新高占比': 0.0861, '回撤风险': 1.7762,
            '回归年度回报率': 0.1046, '长度调整平均最大回撤': 0.1714, '交易胜率': 0.3717,
            '单笔收益': 25.59, '持仓K线数': 972.81, '多头占比': 0.5028, '空头占比': 0.4611,
            '与基准相关性': 0.0727, '与基准空头相关性': -0.148, '波动比': 0.5865,
            '与基准波动相关性': 0.2055, '品种数量': 9}
        """
        return self._inner.stats()

    @property
    def symbol_dict(self) -> list:
        """从 Rust 内部提取符号字典映射"""
        return self._inner.symbol_dict()

    def _map_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """按需将 Rust 内部的整数符号映射回字符串代码。"""
        if "symbol" in df.columns and pd.api.types.is_numeric_dtype(df["symbol"]):
            s_dict = {i: s for i, s in enumerate(self.symbol_dict)}
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

    @property
    def dailys(self) -> pd.DataFrame:
        """品种每日的交易信息

        columns = ['date', 'symbol', 'edge', 'return', 'cost', 'n1b', 'turnover']

        其中:
            date        交易日，
            symbol      合约代码，
            n1b         品种每日收益率，
            edge        策略每日收益率，
            return      策略每日收益率减去交易成本后的真实收益，
            cost        交易成本
            turnover    当日的单边换手率
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
    def long_stats(self):
        """多头收益统计

        输出样例如下：
            {'绝对收益': 0.5073,
            '年化': 0.0679,
            '夏普': 1.0786,
            '最大回撤': 0.0721,
            '卡玛': 0.9418,
            '日胜率': 0.5276,
            '日盈亏比': 1.1263,
            '日赢面': 0.1218,
            '年化波动率': 0.063,
            '下行波动率': 0.0478,
            '非零覆盖': 0.9421,
            '盈亏平衡点': 0.9819,
            '新高间隔': 222.0,
            '新高占比': 0.0728,
            '回撤风险': 1.1444,
            '回归年度回报率': 0.0745,
            '长度调整平均最大回撤': 0.2595,
            '开始日期': '2017-01-03',
            '结束日期': '2023-07-31'}
        """
        return self._compute_stats(self.long_daily_return, "total")

    @property
    def short_stats(self):
        """空头收益统计"""
        return self._compute_stats(self.short_daily_return, "total")

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

    def get_symbol_daily(self, symbol: str) -> pd.DataFrame:
        """获取某个合约的每日收益率

        :param symbol: str，合约代码
        :return: pd.DataFrame，品种每日收益率，
            columns = ['date', 'symbol', 'edge', 'return', 'cost', 'n1b',
                        'turnover', 'long_return', 'short_return', ...]
        """
        df = self.dailys
        return df[df['symbol'] == symbol].copy()

    def get_symbol_pairs(self, symbol: str) -> pd.DataFrame:
        """获取某个合约的开平交易记录

        :param symbol: str，合约代码
        :return: pd.DataFrame，品种的开平仓交易记录
        """
        df = self.pairs
        symbol_col = '标的代码' if '标的代码' in df.columns else 'symbol'
        return df[df[symbol_col] == symbol].copy()
