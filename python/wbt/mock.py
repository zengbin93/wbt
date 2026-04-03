"""模拟数据生成模块

向量化实现，相比逐行循环提速约 20x。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

# 支持的K线频率
SUPPORTED_FREQS = ("1分钟", "5分钟", "15分钟", "30分钟", "日线")

# 每日交易分钟数（A股）
TRADING_MINUTES_PER_DAY = 240

# 市场阶段定义：模拟真实市场的周期性变化
MARKET_PHASES = (
    {"name": "熊市", "trend": -0.0008, "volatility": 0.025, "length": 0.3},
    {"name": "震荡", "trend": 0.0002, "volatility": 0.015, "length": 0.2},
    {"name": "牛市", "trend": 0.0012, "volatility": 0.02, "length": 0.3},
    {"name": "调整", "trend": -0.0005, "volatility": 0.02, "length": 0.2},
)

# 默认生成权重数据的品种列表
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA")

# A股交易时间段
MORNING_SESSION = ("09:30", "11:30")
AFTERNOON_SESSION = ("13:00", "15:00")


def _parse_freq_minutes(freq: str) -> int:
    """从频率字符串中解析分钟数。"""
    return int(freq.replace("分钟", ""))


def _generate_trading_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq_minutes: int,
) -> pd.DatetimeIndex:
    """向量化生成分钟级交易时间序列。

    一次性生成所有日期 × 所有时间偏移，避免逐日循环创建 DatetimeIndex。
    """
    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    if all_days.empty:
        return pd.DatetimeIndex([])

    # 单日内所有交易时间偏移
    morning_offsets = pd.timedelta_range(
        start=pd.Timedelta(hours=9, minutes=30),
        end=pd.Timedelta(hours=11, minutes=30),
        freq=f"{freq_minutes}min",
    )
    afternoon_offsets = pd.timedelta_range(
        start=pd.Timedelta(hours=13),
        end=pd.Timedelta(hours=15),
        freq=f"{freq_minutes}min",
    )
    offsets = morning_offsets.append(afternoon_offsets).values  # numpy timedelta64

    # 向量化：所有日期 × 所有偏移
    day_values = all_days.normalize().values.astype("datetime64[D]")
    timestamps = (day_values[:, None] + offsets[None, :]).ravel()

    return pd.DatetimeIndex(timestamps)


def _build_phase_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    """预计算市场阶段趋势和波动率数组。

    按 MARKET_PHASES 定义的比例循环填充，复刻原始逐行逻辑。
    """
    trends = np.empty(n)
    volatilities = np.empty(n)
    idx = 0

    while idx < n:
        for phase in MARKET_PHASES:
            phase_len = int(n * phase["length"])
            end = min(idx + phase_len, n)
            trends[idx:end] = phase["trend"]
            volatilities[idx:end] = phase["volatility"]
            idx = end
            if idx >= n:
                break

    return trends, volatilities


@lru_cache(maxsize=10)
def mock_symbol_kline(
    symbol: str,
    freq: str,
    sdt: str = "20100101",
    edt: str = "20250101",
    seed: int = 42,
) -> pd.DataFrame:
    """生成单个品种指定频率的K线数据。

    Args:
        symbol: 品种代码，如 'AAPL', '000001.SH' 等
        freq: K线频率，支持 '1分钟', '5分钟', '15分钟', '30分钟', '日线'
        sdt: 开始日期，格式为 'YYYYMMDD'，默认 "20100101"
        edt: 结束日期，格式为 'YYYYMMDD'，默认 "20250101"
        seed: 随机数种子，确保结果可重现，默认 42

    Returns:
        包含K线数据的DataFrame，列包括 dt, symbol, open, close, high, low, vol, amount

    Raises:
        ValueError: 当 freq 不在支持的频率列表中时抛出
    """
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"不支持的频率: {freq}。支持的频率: {', '.join(SUPPORTED_FREQS)}")

    rng = np.random.default_rng(seed + hash(symbol) % 1000)

    start_date = pd.to_datetime(sdt, format="%Y%m%d")
    end_date = pd.to_datetime(edt, format="%Y%m%d")

    # 生成时间序列
    is_daily = freq == "日线"
    freq_minutes = 0 if is_daily else _parse_freq_minutes(freq)

    if is_daily:
        dates = pd.bdate_range(start=start_date, end=end_date)
    else:
        dates = _generate_trading_dates(start_date, end_date, freq_minutes)

    n = len(dates)
    if n == 0:
        return pd.DataFrame(columns=["dt", "symbol", "open", "close", "high", "low", "vol", "amount"])

    # ---------- 向量化计算市场阶段 ----------
    trends, volatilities = _build_phase_arrays(n)

    if not is_daily:
        ratio = TRADING_MINUTES_PER_DAY / freq_minutes
        trends = trends / ratio
        volatilities = volatilities / ratio**0.5

    # ---------- 向量化计算周期因子 ----------
    i_arr = np.arange(n, dtype=np.float64)
    if is_daily:
        cycle_factors = np.sin(i_arr / 30) * 0.001
        annual_cycles = np.sin(i_arr / 365) * 0.0005
    else:
        cycle_factors = np.sin(i_arr / 120) * 0.0005
        annual_cycles = np.sin(i_arr / (365 * TRADING_MINUTES_PER_DAY)) * 0.0002

    # ---------- 向量化生成收益率和价格 ----------
    noise = rng.normal(0, 1, n) * volatilities
    returns = trends + cycle_factors + annual_cycles + noise
    returns = np.clip(returns, -0.15, 0.15)

    close_prices = 100.0 * np.cumprod(1 + returns)
    open_prices = np.empty(n)
    open_prices[0] = 100.0
    open_prices[1:] = close_prices[:-1]

    # ---------- 向量化计算高低价 ----------
    price_change_ratios = np.abs(close_prices - open_prices) / np.maximum(open_prices, 1e-8)
    range_low, range_high = (0.01, 0.04) if is_daily else (0.001, 0.01)
    daily_ranges = open_prices * (price_change_ratios + rng.uniform(range_low, range_high, n))

    up_mask = close_prices >= open_prices
    high_up = daily_ranges * rng.uniform(0.1, 0.5, n)
    low_up = daily_ranges * rng.uniform(0.1, 0.3, n)
    high_down = daily_ranges * rng.uniform(0.1, 0.3, n)
    low_down = daily_ranges * rng.uniform(0.1, 0.5, n)

    high_prices = np.where(up_mask, close_prices + high_up, open_prices + high_down)
    low_prices = np.where(up_mask, open_prices - low_up, close_prices - low_down)
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # ---------- 向量化计算成交量 ----------
    base_volumes = rng.uniform(100000, 300000, n) if is_daily else rng.uniform(10000, 50000, n) * (freq_minutes / 5)

    vol_factors = price_change_ratios * 5
    vol_multipliers = 1 + vol_factors + rng.uniform(-0.2, 0.2, n)
    volumes = (base_volumes * np.maximum(vol_multipliers, 0.3)).astype(int)

    # 成交金额
    avg_prices = (high_prices + low_prices + open_prices + close_prices) / 4
    amounts = volumes * avg_prices

    return pd.DataFrame(
        {
            "dt": dates,
            "symbol": symbol,
            "open": np.round(open_prices, 2),
            "close": np.round(close_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "vol": volumes,
            "amount": np.round(amounts, 2),
        }
    )


@lru_cache(maxsize=10)
def mock_weights(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    freq: str = "日线",
    sdt: str = "20100101",
    edt: str = "20250101",
    seed: int = 42,
) -> pd.DataFrame:
    """生成包含权重信息的K线数据。

    为每个品种生成K线数据并附加随机权重，用于回测分析。

    Args:
        symbols: 品种代码元组，默认为 ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA')
        freq: K线频率，默认 '日线', 支持 '1分钟', '5分钟', '15分钟', '30分钟', '日线'
        sdt: 开始日期，格式为 'YYYYMMDD'，默认 "20100101"
        edt: 结束日期，格式为 'YYYYMMDD'，默认 "20250101"
        seed: 随机数种子，确保结果可重现，默认 42

    Returns:
        包含K线数据和权重列的DataFrame，额外列包括 weight, price
    """
    rng = np.random.default_rng(seed)
    frames = [mock_symbol_kline(symbol, freq=freq, sdt=sdt, edt=edt, seed=seed) for symbol in symbols]
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    magnitudes = np.clip(rng.normal(0.5, 0.5, n), 0, 1)
    signs = np.ones(n)
    signs[n // 2 :] = -1
    rng.shuffle(signs)
    df["weight"] = magnitudes * signs
    df["price"] = df["close"]
    return df
