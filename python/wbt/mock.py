"""
模拟数据生成模块
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
    """从频率字符串中解析分钟数。

    Args:
        freq: K线频率，如 '1分钟', '5分钟', '30分钟'

    Returns:
        对应的分钟数整数值
    """
    return int(freq.replace("分钟", ""))


def _generate_trading_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq_minutes: int,
) -> pd.DatetimeIndex:
    """生成分钟级交易时间序列（含A股交易时间段过滤）。

    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq_minutes: K线分钟间隔

    Returns:
        包含所有交易日交易时段的时间索引
    """
    trading_days = pd.date_range(start=start_date, end=end_date, freq="D")
    dates: list[pd.Timestamp] = []

    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")
        for session_start, session_end in (MORNING_SESSION, AFTERNOON_SESSION):
            times = pd.date_range(
                start=f"{day_str} {session_start}",
                end=f"{day_str} {session_end}",
                freq=f"{freq_minutes}min",
            )
            dates.extend(times.tolist())

    return pd.DatetimeIndex(dates)


def _compute_cycle_factors(idx: int, freq: str) -> tuple[float, float]:
    """计算周期性波动因子。

    Args:
        idx: 当前时间步索引
        freq: K线频率

    Returns:
        (cycle_factor, annual_cycle) 周期性波动因子元组
    """
    if freq == "日线":
        return np.sin(idx / 30) * 0.001, np.sin(idx / 365) * 0.0005

    return np.sin(idx / 120) * 0.0005, np.sin(idx / (365 * TRADING_MINUTES_PER_DAY)) * 0.0002


def _adjust_for_intraday(trend: float, volatility: float, freq_minutes: int) -> tuple[float, float]:
    """将日线级别的趋势和波动率调整为分钟级别。

    Args:
        trend: 日线趋势值
        volatility: 日线波动率
        freq_minutes: K线分钟间隔

    Returns:
        (adjusted_trend, adjusted_volatility) 调整后的元组
    """
    ratio = TRADING_MINUTES_PER_DAY / freq_minutes
    return trend / ratio, volatility / ratio**0.5


def _compute_price_range(
    base_price: float,
    open_price: float,
    close_price: float,
    freq: str,
) -> tuple[float, float]:
    """计算高低价，确保价格关系正确。

    Args:
        base_price: 基准价格
        open_price: 开盘价
        close_price: 收盘价
        freq: K线频率

    Returns:
        (high_price, low_price) 高低价元组
    """
    price_change_ratio = abs(close_price - open_price) / open_price
    range_low, range_high = (0.01, 0.04) if freq == "日线" else (0.001, 0.01)
    daily_range = base_price * (price_change_ratio + np.random.uniform(range_low, range_high))

    if close_price > open_price:
        high_price = close_price + daily_range * np.random.uniform(0.1, 0.5)
        low_price = open_price - daily_range * np.random.uniform(0.1, 0.3)
    else:
        high_price = open_price + daily_range * np.random.uniform(0.1, 0.3)
        low_price = close_price - daily_range * np.random.uniform(0.1, 0.5)

    return max(high_price, open_price, close_price), min(low_price, open_price, close_price)


def _compute_volume(freq: str, freq_minutes: int, price_change_ratio: float) -> int:
    """计算成交量。

    Args:
        freq: K线频率
        freq_minutes: K线分钟间隔（日线为0）
        price_change_ratio: 价格变化比率

    Returns:
        成交量
    """
    if freq == "日线":
        base_volume = np.random.uniform(100000, 300000)
    else:
        base_volume = np.random.uniform(10000, 50000) * (freq_minutes / 5)

    volatility_factor = price_change_ratio * 5
    volume_multiplier = 1 + volatility_factor + np.random.uniform(-0.2, 0.2)
    return int(base_volume * max(volume_multiplier, 0.3))


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

    np.random.seed(seed + hash(symbol) % 1000)

    start_date = pd.to_datetime(sdt, format="%Y%m%d")
    end_date = pd.to_datetime(edt, format="%Y%m%d")

    # 生成时间序列
    is_daily = freq == "日线"
    freq_minutes = 0 if is_daily else _parse_freq_minutes(freq)

    if is_daily:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
    else:
        dates = _generate_trading_dates(start_date, end_date, freq_minutes)

    # K线生成
    base_price = 100.0
    total_periods = len(dates)
    phase_idx = 0
    phase_periods = 0
    current_phase = MARKET_PHASES[phase_idx]
    data: list[dict] = []

    for i, dt in enumerate(dates):
        # 切换市场阶段
        if phase_periods >= total_periods * current_phase["length"]:
            phase_idx = (phase_idx + 1) % len(MARKET_PHASES)
            current_phase = MARKET_PHASES[phase_idx]
            phase_periods = 0

        trend = current_phase["trend"]
        volatility = current_phase["volatility"]

        if not is_daily:
            trend, volatility = _adjust_for_intraday(trend, volatility, freq_minutes)

        cycle_factor, annual_cycle = _compute_cycle_factors(i, freq)
        noise = np.random.normal(0, volatility)

        # 计算开盘价和收盘价
        open_price = base_price
        close_price = base_price * (1 + trend + cycle_factor + annual_cycle + noise)
        if close_price <= 0:
            close_price = base_price * 0.95

        # 计算高低价
        high_price, low_price = _compute_price_range(base_price, open_price, close_price, freq)

        # 计算成交量和成交金额
        price_change_ratio = abs(close_price - open_price) / open_price
        volume = _compute_volume(freq, freq_minutes, price_change_ratio)
        avg_price = (high_price + low_price + open_price + close_price) / 4
        amount = volume * avg_price

        data.append(
            {
                "dt": dt,
                "symbol": symbol,
                "open": round(open_price, 2),
                "close": round(close_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "vol": volume,
                "amount": round(amount, 2),
            }
        )

        base_price = close_price
        phase_periods += 1

    return pd.DataFrame(data)


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
        edt: 结束日期，格式为 'YYYYMMDD'，默认 "202501
        seed: 随机数种子，确保结果可重现，默认 42

    Returns:
        包含K线数据和权重列的DataFrame，额外列包括 weight, price
    """
    np.random.seed(seed)
    frames = [mock_symbol_kline(symbol, freq=freq, sdt=sdt, edt=edt, seed=seed) for symbol in symbols]
    df = pd.concat(frames, ignore_index=True)
    df["weight"] = np.random.normal(-1, 1, len(df)).clip(-1, 1)
    df["price"] = df["close"]
    return df
