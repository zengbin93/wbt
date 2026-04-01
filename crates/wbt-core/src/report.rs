use crate::daily_performance::DailyPerformance;
use crate::evaluate_pairs::EvaluatePairs;
use crate::native_engine::DailyTotals;
use chrono::NaiveDate;
use polars::frame::DataFrame;
use serde::Serialize;
use serde_json::{Value, json};

/// 回测报告，包含品种报告、日收益率和统计指标
pub struct Report {
    pub symbols: Vec<SymbolsReport>,
    /// 品种等权日收益
    pub daily_return: DataFrame,
    pub stats: StatsReport,
    pub symbol_dict: Vec<String>,
    /// DailyTotals（用于 alpha_df / 延迟计算）
    pub daily_totals: DailyTotals,
}

/// 单个品种的报告，包含日收益率和交易对数据
#[derive(Serialize)]
pub struct SymbolsReport {
    pub symbol: String,
    pub daily: DataFrame,
    pub pair: DataFrame,
}

/// 统计指标报告，包含回测性能的各项指标
#[derive(Serialize)]
pub struct StatsReport {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    /// 单利计算日收益数据的各项指标
    pub daily_performance: DailyPerformance,
    pub evaluate_pairs: EvaluatePairs,
    /// 多头占比
    pub long_rate: f64,
    /// 空头占比
    pub short_rate: f64,
    /// 与基准相关性
    pub relevance: f64,
    /// 与基准空头相关性
    pub relevance_short: f64,
    /// 波动比
    pub volatility_ratio: f64,
    /// 与基准波动相关性
    pub relevance_volatility: f64,
    /// 品种数量
    pub symbols_count: usize,
}

impl From<Report> for Value {
    fn from(val: Report) -> Self {
        let mut result = serde_json::Map::new();

        for symbol in val.symbols {
            result.insert(
                symbol.symbol,
                json!({
                    "daily": symbol.daily,
                    "pairs": symbol.pair,
                }),
            );
        }

        result.insert("品种等权日收益".into(), json!(val.daily_return));

        result.insert("绩效评价".into(), val.stats.into());

        Value::Object(result)
    }
}

impl From<StatsReport> for Value {
    fn from(val: StatsReport) -> Self {
        let dp = val.daily_performance;
        let ep = val.evaluate_pairs;

        json!({
            "开始日期": val.start_date.to_string(),
            "结束日期": val.end_date.to_string(),
            "绝对收益": dp.absolute_return,
            "年化收益": dp.annual_returns,
            "夏普比率": dp.sharpe_ratio,
            "最大回撤": dp.max_drawdown,
            "卡玛比率": dp.calmar_ratio,
            "日胜率": dp.daily_win_rate,
            "日盈亏比": dp.daily_profit_loss_ratio,
            "日赢面": dp.daily_win_probability,
            "年化波动率": dp.annual_volatility,
            "下行波动率": dp.downside_volatility,
            "非零覆盖": dp.non_zero_coverage,
            "盈亏平衡点": dp.break_even_point,
            "新高间隔": dp.new_high_interval,
            "新高占比": dp.new_high_ratio,
            "回撤风险": dp.drawdown_risk,
            "单笔收益": ep.single_trade_profit,
            "持仓K线数": ep.position_k_days,
            "多头占比": val.long_rate,
            "空头占比": val.short_rate,
            "与基准相关性": val.relevance,
            "与基准空头相关性": val.relevance_short,
            "波动比": val.volatility_ratio,
            "与基准波动相关性": val.relevance_volatility,
            "品种数量": val.symbols_count,
        })
    }
}
