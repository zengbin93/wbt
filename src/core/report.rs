use crate::core::daily_performance::DailyPerformance;
use crate::core::evaluate_pairs::EvaluatePairs;
use crate::core::native_engine::DailyTotals;
use crate::core::period_win_rates::PeriodWinRates;
use chrono::NaiveDate;
use polars::frame::DataFrame;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;

/// 回测报告，包含品种报告、日收益率和统计指标
pub struct Report {
    pub symbols: Vec<SymbolsReport>,
    /// 品种等权日收益
    pub daily_return: DataFrame,
    pub stats: StatsReport,
    pub symbol_dict: Vec<String>,
    /// DailyTotals（用于 alpha_df / 延迟计算）
    pub daily_totals: DailyTotals,
    /// 多头统计
    pub long_stats: HashMap<String, Value>,
    /// 空头统计
    pub short_stats: HashMap<String, Value>,
}

/// 单个品种的报告，包含日收益率和交易对数据
#[derive(Serialize)]
pub struct SymbolsReport {
    pub symbol: String,
    pub daily: DataFrame,
    pub pair: DataFrame,
}

/// 统计指标报告，包含回测性能的各项指标
#[derive(Clone, Serialize)]
pub struct StatsReport {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    /// 单利计算日收益数据的各项指标
    pub daily_performance: DailyPerformance,
    pub evaluate_pairs: EvaluatePairs,
    /// 周期胜率
    pub period_win_rates: PeriodWinRates,
    /// 多头占比
    pub long_rate: f64,
    /// 空头占比
    pub short_rate: f64,
    /// 品种数量
    pub symbols_count: usize,
    /// 交易次数
    pub trade_count: usize,
    /// 年化交易次数
    pub annual_trade_count: f64,
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

        result.insert("多头统计".into(), json!(val.long_stats));
        result.insert("空头统计".into(), json!(val.short_stats));

        Value::Object(result)
    }
}

impl From<StatsReport> for Value {
    fn from(val: StatsReport) -> Self {
        let dp = val.daily_performance;
        let ep = val.evaluate_pairs;
        let pwr = val.period_win_rates;

        let mut result = serde_json::Map::new();

        // 收益
        result.insert("绝对收益".into(), json!(dp.absolute_return));
        result.insert("年化收益".into(), json!(dp.annual_returns));
        result.insert("夏普比率".into(), json!(dp.sharpe_ratio));
        result.insert("卡玛比率".into(), json!(dp.calmar_ratio));
        result.insert("新高占比".into(), json!(dp.new_high_ratio));
        result.insert("单笔盈亏比".into(), json!(ep.single_profit_loss_ratio));
        result.insert("单笔收益".into(), json!(ep.single_trade_profit));
        result.insert("日胜率".into(), json!(dp.daily_win_rate));
        result.insert("周胜率".into(), json!(pwr.week));
        result.insert("月胜率".into(), json!(pwr.month));
        result.insert("季胜率".into(), json!(pwr.quarter));
        result.insert("年胜率".into(), json!(pwr.year));

        // 风险
        result.insert("最大回撤".into(), json!(dp.max_drawdown));
        result.insert("年化波动率".into(), json!(dp.annual_volatility));
        result.insert("下行波动率".into(), json!(dp.downside_volatility));
        result.insert("新高间隔".into(), json!(dp.new_high_interval));

        // 特质
        result.insert("交易次数".into(), json!(val.trade_count));
        result.insert("年化交易次数".into(), json!(val.annual_trade_count));
        result.insert("持仓K线数".into(), json!(ep.position_k_days));
        result.insert("交易胜率".into(), json!(ep.win_rate));
        result.insert("多头占比".into(), json!(val.long_rate));
        result.insert("空头占比".into(), json!(val.short_rate));
        result.insert("品种数量".into(), json!(val.symbols_count));

        // 元数据
        result.insert("开始日期".into(), json!(val.start_date.to_string()));
        result.insert("结束日期".into(), json!(val.end_date.to_string()));

        Value::Object(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evaluate_pairs::EvaluatePairs;
    use crate::core::native_engine::DailyTotals;
    use polars::prelude::{IntoColumn, NamedFrom};

    fn make_stats_report() -> StatsReport {
        StatsReport {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 12, 31).unwrap(),
            daily_performance: DailyPerformance::default(),
            evaluate_pairs: EvaluatePairs::default(),
            period_win_rates: PeriodWinRates::default(),
            long_rate: 0.5,
            short_rate: 0.5,
            symbols_count: 3,
            trade_count: 10,
            annual_trade_count: 120.0,
        }
    }

    #[test]
    fn stats_report_to_value() {
        let stats = make_stats_report();
        let val: Value = stats.into();
        assert!(val.is_object());
        let obj = val.as_object().unwrap();
        assert_eq!(obj["开始日期"], "2024-01-01");
        assert_eq!(obj["结束日期"], "2024-12-31");
        assert_eq!(obj["品种数量"], 3);
        assert_eq!(obj["多头占比"], 0.5);
        assert_eq!(obj["交易次数"], 10);
        assert_eq!(obj["年化交易次数"], 120.0);
        assert_eq!(obj["周胜率"], 0.0);
    }

    #[test]
    fn stats_report_to_value_key_count() {
        let stats = make_stats_report();
        let val: Value = stats.into();
        let obj = val.as_object().unwrap();
        assert_eq!(
            obj.len(),
            25,
            "StatsReport JSON must have exactly 25 keys, got {}",
            obj.len()
        );
    }

    #[test]
    fn report_to_value() {
        let stats = make_stats_report();
        let daily_return = DataFrame::new(vec![
            polars::prelude::Series::new("date".into(), &[0_i32])
                .cast(&polars::prelude::DataType::Date)
                .unwrap()
                .into_column(),
            polars::prelude::Series::new("total".into(), &[0.01_f64]).into_column(),
        ])
        .unwrap();

        let report = Report {
            symbols: vec![SymbolsReport {
                symbol: "TEST".into(),
                daily: DataFrame::empty(),
                pair: DataFrame::empty(),
            }],
            daily_return,
            stats,
            symbol_dict: vec!["TEST".into()],
            daily_totals: DailyTotals {
                date_keys: vec![20240101],
                totals: vec![0.01],
                n1b_totals: vec![0.005],
                start_date_key: 20240101,
                end_date_key: 20240101,
                long_count: 1,
                short_count: 0,
                total_weight_rows: 1,
                strategy_means: vec![0.01],
                benchmark_means: vec![0.005],
            },
            long_stats: HashMap::new(),
            short_stats: HashMap::new(),
        };

        let val: Value = report.into();
        assert!(val.is_object());
        let obj = val.as_object().unwrap();
        assert!(obj.contains_key("TEST"));
        assert!(obj.contains_key("品种等权日收益"));
        assert!(obj.contains_key("绩效评价"));
        assert!(obj.contains_key("多头统计"));
        assert!(obj.contains_key("空头统计"));
    }
}
