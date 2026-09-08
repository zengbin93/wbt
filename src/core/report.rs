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
    pub trade_count: f64,
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

/// Typed values preserve Python floats (including non-finite values); JSON uses null for them.
#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum StatsValue {
    Float(f64),
    Count(usize),
    Text(String),
}

impl From<f64> for StatsValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}
impl From<usize> for StatsValue {
    fn from(value: usize) -> Self {
        Self::Count(value)
    }
}
impl From<String> for StatsValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// The same sources serve full, side, segment and alpha reports; absent groups stay absent.
pub(crate) struct StatsFields<'a> {
    pub dp: &'a DailyPerformance,
    pub pwr: &'a PeriodWinRates,
    pub ep: Option<&'a EvaluatePairs>,
    pub trade_count: Option<f64>,
    pub annual_trade_count: Option<f64>,
    pub long_rate: Option<f64>,
    pub short_rate: Option<f64>,
    pub symbols_count: Option<usize>,
    pub dates: Option<(NaiveDate, NaiveDate)>,
}

macro_rules! stats_schema {
    ($($name:literal => |$s:ident| $value:expr),* $(,)?) => {
        pub(crate) const STATS_FIELD_ORDER: &[&str] = &[$($name),*];
        impl StatsFields<'_> {
            pub(crate) fn values(&self) -> Vec<(&'static str, StatsValue)> {
                let mut values = Vec::with_capacity(STATS_FIELD_ORDER.len());
                $(if let Some(value) = (|$s: &StatsFields<'_>| -> Option<StatsValue> { $value })(self) {
                    values.push(($name, value));
                })*
                values
            }
        }
    };
}

// Canonical Chinese labels, source fields and order — shared by JSON and PyO3.
stats_schema! {
    "绝对收益" => |s| Some((s.dp.absolute_return).into()),
    "年化收益" => |s| Some((s.dp.annual_returns).into()),
    "夏普比率" => |s| Some((s.dp.sharpe_ratio).into()),
    "卡玛比率" => |s| Some((s.dp.calmar_ratio).into()),
    "新高占比" => |s| Some((s.dp.new_high_ratio).into()),
    "单笔盈亏比" => |s| Some((s.ep?.single_profit_loss_ratio).into()),
    "单笔收益" => |s| Some((s.ep?.single_trade_profit).into()),
    "日胜率" => |s| Some((s.dp.daily_win_rate).into()),
    "周胜率" => |s| Some((s.pwr.week).into()),
    "月胜率" => |s| Some((s.pwr.month).into()),
    "季胜率" => |s| Some((s.pwr.quarter).into()),
    "年胜率" => |s| Some((s.pwr.year).into()),
    "最大回撤" => |s| Some((s.dp.max_drawdown).into()),
    "年化波动率" => |s| Some((s.dp.annual_volatility).into()),
    "下行波动率" => |s| Some((s.dp.downside_volatility).into()),
    "新高间隔" => |s| Some((s.dp.new_high_interval).into()),
    "交易次数" => |s| Some((s.trade_count?).into()),
    "年化交易次数" => |s| Some((s.annual_trade_count?).into()),
    "持仓K线数" => |s| Some((s.ep?.position_k_days).into()),
    "交易胜率" => |s| Some((s.ep?.win_rate).into()),
    "多头占比" => |s| Some((s.long_rate?).into()),
    "空头占比" => |s| Some((s.short_rate?).into()),
    "品种数量" => |s| Some((s.symbols_count?).into()),
    "开始日期" => |s| Some((s.dates?.0.to_string()).into()),
    "结束日期" => |s| Some((s.dates?.1.to_string()).into()),
}

impl<'a> StatsFields<'a> {
    pub(crate) fn daily(dp: &'a DailyPerformance, pwr: &'a PeriodWinRates) -> Self {
        Self {
            dp,
            pwr,
            ep: None,
            trade_count: None,
            annual_trade_count: None,
            long_rate: None,
            short_rate: None,
            symbols_count: None,
            dates: None,
        }
    }

    pub(crate) fn to_map(&self) -> HashMap<String, Value> {
        self.values()
            .into_iter()
            .map(|(name, value)| (name.into(), json!(value)))
            .collect()
    }
}

impl StatsReport {
    pub(crate) fn fields(&self) -> StatsFields<'_> {
        StatsFields {
            dp: &self.daily_performance,
            pwr: &self.period_win_rates,
            ep: Some(&self.evaluate_pairs),
            trade_count: Some(self.trade_count),
            annual_trade_count: Some(self.annual_trade_count),
            long_rate: Some(self.long_rate),
            short_rate: Some(self.short_rate),
            symbols_count: Some(self.symbols_count),
            dates: Some((self.start_date, self.end_date)),
        }
    }
}

impl From<StatsReport> for Value {
    fn from(val: StatsReport) -> Self {
        Value::Object(val.fields().to_map().into_iter().collect())
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
            trade_count: 10.0,
            annual_trade_count: 120.0,
        }
    }

    #[test]
    fn stats_schema_preserves_sources_and_nonfinite_boundary() {
        let mut stats = make_stats_report();
        stats.daily_performance.absolute_return = 0.12;
        stats.daily_performance.annual_returns = 0.34;
        stats.daily_performance.new_high_interval = 8.0;
        stats.evaluate_pairs.single_trade_profit = 17.0;
        stats.evaluate_pairs.trade_count = 999.0; // Report count is authoritative.
        stats.period_win_rates.week = 0.75;
        let json: Value = stats.clone().into();
        assert_eq!(json["绝对收益"], 0.12);
        assert_eq!(json["年化收益"], 0.34);
        assert_eq!(json["新高间隔"], 8.0);
        assert_eq!(json["单笔收益"], 17.0);
        assert_eq!(json["交易次数"], 10.0);
        assert_eq!(json["周胜率"], 0.75);
        stats.daily_performance.annual_returns = f64::NAN;
        let fields = stats.fields().values();
        assert!(matches!(fields[1].1, StatsValue::Float(v) if v.is_nan()));
        let json: Value = stats.into();
        assert!(json["年化收益"].is_null());
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
        assert_eq!(obj["交易次数"], 10.0);
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
        let daily_return = DataFrame::new_infer_height(vec![
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
                weight_count_dates: vec![20240101],
                long_count_per_day: vec![1],
                short_count_per_day: vec![0],
                weight_rows_per_day: vec![1],
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
        let symbol_obj = obj["TEST"].as_object().unwrap();
        assert!(symbol_obj.contains_key("daily"));
        assert!(!symbol_obj.contains_key("pairs"));
        assert!(obj.contains_key("品种等权日收益"));
        assert!(obj.contains_key("绩效评价"));
        assert!(obj.contains_key("多头统计"));
        assert!(obj.contains_key("空头统计"));
    }
}
