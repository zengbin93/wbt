use crate::core::daily_performance::daily_performance;
use crate::core::native_engine::{DailyTotals, DailysSoA, PairsSoA, dt_to_date_key_fast};
use crate::core::period_win_rates::period_win_rates;
use crate::core::trade_dir::TradeDir;
use crate::core::utils::{RoundToNthDigit, date_key_to_naive_date, std_inline};
use crate::core::{
    WeightBacktest,
    errors::WbtError,
    evaluate_pairs::evaluate_pairs_soa,
    report::{Report, StatsReport, SymbolsReport},
    utils::WeightType,
};
use anyhow::Context;
use polars::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: aggregate long/short returns from DailysSoA per date
// ---------------------------------------------------------------------------

/// Returns (long_returns, short_returns) aggregated per date, aligned with daily_totals.date_keys.
fn aggregate_long_short_returns(
    dailys_soa: &DailysSoA,
    daily_totals: &DailyTotals,
    weight_type: WeightType,
) -> (Vec<f64>, Vec<f64>) {
    let n_dates = daily_totals.date_keys.len();
    let mut row_by_date = hashbrown::HashMap::with_capacity(n_dates);
    for (row, &dk) in daily_totals.date_keys.iter().enumerate() {
        row_by_date.insert(dk, row);
    }

    let mut long_sum = vec![0.0f64; n_dates];
    let mut short_sum = vec![0.0f64; n_dates];
    let mut long_count = vec![0usize; n_dates];
    let mut short_count = vec![0usize; n_dates];

    for i in 0..dailys_soa.sym_ids.len() {
        let dk = dt_to_date_key_fast(dailys_soa.date_ticks[i], dailys_soa.time_unit);
        if let Some(&row) = row_by_date.get(&dk) {
            long_sum[row] += dailys_soa.long_return[i];
            short_sum[row] += dailys_soa.short_return[i];
            long_count[row] += 1;
            short_count[row] += 1;
        }
    }

    match weight_type {
        WeightType::TS => {
            for row in 0..n_dates {
                if long_count[row] > 0 {
                    long_sum[row] /= long_count[row] as f64;
                }
                if short_count[row] > 0 {
                    short_sum[row] /= short_count[row] as f64;
                }
            }
        }
        WeightType::CS => { /* sum is already what we want */ }
    }

    (long_sum, short_sum)
}

// ---------------------------------------------------------------------------
// Helper: build stats dict from returns + pairs
// ---------------------------------------------------------------------------

fn build_stats_dict(
    date_keys: &[i32],
    returns: &[f64],
    pairs_soa: &PairsSoA,
    trade_dir: TradeDir,
    yearly_days: usize,
) -> Result<HashMap<String, Value>, WbtError> {
    let dp = daily_performance(returns, Some(yearly_days))?;
    let ep = evaluate_pairs_soa(pairs_soa, trade_dir)?;
    let pwr = period_win_rates(date_keys, returns, yearly_days as i64);

    let n_dates = date_keys.len();
    let annual_trade_count = if n_dates > 0 {
        (ep.trade_count as f64 / (n_dates as f64 / yearly_days as f64)).round_to_2_digit()
    } else {
        0.0
    };

    let mut m = HashMap::new();
    // 收益
    m.insert("绝对收益".into(), json!(dp.absolute_return));
    m.insert("年化收益".into(), json!(dp.annual_returns));
    m.insert("夏普比率".into(), json!(dp.sharpe_ratio));
    m.insert("卡玛比率".into(), json!(dp.calmar_ratio));
    m.insert("新高占比".into(), json!(dp.new_high_ratio));
    m.insert("单笔盈亏比".into(), json!(ep.single_profit_loss_ratio));
    m.insert("单笔收益".into(), json!(ep.single_trade_profit));
    m.insert("日胜率".into(), json!(dp.daily_win_rate));
    m.insert("周胜率".into(), json!(pwr.week));
    m.insert("月胜率".into(), json!(pwr.month));
    m.insert("季胜率".into(), json!(pwr.quarter));
    m.insert("年胜率".into(), json!(pwr.year));
    // 风险
    m.insert("最大回撤".into(), json!(dp.max_drawdown));
    m.insert("年化波动率".into(), json!(dp.annual_volatility));
    m.insert("下行波动率".into(), json!(dp.downside_volatility));
    m.insert("新高间隔".into(), json!(dp.new_high_interval));
    // 特质
    m.insert("交易次数".into(), json!(ep.trade_count));
    m.insert("年化交易次数".into(), json!(annual_trade_count));
    m.insert("持仓K线数".into(), json!(ep.position_k_days));
    m.insert("交易胜率".into(), json!(ep.win_rate));
    Ok(m)
}

// ---------------------------------------------------------------------------
// Helper: filter PairsSoA by date range
// ---------------------------------------------------------------------------

fn filter_pairs_by_date(pairs: &PairsSoA, sdt: i32, edt: i32) -> PairsSoA {
    let mut indices = Vec::new();
    for i in 0..pairs.sym_ids.len() {
        let open_dk = dt_to_date_key_fast(pairs.open_dts[i], pairs.time_unit);
        let close_dk = dt_to_date_key_fast(pairs.close_dts[i], pairs.time_unit);
        if open_dk >= sdt && close_dk <= edt {
            indices.push(i);
        }
    }

    PairsSoA {
        sym_ids: indices.iter().map(|&i| pairs.sym_ids[i]).collect(),
        dirs: indices.iter().map(|&i| pairs.dirs[i]).collect(),
        open_dts: indices.iter().map(|&i| pairs.open_dts[i]).collect(),
        close_dts: indices.iter().map(|&i| pairs.close_dts[i]).collect(),
        open_prices: indices.iter().map(|&i| pairs.open_prices[i]).collect(),
        close_prices: indices.iter().map(|&i| pairs.close_prices[i]).collect(),
        hold_bars: indices.iter().map(|&i| pairs.hold_bars[i]).collect(),
        event_seqs: indices.iter().map(|&i| pairs.event_seqs[i]).collect(),
        profit_bps: indices.iter().map(|&i| pairs.profit_bps[i]).collect(),
        counts: indices.iter().map(|&i| pairs.counts[i]).collect(),
        time_unit: pairs.time_unit,
        symbol_dict: pairs.symbol_dict.clone(),
    }
}

// ---------------------------------------------------------------------------
// WeightBacktest impl
// ---------------------------------------------------------------------------

impl WeightBacktest {
    pub(crate) fn build_daily_return_df(
        dailys_soa: &DailysSoA,
        daily_totals: &DailyTotals,
        weight_type: WeightType,
    ) -> Result<DataFrame, WbtError> {
        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let dr_dates: Vec<i32> = daily_totals
            .date_keys
            .iter()
            .map(|dk| {
                let nd = date_key_to_naive_date(*dk);
                (nd - epoch).num_days() as i32
            })
            .collect();

        let mut row_by_date = hashbrown::HashMap::with_capacity(daily_totals.date_keys.len());
        for (row, date_key) in daily_totals.date_keys.iter().copied().enumerate() {
            row_by_date.insert(date_key, row);
        }

        let mut per_symbol =
            vec![vec![None; daily_totals.date_keys.len()]; dailys_soa.symbol_dict.len()];
        for i in 0..dailys_soa.sym_ids.len() {
            let date_key = dt_to_date_key_fast(dailys_soa.date_ticks[i], dailys_soa.time_unit);
            if let Some(&row) = row_by_date.get(&date_key) {
                per_symbol[dailys_soa.sym_ids[i] as usize][row] = Some(dailys_soa.ret[i]);
            }
        }

        let mut total_values = Vec::with_capacity(daily_totals.date_keys.len());
        for row in 0..daily_totals.date_keys.len() {
            let mut sum = 0.0;
            let mut count = 0usize;
            for sym_values in &per_symbol {
                if let Some(value) = sym_values[row] {
                    sum += value;
                    count += 1;
                }
            }
            let total = match weight_type {
                WeightType::TS => {
                    if count > 0 {
                        sum / count as f64
                    } else {
                        0.0
                    }
                }
                WeightType::CS => sum,
            };
            total_values.push(total);
        }

        let mut columns = Vec::with_capacity(dailys_soa.symbol_dict.len() + 2);
        columns.push(
            Series::new("date".into(), dr_dates)
                .cast(&DataType::Date)
                .map_err(WbtError::Polars)?
                .into_column(),
        );
        for (sym, values) in dailys_soa.symbol_dict.iter().zip(per_symbol) {
            columns.push(Series::new(sym.as_str().into(), values).into_column());
        }
        columns.push(Series::new("total".into(), total_values).into_column());

        DataFrame::new(columns).map_err(WbtError::Polars)
    }

    /// 执行回测逻辑并计算性能指标
    pub fn do_backtest(
        &mut self,
        weight_type: WeightType,
        yearly_days: usize,
    ) -> Result<(), WbtError> {
        let (symbols_report, dailys_soa, pairs_soa, daily_totals, symbol_dict) = self
            .process_symbols(weight_type)
            .context("Failed to process symbols in parallel")?;

        let start_date = date_key_to_naive_date(daily_totals.start_date_key);
        let end_date = date_key_to_naive_date(daily_totals.end_date_key);

        let dp = daily_performance(&daily_totals.totals, Some(yearly_days))?;

        let ep = evaluate_pairs_soa(&pairs_soa, TradeDir::LongShort)?;

        let pwr = period_win_rates(
            &daily_totals.date_keys,
            &daily_totals.totals,
            yearly_days as i64,
        );

        let total_rows = daily_totals.total_weight_rows as f64;
        let (long_rate, short_rate) = if total_rows > 0.0 {
            (
                (daily_totals.long_count as f64 / total_rows).round_to_4_digit(),
                (daily_totals.short_count as f64 / total_rows).round_to_4_digit(),
            )
        } else {
            (0.0, 0.0)
        };

        let n_dates = daily_totals.date_keys.len();
        let annual_trade_count = if n_dates > 0 {
            (ep.trade_count as f64 / (n_dates as f64 / yearly_days as f64)).round_to_2_digit()
        } else {
            0.0
        };

        let trade_count = ep.trade_count;

        let stats = StatsReport {
            start_date,
            end_date,
            daily_performance: dp,
            evaluate_pairs: ep,
            period_win_rates: pwr,
            long_rate,
            short_rate,
            symbols_count: self.symbols.len(),
            trade_count,
            annual_trade_count,
        };

        // Build long/short stats
        let (long_returns, short_returns) =
            aggregate_long_short_returns(&dailys_soa, &daily_totals, weight_type);

        let long_stats = build_stats_dict(
            &daily_totals.date_keys,
            &long_returns,
            &pairs_soa,
            TradeDir::Long,
            yearly_days,
        )?;
        let short_stats = build_stats_dict(
            &daily_totals.date_keys,
            &short_returns,
            &pairs_soa,
            TradeDir::Short,
            yearly_days,
        )?;

        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let dr_dates: Vec<i32> = daily_totals
            .date_keys
            .iter()
            .map(|dk| {
                let nd = date_key_to_naive_date(*dk);
                (nd - epoch).num_days() as i32
            })
            .collect();
        let daily_return_df = DataFrame::new(vec![
            Series::new("date".into(), dr_dates)
                .cast(&DataType::Date)
                .map_err(WbtError::Polars)?
                .into_column(),
            Series::new("total".into(), daily_totals.totals.clone()).into_column(),
        ])
        .map_err(WbtError::Polars)?;

        self.daily_return_cache = None;
        self.dailys_cache = None;
        self.pairs_cache = None;
        self.weight_type = Some(weight_type);
        self.yearly_days = yearly_days;
        self.dailys_soa = Some(dailys_soa);
        self.pairs_soa = Some(pairs_soa);

        self.report = Some(Report {
            symbols: symbols_report,
            daily_return: daily_return_df,
            stats,
            symbol_dict,
            daily_totals,
            long_stats,
            short_stats,
        });

        Ok(())
    }

    // -----------------------------------------------------------------------
    // segment_stats
    // -----------------------------------------------------------------------

    /// 对指定日期范围 [sdt, edt] 做分段统计。
    /// kind: "多空" | "多头" | "空头"
    pub fn segment_stats(
        &self,
        sdt: Option<i32>,
        edt: Option<i32>,
        kind: &str,
    ) -> Result<HashMap<String, Value>, WbtError> {
        let dailys_soa = self
            .dailys_soa
            .as_ref()
            .ok_or_else(|| WbtError::NoneValue("dailys_soa not computed yet".into()))?;
        let pairs_soa = self
            .pairs_soa
            .as_ref()
            .ok_or_else(|| WbtError::NoneValue("pairs_soa not computed yet".into()))?;
        let report = self
            .report
            .as_ref()
            .ok_or_else(|| WbtError::NoneValue("report not computed yet".into()))?;
        let weight_type = self
            .weight_type
            .ok_or_else(|| WbtError::NoneValue("weight_type not set".into()))?;

        let daily_totals = &report.daily_totals;

        let actual_sdt = sdt.unwrap_or(daily_totals.start_date_key);
        let actual_edt = edt.unwrap_or(daily_totals.end_date_key);

        // Filter dailys by date range and aggregate returns per date
        let n_dates = daily_totals.date_keys.len();
        let mut row_by_date = hashbrown::HashMap::with_capacity(n_dates);
        for (row, &dk) in daily_totals.date_keys.iter().enumerate() {
            row_by_date.insert(dk, row);
        }

        // Collect per-date sums based on kind
        let mut date_sum = vec![0.0f64; n_dates];
        let mut date_count = vec![0usize; n_dates];

        for i in 0..dailys_soa.sym_ids.len() {
            let dk = dt_to_date_key_fast(dailys_soa.date_ticks[i], dailys_soa.time_unit);
            if dk < actual_sdt || dk > actual_edt {
                continue;
            }
            if let Some(&row) = row_by_date.get(&dk) {
                let val = match kind {
                    "多头" => dailys_soa.long_return[i],
                    "空头" => dailys_soa.short_return[i],
                    _ => dailys_soa.ret[i], // "多空" or default
                };
                date_sum[row] += val;
                date_count[row] += 1;
            }
        }

        // Build filtered date_keys and returns
        let mut filtered_date_keys = Vec::new();
        let mut filtered_returns = Vec::new();
        for row in 0..n_dates {
            let dk = daily_totals.date_keys[row];
            if dk < actual_sdt || dk > actual_edt {
                continue;
            }
            if date_count[row] == 0 {
                continue;
            }
            let val = match weight_type {
                WeightType::TS => date_sum[row] / date_count[row] as f64,
                WeightType::CS => date_sum[row],
            };
            filtered_date_keys.push(dk);
            filtered_returns.push(val);
        }

        // Filter pairs
        let trade_dir = match kind {
            "多头" => TradeDir::Long,
            "空头" => TradeDir::Short,
            _ => TradeDir::LongShort,
        };
        let filtered_pairs = filter_pairs_by_date(pairs_soa, actual_sdt, actual_edt);

        build_stats_dict(
            &filtered_date_keys,
            &filtered_returns,
            &filtered_pairs,
            trade_dir,
            self.yearly_days,
        )
    }

    // -----------------------------------------------------------------------
    // long_alpha_stats
    // -----------------------------------------------------------------------

    /// 波动率调整后的多头超额收益统计
    pub fn long_alpha_stats(&self) -> Result<HashMap<String, Value>, WbtError> {
        let dailys_soa = self
            .dailys_soa
            .as_ref()
            .ok_or_else(|| WbtError::NoneValue("dailys_soa not computed yet".into()))?;
        let report = self
            .report
            .as_ref()
            .ok_or_else(|| WbtError::NoneValue("report not computed yet".into()))?;
        let weight_type = self
            .weight_type
            .ok_or_else(|| WbtError::NoneValue("weight_type not set".into()))?;

        let daily_totals = &report.daily_totals;
        let yearly_days = self.yearly_days;

        let (long_returns, _) = aggregate_long_short_returns(dailys_soa, daily_totals, weight_type);
        let bench_returns = &daily_totals.benchmark_means;

        let yd_sqrt = (yearly_days as f64).sqrt();
        let long_vol = std_inline(&long_returns) * yd_sqrt;
        let bench_vol = std_inline(bench_returns) * yd_sqrt;

        // If either vol is near zero, return default zeros
        if long_vol < 1e-12 || bench_vol < 1e-12 {
            let dp = daily_performance(&[], None)?;
            let pwr = period_win_rates(&[], &[], yearly_days as i64);
            let mut m = HashMap::new();
            m.insert("绝对收益".into(), json!(dp.absolute_return));
            m.insert("年化收益".into(), json!(dp.annual_returns));
            m.insert("夏普比率".into(), json!(dp.sharpe_ratio));
            m.insert("卡玛比率".into(), json!(dp.calmar_ratio));
            m.insert("新高占比".into(), json!(dp.new_high_ratio));
            m.insert("日胜率".into(), json!(dp.daily_win_rate));
            m.insert("周胜率".into(), json!(pwr.week));
            m.insert("月胜率".into(), json!(pwr.month));
            m.insert("季胜率".into(), json!(pwr.quarter));
            m.insert("年胜率".into(), json!(pwr.year));
            m.insert("最大回撤".into(), json!(dp.max_drawdown));
            m.insert("年化波动率".into(), json!(dp.annual_volatility));
            m.insert("下行波动率".into(), json!(dp.downside_volatility));
            m.insert("新高间隔".into(), json!(dp.new_high_interval));
            return Ok(m);
        }

        let target_vol = 0.20;
        let long_scale = target_vol / long_vol;
        let bench_scale = target_vol / bench_vol;

        let alpha_daily: Vec<f64> = long_returns
            .iter()
            .zip(bench_returns.iter())
            .map(|(&lr, &br)| lr * long_scale - br * bench_scale)
            .collect();

        let dp = daily_performance(&alpha_daily, Some(yearly_days))?;
        let pwr = period_win_rates(&daily_totals.date_keys, &alpha_daily, yearly_days as i64);

        let mut m = HashMap::new();
        m.insert("绝对收益".into(), json!(dp.absolute_return));
        m.insert("年化收益".into(), json!(dp.annual_returns));
        m.insert("夏普比率".into(), json!(dp.sharpe_ratio));
        m.insert("卡玛比率".into(), json!(dp.calmar_ratio));
        m.insert("新高占比".into(), json!(dp.new_high_ratio));
        m.insert("日胜率".into(), json!(dp.daily_win_rate));
        m.insert("周胜率".into(), json!(pwr.week));
        m.insert("月胜率".into(), json!(pwr.month));
        m.insert("季胜率".into(), json!(pwr.quarter));
        m.insert("年胜率".into(), json!(pwr.year));
        m.insert("最大回撤".into(), json!(dp.max_drawdown));
        m.insert("年化波动率".into(), json!(dp.annual_volatility));
        m.insert("下行波动率".into(), json!(dp.downside_volatility));
        m.insert("新高间隔".into(), json!(dp.new_high_interval));
        Ok(m)
    }

    /// 并行处理所有交易品种，返回 SoA 而非 DataFrame
    #[allow(clippy::type_complexity)]
    fn process_symbols(
        &self,
        weight_type: WeightType,
    ) -> Result<
        (
            Vec<SymbolsReport>,
            DailysSoA,
            PairsSoA,
            DailyTotals,
            Vec<String>,
        ),
        WbtError,
    > {
        let symbols: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
        let weight_type_is_ts = matches!(weight_type, WeightType::TS);
        crate::core::native_engine::NativeEngine::process_all(
            &self.dfw,
            &symbols,
            self.digits,
            self.fee_rate,
            weight_type_is_ts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_dataframe() -> DataFrame {
        let n = 20;
        let dates: Vec<String> = (0..10)
            .flat_map(|d| {
                vec![
                    format!("2024-01-{:02} 09:30:00", d + 1),
                    format!("2024-01-{:02} 09:30:00", d + 1),
                ]
            })
            .collect();
        let symbols: Vec<&str> = (0..10).flat_map(|_| vec!["SYM_A", "SYM_B"]).collect();
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                let cycle = (i / 2) as f64;
                if i % 2 == 0 {
                    (cycle * 0.1 - 0.2).clamp(-1.0, 1.0)
                } else {
                    (-cycle * 0.15 + 0.3).clamp(-1.0, 1.0)
                }
            })
            .collect();
        let prices: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.5 + ((i as f64) * 0.7).sin())
            .collect();

        df! {
            "dt" => dates,
            "symbol" => symbols,
            "weight" => weights,
            "price" => prices
        }
        .unwrap()
    }

    #[test]
    fn backtest_full_flow_ts() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::TS, 252).unwrap();

        let abs_ret = {
            let report = wb.report.as_ref().unwrap();
            // 2 symbols
            assert_eq!(report.symbol_dict.len(), 2);
            assert!(report.symbol_dict.contains(&"SYM_A".to_string()));
            assert!(report.symbol_dict.contains(&"SYM_B".to_string()));
            assert_eq!(report.stats.symbols_count, 2);

            // long_rate + short_rate should be <= 1.0 and >= 0.0
            assert!(report.stats.long_rate >= 0.0 && report.stats.long_rate <= 1.0);
            assert!(report.stats.short_rate >= 0.0 && report.stats.short_rate <= 1.0);

            // New fields exist (trade_count is usize, always >= 0)
            let _ = report.stats.trade_count;
            assert!(report.stats.annual_trade_count >= 0.0);

            // long_stats and short_stats populated
            assert!(report.long_stats.contains_key("年化收益"));
            assert!(report.short_stats.contains_key("年化收益"));

            report.stats.daily_performance.absolute_return
        };

        let daily_return = wb.daily_return_df().unwrap();
        // daily_return should expose per-symbol daily returns plus total
        assert_eq!(daily_return.width(), 4);
        assert!(daily_return.column("SYM_A").is_ok());
        assert!(daily_return.column("SYM_B").is_ok());
        assert!(daily_return.height() > 0);
        let sym_a: Vec<f64> = daily_return
            .column("SYM_A")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let sym_b: Vec<f64> = daily_return
            .column("SYM_B")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let total: Vec<f64> = daily_return
            .column("total")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        for i in 0..total.len() {
            assert!(
                (total[i] - (sym_a[i] + sym_b[i]) / 2.0).abs() < 1e-10,
                "total[{i}] should equal mean(symbol returns)"
            );
        }
        // total column values should sum to absolute_return (within rounding)
        let total_sum: f64 = daily_return
            .column("total")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .sum()
            .unwrap();
        assert!(
            (total_sum - abs_ret).abs() < 0.01,
            "daily total sum {total_sum} should ≈ absolute_return {abs_ret}"
        );

        // dailys_df should have 15 columns and rows = days * symbols
        let dailys = wb.dailys_df().unwrap();
        assert_eq!(dailys.width(), 15);
        assert!(dailys.height() > 0);

        // alpha should have 4 columns and same rows as daily_return
        let alpha = wb.alpha_df().unwrap();
        assert_eq!(alpha.width(), 4);
        // alpha excess = strategy - benchmark
        let excess: Vec<f64> = alpha
            .column("超额")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let strategy: Vec<f64> = alpha
            .column("策略")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let benchmark: Vec<f64> = alpha
            .column("基准")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        for i in 0..excess.len() {
            assert!(
                (excess[i] - (strategy[i] - benchmark[i])).abs() < 1e-10,
                "alpha excess[{i}] should equal strategy - benchmark"
            );
        }
    }

    #[test]
    fn backtest_full_flow_cs() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::CS, 252).unwrap();
        let report = wb.report.as_ref().unwrap();
        assert_eq!(report.stats.symbols_count, 2);
        // CS mode: total = sum (not mean), so daily values should differ from TS
        assert!(report.daily_return.height() > 0);
    }

    #[test]
    fn backtest_ts_vs_cs_differ() {
        let df1 = make_test_dataframe();
        let df2 = make_test_dataframe();
        let mut wb_ts = WeightBacktest::new(df1, 2, Some(0.0002)).unwrap();
        let mut wb_cs = WeightBacktest::new(df2, 2, Some(0.0002)).unwrap();
        wb_ts.backtest(Some(1), WeightType::TS, 252).unwrap();
        wb_cs.backtest(Some(1), WeightType::CS, 252).unwrap();
        let ts_total = &wb_ts.report.as_ref().unwrap().daily_return;
        let cs_total = &wb_cs.report.as_ref().unwrap().daily_return;
        assert_ne!(
            ts_total
                .column("total")
                .unwrap()
                .as_materialized_series()
                .f64()
                .unwrap()
                .sum(),
            cs_total
                .column("total")
                .unwrap()
                .as_materialized_series()
                .f64()
                .unwrap()
                .sum(),
        );
    }

    #[test]
    fn backtest_pairs_df() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::TS, 252).unwrap();
        let pairs = wb.pairs_df().unwrap().unwrap();
        assert!(pairs.height() > 0);
        assert!(pairs.column("symbol").is_ok());
        assert!(pairs.column("交易方向").is_ok());
    }

    #[test]
    fn segment_stats_full_range() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::TS, 252).unwrap();
        // Full range segment should produce valid stats
        let stats = wb.segment_stats(None, None, "多空").unwrap();
        assert!(stats.contains_key("年化收益"));
        assert!(stats.contains_key("交易次数"));
    }

    #[test]
    fn segment_stats_long_only() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::TS, 252).unwrap();
        let stats = wb.segment_stats(None, None, "多头").unwrap();
        assert!(stats.contains_key("年化收益"));
    }

    #[test]
    fn long_alpha_stats_runs() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::TS, 252).unwrap();
        let alpha = wb.long_alpha_stats().unwrap();
        assert!(alpha.contains_key("年化收益"));
        assert!(alpha.contains_key("周胜率"));
    }

    #[test]
    fn yearly_days_stored() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        assert_eq!(wb.yearly_days, 252); // default
        wb.backtest(Some(1), WeightType::TS, 365).unwrap();
        assert_eq!(wb.yearly_days, 365);
    }

    /// CS mode: total column = sum (not mean) of per-symbol columns for each row.
    #[test]
    fn backtest_cs_mode_total_is_sum() {
        // Build a 2-symbol, 5-day DataFrame with known weights and prices
        let dates: Vec<String> = (0..5)
            .flat_map(|d| {
                vec![
                    format!("2024-01-{:02} 09:30:00", d + 1),
                    format!("2024-01-{:02} 09:30:00", d + 1),
                ]
            })
            .collect();
        let symbols: Vec<&str> = (0..5).flat_map(|_| vec!["A", "B"]).collect();
        let weights: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { 0.3 } else { -0.2 })
            .collect();
        let prices: Vec<f64> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    100.0 + (i / 2) as f64 * 0.5
                } else {
                    150.0 - (i / 2) as f64 * 0.3
                }
            })
            .collect();

        let df = df! {
            "dt" => dates,
            "symbol" => symbols,
            "weight" => weights,
            "price" => prices
        }
        .unwrap();

        let mut wb = WeightBacktest::new(df, 2, Some(0.0)).unwrap();
        wb.backtest(Some(1), WeightType::CS, 252).unwrap();

        let dr = wb.daily_return_df().unwrap();
        let sym_a: Vec<Option<f64>> = dr
            .column("A")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_iter()
            .collect();
        let sym_b: Vec<Option<f64>> = dr
            .column("B")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_iter()
            .collect();
        let total: Vec<Option<f64>> = dr
            .column("total")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_iter()
            .collect();

        for i in 0..total.len() {
            let a = sym_a[i].unwrap_or(0.0);
            let b = sym_b[i].unwrap_or(0.0);
            let t = total[i].unwrap_or(0.0);
            assert!(
                (t - (a + b)).abs() < 1e-10,
                "CS mode: total[{i}]={t} should equal A[{i}]({a}) + B[{i}]({b})"
            );
        }
    }

    /// Invalid kind in segment_stats should fall back to "多空" (default) behaviour.
    #[test]
    fn segment_stats_invalid_kind_defaults_to_all() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::TS, 252).unwrap();

        let stats_default = wb.segment_stats(None, None, "多空").unwrap();
        let stats_invalid = wb.segment_stats(None, None, "invalid_kind").unwrap();

        // Both should produce the same 年化收益 value since invalid falls back to ret
        let default_ret = stats_default["年化收益"].as_f64().unwrap_or(0.0);
        let invalid_ret = stats_invalid["年化收益"].as_f64().unwrap_or(0.0);
        assert!(
            (default_ret - invalid_ret).abs() < 1e-10,
            "invalid kind should produce same result as 多空: {default_ret} vs {invalid_ret}"
        );
    }
}
