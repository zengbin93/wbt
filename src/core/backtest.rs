use crate::core::daily_performance::daily_performance;
use crate::core::native_engine::{DailyTotals, DailysSoA, PairsSoA};
use crate::core::trade_dir::TradeDir;
use crate::core::utils::{
    RoundToNthDigit, date_key_to_naive_date, pearson_corr_inline, std_inline,
};
use crate::core::{
    WeightBacktest,
    errors::WbtError,
    evaluate_pairs::evaluate_pairs_soa,
    report::{Report, StatsReport, SymbolsReport},
    utils::WeightType,
};
use anyhow::Context;
use polars::prelude::*;

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

        let mut per_symbol = vec![vec![None; daily_totals.date_keys.len()]; dailys_soa.symbol_dict.len()];
        for i in 0..dailys_soa.sym_ids.len() {
            let date_key = crate::core::native_engine::dt_to_date_key_fast(
                dailys_soa.date_ticks[i],
                dailys_soa.time_unit,
            );
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
        for (sym, values) in dailys_soa.symbol_dict.iter().zip(per_symbol.into_iter()) {
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

        let total_rows = daily_totals.total_weight_rows as f64;
        let (long_rate, short_rate) = if total_rows > 0.0 {
            (
                (daily_totals.long_count as f64 / total_rows).round_to_4_digit(),
                (daily_totals.short_count as f64 / total_rows).round_to_4_digit(),
            )
        } else {
            (0.0, 0.0)
        };

        let strategy: &[f64] = &daily_totals.strategy_means;
        let benchmark: &[f64] = &daily_totals.benchmark_means;
        let n_days = strategy.len();

        let strategy_std = std_inline(strategy);
        let benchmark_std = std_inline(benchmark);
        let volatility_ratio = if benchmark_std > 0.0 {
            (strategy_std / benchmark_std).round_to_4_digit()
        } else {
            0.0
        };

        let benchmark_abs: Vec<f64> = benchmark.iter().map(|x| x.abs()).collect();
        let relevance_volatility = pearson_corr_inline(strategy, &benchmark_abs).round_to_4_digit();
        let relevance = pearson_corr_inline(strategy, benchmark).round_to_4_digit();

        let mut short_strategy = Vec::with_capacity(n_days / 3);
        let mut short_benchmark = Vec::with_capacity(n_days / 3);
        for i in 0..n_days {
            if benchmark[i] < 0.0 {
                short_strategy.push(strategy[i]);
                short_benchmark.push(benchmark[i]);
            }
        }
        let relevance_short =
            pearson_corr_inline(&short_strategy, &short_benchmark).round_to_4_digit();

        let stats = StatsReport {
            start_date,
            end_date,
            daily_performance: dp,
            evaluate_pairs: ep,
            long_rate,
            short_rate,
            volatility_ratio,
            relevance_volatility,
            relevance,
            relevance_short,
            symbols_count: self.symbols.len(),
        };

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
        self.dailys_soa = Some(dailys_soa);
        self.pairs_soa = Some(pairs_soa);

        self.report = Some(Report {
            symbols: symbols_report,
            daily_return: daily_return_df,
            stats,
            symbol_dict,
            daily_totals,
        });

        Ok(())
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
        let excess: Vec<f64> = alpha.column("超额").unwrap().as_materialized_series().f64().unwrap().into_no_null_iter().collect();
        let strategy: Vec<f64> = alpha.column("策略").unwrap().as_materialized_series().f64().unwrap().into_no_null_iter().collect();
        let benchmark: Vec<f64> = alpha.column("基准").unwrap().as_materialized_series().f64().unwrap().into_no_null_iter().collect();
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
}
