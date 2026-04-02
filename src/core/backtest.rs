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

        assert!(wb.report.is_some());
        let report = wb.report.as_ref().unwrap();
        assert!(!report.symbol_dict.is_empty());
        assert!(!report.daily_return.is_empty());

        let dailys = wb.dailys_df().unwrap();
        assert!(dailys.height() > 0);
        assert!(dailys.column("symbol").is_ok());
        assert!(dailys.column("return").is_ok());

        let alpha = wb.alpha_df().unwrap();
        assert!(alpha.column("date").is_ok());
        assert!(alpha.column("超额").is_ok());
        assert!(alpha.column("策略").is_ok());
        assert!(alpha.column("基准").is_ok());
    }

    #[test]
    fn backtest_full_flow_cs() {
        let df = make_test_dataframe();
        let mut wb = WeightBacktest::new(df, 2, Some(0.0002)).unwrap();
        wb.backtest(Some(1), WeightType::CS, 252).unwrap();
        assert!(wb.report.is_some());
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
        let _ = wb.pairs_df().unwrap();
    }
}
