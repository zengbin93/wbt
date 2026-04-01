use crate::daily_performance::daily_performance;
use crate::native_engine::{DailyTotals, DailysSoA, PairsSoA};
use crate::trade_dir::TradeDir;
use crate::utils::{RoundToNthDigit, date_key_to_naive_date, pearson_corr_inline, std_inline};
use crate::{
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
        crate::native_engine::NativeEngine::process_all(
            &self.dfw,
            &symbols,
            self.digits,
            self.fee_rate,
            weight_type_is_ts,
        )
    }
}
