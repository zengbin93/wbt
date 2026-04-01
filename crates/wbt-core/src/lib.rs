use crate::native_engine::{DailysSoA, PairsSoA};
use anyhow::Context;
use errors::WbtError;
use polars::prelude::*;
use report::Report;

mod backtest;
pub mod daily_performance;
pub mod errors;
mod evaluate_pairs;
pub mod native_engine;
mod report;
pub mod trade_dir;
pub mod utils;

pub use utils::WeightType;

/// 持仓权重回测
pub struct WeightBacktest {
    pub dfw: DataFrame,
    pub digits: i64,
    pub fee_rate: f64,
    pub symbols: Vec<Arc<str>>,
    /// 原始 SoA 数据（延迟物化）
    dailys_soa: Option<DailysSoA>,
    pairs_soa: Option<PairsSoA>,
    /// Lazy 缓存
    dailys_cache: Option<DataFrame>,
    pairs_cache: Option<DataFrame>,
    pub report: Option<Report>,
}

impl WeightBacktest {
    /// 创建持仓权重回测对象
    pub fn new(
        dfw: DataFrame,
        digits: i64,
        fee_rate: Option<f64>,
    ) -> Result<Self, WbtError> {
        // dt列格式转换
        let mut dfw = Self::convert_datetime(dfw).context("Failed to convert datetime")?;
        // weight列格式处理
        Self::round_weight(&mut dfw).context("Failed to round weight")?;

        let symbols = Self::unique_symbols(&dfw).context("Failed to unique_symbols")?;

        // O(N) Counting Sort 替代 Polars 通用排序
        let dfw = {
            let n_rows = dfw.height();
            let n_syms = symbols.len();

            let mut order_map: hashbrown::HashMap<&str, u32> =
                hashbrown::HashMap::with_capacity(n_syms);
            for (idx, sym) in symbols.iter().enumerate() {
                order_map.insert(sym.as_ref(), idx as u32);
            }
            let sym_ca = dfw.column("symbol")?.as_materialized_series().str()?;
            let sym_ids: Vec<u32> = sym_ca
                .into_iter()
                .map(|opt_s| opt_s.and_then(|s| order_map.get(s).copied()).unwrap_or(0))
                .collect();
            drop(order_map);

            let mut bucket_counts = vec![0u32; n_syms];
            for &sid in &sym_ids {
                bucket_counts[sid as usize] += 1;
            }
            let mut write_pos = vec![0u32; n_syms];
            let mut acc = 0u32;
            for i in 0..n_syms {
                write_pos[i] = acc;
                acc += bucket_counts[i];
            }

            let mut perm = vec![0u32; n_rows];
            for (i, &sid_val) in sym_ids.iter().enumerate().take(n_rows) {
                let sid = sid_val as usize;
                perm[write_pos[sid] as usize] = i as u32;
                write_pos[sid] += 1;
            }

            let perm_idx = IdxCa::new(PlSmallStr::from("idx"), &perm);
            let sym_id_vals: Vec<u32> = perm.iter().map(|&i| sym_ids[i as usize]).collect();

            DataFrame::new(vec![
                Column::new("sym_id".into(), sym_id_vals),
                dfw.column("dt")?.as_materialized_series().take(&perm_idx)?.into_column(),
                dfw.column("weight")?.as_materialized_series().take(&perm_idx)?.into_column(),
                dfw.column("price")?.as_materialized_series().take(&perm_idx)?.into_column(),
                dfw.column("symbol")?.as_materialized_series().take(&perm_idx)?.into_column(),
            ])?
        };

        let wb = Self {
            dfw,
            digits,
            symbols,
            fee_rate: fee_rate.unwrap_or(0.0002),
            dailys_soa: None,
            pairs_soa: None,
            dailys_cache: None,
            pairs_cache: None,
            report: None,
        };
        Ok(wb)
    }

    /// 执行回测并计算性能指标
    pub fn backtest(
        &mut self,
        n_jobs: Option<usize>,
        weight_type: WeightType,
        yearly_days: usize,
    ) -> Result<(), WbtError> {
        let n_jobs = n_jobs.unwrap_or(4);

        let pool = rayon::ThreadPoolBuilder::new()
            .stack_size(64 * 1024 * 1024)
            .num_threads(n_jobs)
            .build()
            .context("Failed to create thread pool")?;

        pool.install(|| self.do_backtest(weight_type, yearly_days))
    }

    /// 按需构建 dailys DataFrame（延迟物化，结果缓存）
    pub fn dailys_df(&mut self) -> Result<&mut DataFrame, WbtError> {
        if self.dailys_cache.is_none() {
            let df = self
                .dailys_soa
                .as_ref()
                .ok_or_else(|| {
                    WbtError::NoneValue("dailys_soa not computed yet".into())
                })?
                .to_dataframe()?;
            self.dailys_cache = Some(df);
        }
        Ok(self.dailys_cache.as_mut().unwrap())
    }

    /// 按需构建 pairs DataFrame（延迟物化，结果缓存）
    pub fn pairs_df(&mut self) -> Result<Option<&mut DataFrame>, WbtError> {
        if self.pairs_soa.is_none() {
            return Ok(None);
        }
        if self.pairs_cache.is_none() {
            let df = self.pairs_soa.as_ref().unwrap().to_dataframe()?;
            self.pairs_cache = Some(df);
        }
        Ok(self.pairs_cache.as_mut())
    }

    /// 按需构建 alpha DataFrame（从 DailyTotals 直接计算）
    pub fn alpha_df(&self) -> Result<DataFrame, WbtError> {
        let report = self
            .report
            .as_ref()
            .ok_or_else(|| WbtError::NoneValue("report not computed yet".into()))?;
        let dt = &report.daily_totals;
        let n = dt.strategy_means.len();

        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let dr_dates: Vec<i32> = dt
            .date_keys
            .iter()
            .map(|dk| {
                let nd = utils::date_key_to_naive_date(*dk);
                (nd - epoch).num_days() as i32
            })
            .collect();

        let excess: Vec<f64> = (0..n)
            .map(|i| dt.strategy_means[i] - dt.benchmark_means[i])
            .collect();

        DataFrame::new(vec![
            Series::new("date".into(), dr_dates)
                .cast(&DataType::Date)
                .map_err(WbtError::Polars)?
                .into_column(),
            Series::new("超额".into(), excess).into_column(),
            Series::new("策略".into(), &dt.strategy_means).into_column(),
            Series::new("基准".into(), &dt.benchmark_means).into_column(),
        ])
        .map_err(WbtError::Polars)
    }
}

// --- Utility methods (from utils.rs source) ---
impl WeightBacktest {
    /// 从 DataFrame 中的 `symbol` 列获取唯一品种集合
    pub(crate) fn unique_symbols(df: &DataFrame) -> Result<Vec<Arc<str>>, WbtError> {
        let symbols_series = df.column("symbol")?.as_materialized_series().str()?;
        let mut unique_symbols_set = hashbrown::HashSet::new();
        for symbol in symbols_series.into_iter().flatten() {
            unique_symbols_set.insert(symbol);
        }
        let mut unique_symbols: Vec<Arc<str>> =
            unique_symbols_set.into_iter().map(Arc::from).collect();
        unique_symbols.sort_unstable();
        Ok(unique_symbols)
    }

    fn sort_by_dt(df: DataFrame) -> Result<DataFrame, WbtError> {
        df.lazy()
            .sort(
                ["dt"],
                SortMultipleOptions::default().with_order_descending(false),
            )
            .collect()
            .map_err(|e| anyhow::anyhow!("Failed to sort by dt: {e}").into())
    }

    /// 将 DataFrame 中的 `dt` 列转换为 datetime 格式
    pub(crate) fn convert_datetime(mut df: DataFrame) -> Result<DataFrame, WbtError> {
        let dt_col = df.column("dt")?.as_materialized_series().clone();
        let dt_type = dt_col.dtype().clone();

        match &dt_type {
            DataType::Datetime(TimeUnit::Nanoseconds, _) => {
                Ok(Self::sort_by_dt(df)?)
            }
            DataType::Datetime(TimeUnit::Milliseconds, _) => {
                let dt_cast = dt_col
                    .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;
                let _ = df.replace("dt", dt_cast)?;
                Ok(Self::sort_by_dt(df)?)
            }
            DataType::Int64 => {
                let parsed_col = dt_col
                    .i64()?
                    .into_iter()
                    .map(|opt_ts| opt_ts.map(|ts| ts * 1000));
                let dt_s = Series::from_iter(parsed_col)
                    .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;
                let _ = df.replace("dt", dt_s)?;
                Ok(Self::sort_by_dt(df)?)
            }
            DataType::String => {
                let df = df
                    .lazy()
                    .with_column(col("dt").str().to_datetime(
                        Some(TimeUnit::Milliseconds),
                        None,
                        StrptimeOptions {
                            format: Some("%Y-%m-%d %H:%M:%S".into()),
                            strict: true,
                            exact: false,
                            cache: true,
                        },
                        lit("raise"),
                    ))
                    .sort(
                        ["dt"],
                        SortMultipleOptions::default().with_order_descending(false),
                    )
                    .collect()
                    .context("Failed to convert datetime")?;

                Ok(df)
            }
            _ => {
                return Err(anyhow::anyhow!("Unsupported datetime type: {:?}", dt_type).into());
            }
        }
    }

    /// 四舍五入 DataFrame 中的 `weight` 列，保留 4 位小数
    pub(crate) fn round_weight(df: &mut DataFrame) -> Result<(), WbtError> {
        let weight_s = df.column("weight")?.as_materialized_series().clone();
        let rounded = weight_s
            .f64()
            .unwrap()
            .into_iter()
            .map(|opt| opt.map(|val| (val * 10000.0).round() / 10000.0))
            .collect::<Float64Chunked>();
        let _ = df.replace("weight", rounded.into_series())?;
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_example_data() -> DataFrame {
        df! {
            "dt" => &[
                "2019-01-02 09:01:00",
                "2019-01-03 09:02:00",
                "2019-01-04 09:03:00",
                "2019-01-05 09:04:00",
                "2019-01-06 09:05:00"
            ],
            "symbol" => &["DLi9001"; 5],
            "weight" => &[
                0.511,
                0.000,
                -0.250,
                0.000,
                0.000
            ],
            "price" => &[
                961.695,
                960.720,
                962.669,
                960.720,
                961.695
            ]
        }
        .unwrap()
    }

    #[test]
    fn test_round_weight() {
        let mut df = raw_example_data();
        WeightBacktest::round_weight(&mut df).unwrap();
        println!("{df:?}");
    }

    #[test]
    fn test_convert_datetime() {
        let df = raw_example_data();
        let df = WeightBacktest::convert_datetime(df).unwrap();
        println!("{df:?}");
    }

    #[test]
    fn test_unique_symbols() {
        let df = raw_example_data();
        let symbols = WeightBacktest::unique_symbols(&df).unwrap();
        assert_eq!(symbols, vec![Arc::from("DLi9001")]);
    }
}
