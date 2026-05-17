use crate::core::cal_yearly_days::cal_yearly_days;
use crate::core::daily_performance::{DailyPerformance, daily_performance};
use chrono::NaiveDate;
use polars::prelude::*;

/// 滚动日度绩效（与 czsc.utils.analysis.stats.rolling_daily_performance 行为一致）。
///
/// 业务规则（全部在 Rust 内）：
/// 1. dates / returns 长度需一致；
/// 2. 按 dates 升序排序（不要求调用方预排）；
/// 3. returns 中的 NaN 视为 0；
/// 4. yearly_days 未提供时，调用 cal_yearly_days 自动推断；
/// 5. 滚动窗口：跳过前 min_periods 个点，每个 edt 取 [edt-window, edt] 区间（两端闭）。
pub fn rolling_daily_performance(
    dates: Vec<NaiveDate>,
    returns: Vec<f64>,
    window: i64,
    min_periods: usize,
    yearly_days: Option<usize>,
) -> PolarsResult<DataFrame> {
    if dates.len() != returns.len() {
        return Err(PolarsError::ComputeError(
            "dates 与 returns 长度必须一致".into(),
        ));
    }

    let mut indexed: Vec<(NaiveDate, f64)> = dates
        .into_iter()
        .zip(returns)
        .map(|(d, r)| (d, if r.is_nan() { 0.0 } else { r }))
        .collect();
    indexed.sort_by_key(|(d, _)| *d);

    let dates: Vec<NaiveDate> = indexed.iter().map(|(d, _)| *d).collect();
    let returns: Vec<f64> = indexed.iter().map(|(_, r)| *r).collect();

    let yd = yearly_days.unwrap_or_else(|| cal_yearly_days(&dates) as usize);
    let n = dates.len();

    let mut sdt_vec: Vec<NaiveDate> = Vec::with_capacity(n);
    let mut edt_vec: Vec<NaiveDate> = Vec::with_capacity(n);
    let mut perfs: Vec<DailyPerformance> = Vec::with_capacity(n);
    if min_periods < n {
        for end_idx in min_periods..n {
            let edt = dates[end_idx];
            let sdt = edt - chrono::Duration::days(window);
            let start_idx = dates.partition_point(|d| *d < sdt);
            let slice = &returns[start_idx..=end_idx];
            let perf = daily_performance(slice, Some(yd))
                .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?;
            perfs.push(perf);
            sdt_vec.push(sdt);
            edt_vec.push(edt);
        }
    }

    build_dataframe(&sdt_vec, &edt_vec, &perfs)
}

fn build_dataframe(
    sdt: &[NaiveDate],
    edt: &[NaiveDate],
    perfs: &[DailyPerformance],
) -> PolarsResult<DataFrame> {
    let to_date_series = |name: &str, v: &[NaiveDate]| {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let days: Vec<i32> = v.iter().map(|d| (*d - epoch).num_days() as i32).collect();
        Series::new(name.into(), days)
            .cast(&DataType::Date)
            .map(|s| s.into_column())
    };

    let abs_ret: Vec<f64> = perfs.iter().map(|p| p.absolute_return).collect();
    let ann_ret: Vec<f64> = perfs.iter().map(|p| p.annual_returns).collect();
    let sharpe: Vec<f64> = perfs.iter().map(|p| p.sharpe_ratio).collect();
    let mdd: Vec<f64> = perfs.iter().map(|p| p.max_drawdown).collect();
    let calmar: Vec<f64> = perfs.iter().map(|p| p.calmar_ratio).collect();
    let win_rate: Vec<f64> = perfs.iter().map(|p| p.daily_win_rate).collect();
    let pl_ratio: Vec<f64> = perfs.iter().map(|p| p.daily_profit_loss_ratio).collect();
    let win_prob: Vec<f64> = perfs.iter().map(|p| p.daily_win_probability).collect();
    let ann_vol: Vec<f64> = perfs.iter().map(|p| p.annual_volatility).collect();
    let down_vol: Vec<f64> = perfs.iter().map(|p| p.downside_volatility).collect();
    let nz_cov: Vec<f64> = perfs.iter().map(|p| p.non_zero_coverage).collect();
    let bep: Vec<f64> = perfs.iter().map(|p| p.break_even_point).collect();
    let nh_int: Vec<f64> = perfs.iter().map(|p| p.new_high_interval).collect();
    let nh_ratio: Vec<f64> = perfs.iter().map(|p| p.new_high_ratio).collect();
    let dd_risk: Vec<f64> = perfs.iter().map(|p| p.drawdown_risk).collect();
    let ann_lr: Vec<Option<f64>> = perfs.iter().map(|p| p.annual_lin_reg_cumsum_return).collect();
    let la_mdd: Vec<f64> = perfs.iter().map(|p| p.length_adjusted_average_max_drawdown).collect();

    DataFrame::new_infer_height(vec![
        Series::new("绝对收益".into(), abs_ret).into_column(),
        Series::new("年化".into(), ann_ret).into_column(),
        Series::new("夏普".into(), sharpe).into_column(),
        Series::new("最大回撤".into(), mdd).into_column(),
        Series::new("卡玛".into(), calmar).into_column(),
        Series::new("日胜率".into(), win_rate).into_column(),
        Series::new("日盈亏比".into(), pl_ratio).into_column(),
        Series::new("日赢面".into(), win_prob).into_column(),
        Series::new("年化波动率".into(), ann_vol).into_column(),
        Series::new("下行波动率".into(), down_vol).into_column(),
        Series::new("非零覆盖".into(), nz_cov).into_column(),
        Series::new("盈亏平衡点".into(), bep).into_column(),
        Series::new("新高间隔".into(), nh_int).into_column(),
        Series::new("新高占比".into(), nh_ratio).into_column(),
        Series::new("回撤风险".into(), dd_risk).into_column(),
        Series::new("回归年度回报率".into(), ann_lr).into_column(),
        Series::new("长度调整平均最大回撤".into(), la_mdd).into_column(),
        to_date_series("sdt", sdt)?,
        to_date_series("edt", edt)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn mismatched_lengths_return_error() {
        let res = rolling_daily_performance(
            vec![d(2024, 1, 1)],
            vec![0.01, 0.02],
            252,
            1,
            Some(252),
        );
        assert!(matches!(res, Err(PolarsError::ComputeError(_))));
    }

    #[test]
    fn min_periods_skips_warmup() {
        let dates: Vec<NaiveDate> = (0..400).map(|i| d(2022, 1, 1) + chrono::Duration::days(i)).collect();
        let returns: Vec<f64> = (0..400).map(|i| (i as f64) * 0.0001).collect();
        let df = rolling_daily_performance(dates, returns, 252, 100, Some(252)).unwrap();
        assert_eq!(df.height(), 300); // 400 - 100
    }

    #[test]
    fn nan_returns_treated_as_zero() {
        let dates: Vec<NaiveDate> = (0..400)
            .map(|i| d(2022, 1, 1) + chrono::Duration::days(i))
            .collect();
        let mut returns: Vec<f64> = vec![0.001; 400];
        // Scatter NaN across the series so every rolling window contains at least one
        for i in (10..400).step_by(20) {
            returns[i] = f64::NAN;
        }
        let df = rolling_daily_performance(dates, returns, 252, 100, Some(252)).unwrap();
        assert_eq!(df.height(), 300);

        // 验证 NaN→0 后所有窗口的核心指标都是有限值（否则 NaN 会污染下游计算）
        for col in ["年化", "夏普", "最大回撤"] {
            let series = df.column(col).unwrap().as_materialized_series();
            let f = series.f64().unwrap();
            for opt in f.into_iter() {
                let v = opt.expect("metric should be non-null");
                assert!(v.is_finite(), "column {col} contains non-finite value");
            }
        }
    }

    #[test]
    fn yearly_days_auto_inferred_when_none() {
        let dates: Vec<NaiveDate> = (0..400).map(|i| d(2022, 1, 1) + chrono::Duration::days(i)).collect();
        let returns: Vec<f64> = vec![0.001; 400];
        let df = rolling_daily_performance(dates, returns, 252, 100, None).unwrap();
        assert!(df.height() > 0);
    }

    #[test]
    fn unsorted_input_is_sorted_internally() {
        let mut dates: Vec<NaiveDate> = (0..400).map(|i| d(2022, 1, 1) + chrono::Duration::days(i)).collect();
        dates.reverse();
        let returns: Vec<f64> = vec![0.001; 400];
        let df = rolling_daily_performance(dates, returns, 252, 100, Some(252)).unwrap();
        let edt_col = df.column("edt").unwrap();
        let edts: Vec<i32> = edt_col
            .as_materialized_series()
            .cast(&DataType::Int32)
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!(edts.windows(2).all(|w| w[0] <= w[1]));
    }
}
