//! is_good_strategy: 评价策略能不能搞
//!
//! 提供两种判定模式：
//! - `history`：每个完整自然年绝对收益>0 或 波动率归一多头超额>0；且全样本多头超额最大回撤<阈值
//! - `recent` ：过去一年绝对收益>0 或 波动率归一多头超额>0；且过去一年多头超额最大回撤<阈值 且 <错开 recent 窗口后的历史最大回撤
//!
//! 业务口径与 [`crate::core::backtest::long_alpha_stats`] 中的波动率归一化保持一致。

use crate::core::daily_performance::daily_performance;
use crate::core::errors::WbtError;
use crate::core::utils::{date_key_to_naive_date, std_inline};
use chrono::{Datelike, NaiveDate};
use serde_json::{Value, json};
use std::collections::HashMap;

/// `is_good_strategy` 的判定模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    History,
    Recent,
}

/// 单个完整自然年的指标聚合。`year_passed` 字段由 [`judge`] 主流程根据
/// 业务条件（绝对收益>0 或 多头超额>0）回填，本算子内默认填 `false`。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct YearMetric {
    pub year: i32,
    pub abs_return: f64,
    pub alpha_return: f64,
    pub days: usize,
    pub is_complete_year: bool,
    pub year_passed: bool,
}

/// 给定多头日收益与基准日收益，按目标年化波动率分别归一化后做差，得到波动率归一多头超额日序列。
///
/// 与 `long_alpha_stats` 中的口径一致：当任一序列的年化波动率 < 1e-12 时，退化为全 0 序列。
pub(crate) fn compute_vol_adjusted_alpha(
    long: &[f64],
    bench: &[f64],
    yearly_days: usize,
    target_vol: f64,
) -> Vec<f64> {
    let yd_sqrt = (yearly_days as f64).sqrt();
    let long_vol = std_inline(long) * yd_sqrt;
    let bench_vol = std_inline(bench) * yd_sqrt;
    if long_vol < 1e-12 || bench_vol < 1e-12 {
        return vec![0.0; long.len()];
    }
    let long_scale = target_vol / long_vol;
    let bench_scale = target_vol / bench_vol;
    long.iter()
        .zip(bench.iter())
        .map(|(&l, &b)| l * long_scale - b * bench_scale)
        .collect()
}

/// 按年聚合策略绝对收益与波动率归一多头超额，输出每年的复利收益、交易日数和"完整自然年"标记。
///
/// `date_keys` 为自 1970-01-01 起的天数（与 `DailyTotals::date_keys` 一致）；
/// `strategy_daily` / `alpha_daily` 与 `date_keys` 等长且一一对应。
/// 复利公式 `(1+r1)*(1+r2)*...*(1+rn) - 1` 与 [`crate::core::yearly_return`] 保持一致。
pub(crate) fn compute_yearly_metrics(
    date_keys: &[i32],
    strategy_daily: &[f64],
    alpha_daily: &[f64],
    min_year_days: usize,
) -> Vec<YearMetric> {
    use std::collections::BTreeMap;

    let mut buckets: BTreeMap<i32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (i, &dk) in date_keys.iter().enumerate() {
        let nd = date_key_to_naive_date(dk);
        let year = nd.year();
        let entry = buckets.entry(year).or_default();
        entry.0.push(strategy_daily[i]);
        entry.1.push(alpha_daily[i]);
    }

    buckets
        .into_iter()
        .map(|(year, (strat, alpha))| {
            let days = strat.len();
            let abs_return = strat.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
            let alpha_return = alpha.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
            YearMetric {
                year,
                abs_return,
                alpha_return,
                days,
                is_complete_year: days >= min_year_days,
                year_passed: false,
            }
        })
        .collect()
}

/// 最近窗口（取序列尾部 `min(len, recent_days)` 天）的指标聚合。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecentMetric {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub actual_days: usize,
    pub abs_return: f64,
    pub alpha_return: f64,
    pub alpha_max_drawdown: f64,
}

/// 取序列尾部 `recent_days` 天（不足则取全部）做最近窗口聚合。
///
/// `alpha_max_drawdown` 取 `daily_performance(&alpha_window).max_drawdown.abs()`。
pub(crate) fn compute_recent_window(
    date_keys: &[i32],
    strategy_daily: &[f64],
    alpha_daily: &[f64],
    recent_days: usize,
    yearly_days: usize,
) -> Result<RecentMetric, WbtError> {
    let n = date_keys.len();
    let actual = n.min(recent_days);
    let start_idx = n - actual;
    let strat_w = &strategy_daily[start_idx..];
    let alpha_w = &alpha_daily[start_idx..];

    let abs_return = strat_w.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
    let alpha_return = alpha_w.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
    let dp = daily_performance(alpha_w, Some(yearly_days))?;
    let alpha_max_drawdown = dp.max_drawdown.abs();

    let start_date = date_key_to_naive_date(date_keys[start_idx]);
    let end_date = date_key_to_naive_date(date_keys[n - 1]);

    Ok(RecentMetric {
        start_date,
        end_date,
        actual_days: actual,
        abs_return,
        alpha_return,
        alpha_max_drawdown,
    })
}

/// 全样本 alpha 序列的最大回撤（取绝对值）。用于 `history` 模式条件 B。
pub(crate) fn compute_history_max_dd_full(
    alpha_daily: &[f64],
    yearly_days: usize,
) -> Result<f64, WbtError> {
    let dp = daily_performance(alpha_daily, Some(yearly_days))?;
    Ok(dp.max_drawdown.abs())
}

/// 剔除尾部 `min(len, recent_days)` 天后计算历史 alpha 最大回撤（取绝对值）。
///
/// 用于 `recent` 模式条件 D，避免与 recent 窗口重叠。
/// 剩余长度 0 时返回 `None`，由调用方判定 `is_good=false`。
pub(crate) fn compute_history_max_dd_excl_recent(
    alpha_daily: &[f64],
    recent_days: usize,
    yearly_days: usize,
) -> Result<Option<f64>, WbtError> {
    let n = alpha_daily.len();
    let recent_actual = n.min(recent_days);
    if n <= recent_actual {
        return Ok(None);
    }
    let head = &alpha_daily[..n - recent_actual];
    let dp = daily_performance(head, Some(yearly_days))?;
    Ok(Some(dp.max_drawdown.abs()))
}

/// 顶层判定函数。组装两种模式所需的全部子指标，返回一个 `HashMap<String, Value>`，
/// 顶层 key 使用英文 snake_case（详见方案子文档的"返回值结构"）。
///
/// `long_daily` 应来自 `aggregate_long_short_returns(...).0`，`bench_daily` 来自
/// `report.daily_totals.benchmark_means`，`strategy_daily` 来自 `report.daily_totals.strategy_means`。
#[allow(clippy::too_many_arguments)]
pub fn judge(
    mode: Mode,
    date_keys: &[i32],
    strategy_daily: &[f64],
    bench_daily: &[f64],
    long_daily: &[f64],
    yearly_days: usize,
    target_vol: f64,
    max_dd_threshold: f64,
    min_year_days: usize,
    recent_days: usize,
) -> Result<HashMap<String, Value>, WbtError> {
    let alpha_daily = compute_vol_adjusted_alpha(long_daily, bench_daily, yearly_days, target_vol);
    let mut out: HashMap<String, Value> = HashMap::new();
    let mut reasons: Vec<String> = Vec::new();

    match mode {
        Mode::History => {
            out.insert("mode".into(), json!("history"));

            let mut yearly =
                compute_yearly_metrics(date_keys, strategy_daily, &alpha_daily, min_year_days);
            for m in yearly.iter_mut() {
                if m.is_complete_year {
                    m.year_passed = m.abs_return > 0.0 || m.alpha_return > 0.0;
                }
            }

            let complete: Vec<&YearMetric> = yearly.iter().filter(|m| m.is_complete_year).collect();
            let cond_yearly = if complete.is_empty() {
                reasons.push("no complete year".into());
                false
            } else {
                let mut ok = true;
                for m in &complete {
                    if !m.year_passed {
                        reasons.push(format!(
                            "year {} both metrics ≤ 0 (abs_return={:.6}, alpha_return={:.6})",
                            m.year, m.abs_return, m.alpha_return
                        ));
                        ok = false;
                    }
                }
                ok
            };

            let history_dd_full = compute_history_max_dd_full(&alpha_daily, yearly_days)?;
            let cond_dd = history_dd_full < max_dd_threshold;
            if !cond_dd {
                reasons.push(format!(
                    "history_alpha_max_drawdown {:.6} ≥ threshold {:.6}",
                    history_dd_full, max_dd_threshold
                ));
            }

            out.insert(
                "yearly_metrics".into(),
                Value::Array(yearly.iter().map(year_metric_to_value).collect()),
            );
            out.insert("complete_year_count".into(), json!(complete.len()));
            out.insert("history_alpha_max_drawdown".into(), json!(history_dd_full));
            out.insert("cond_yearly_passed".into(), json!(cond_yearly));
            out.insert("cond_history_dd_passed".into(), json!(cond_dd));
            out.insert("is_good".into(), json!(cond_yearly && cond_dd));
        }
        Mode::Recent => {
            out.insert("mode".into(), json!("recent"));

            let recent = compute_recent_window(
                date_keys,
                strategy_daily,
                &alpha_daily,
                recent_days,
                yearly_days,
            )?;
            let history_dd_excl =
                compute_history_max_dd_excl_recent(&alpha_daily, recent_days, yearly_days)?;

            let window_empty = history_dd_excl.is_none();
            let cond_return = recent.abs_return > 0.0 || recent.alpha_return > 0.0;
            if !cond_return {
                reasons.push(format!(
                    "recent abs_return {:.6} ≤ 0 and alpha_return {:.6} ≤ 0",
                    recent.abs_return, recent.alpha_return
                ));
            }

            let (cond_dd, history_dd_excl_value) = match history_dd_excl {
                Some(h) => {
                    let ok = recent.alpha_max_drawdown < max_dd_threshold
                        && recent.alpha_max_drawdown < h;
                    if !ok {
                        if recent.alpha_max_drawdown >= max_dd_threshold {
                            reasons.push(format!(
                                "recent_alpha_max_drawdown {:.6} ≥ threshold {:.6}",
                                recent.alpha_max_drawdown, max_dd_threshold
                            ));
                        }
                        if recent.alpha_max_drawdown >= h {
                            reasons.push(format!(
                                "recent_alpha_max_drawdown {:.6} ≥ history_excl {:.6}",
                                recent.alpha_max_drawdown, h
                            ));
                        }
                    }
                    (ok, json!(h))
                }
                None => {
                    reasons.push("history window empty (sample too short)".into());
                    (false, json!(0.0))
                }
            };

            out.insert(
                "recent_start_date".into(),
                json!(recent.start_date.format("%Y-%m-%d").to_string()),
            );
            out.insert(
                "recent_end_date".into(),
                json!(recent.end_date.format("%Y-%m-%d").to_string()),
            );
            out.insert("recent_actual_days".into(), json!(recent.actual_days));
            out.insert("recent_abs_return".into(), json!(recent.abs_return));
            out.insert("recent_alpha_return".into(), json!(recent.alpha_return));
            out.insert(
                "recent_alpha_max_drawdown".into(),
                json!(recent.alpha_max_drawdown),
            );
            out.insert(
                "history_alpha_max_drawdown_excl_recent".into(),
                history_dd_excl_value,
            );
            out.insert("history_window_empty".into(), json!(window_empty));
            out.insert("cond_recent_return_passed".into(), json!(cond_return));
            out.insert("cond_recent_dd_passed".into(), json!(cond_dd));
            out.insert("is_good".into(), json!(cond_return && cond_dd));
        }
    }

    out.insert("reason".into(), json!(reasons.join("; ")));
    Ok(out)
}

fn year_metric_to_value(m: &YearMetric) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("year".into(), json!(m.year));
    obj.insert("abs_return".into(), json!(m.abs_return));
    obj.insert("alpha_return".into(), json!(m.alpha_return));
    obj.insert("days".into(), json!(m.days));
    obj.insert("is_complete_year".into(), json!(m.is_complete_year));
    obj.insert("year_passed".into(), json!(m.year_passed));
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 当 long 与 bench 完全相等时，各自归一到同一 target_vol，差应为 0。
    #[test]
    fn vol_adjusted_alpha_equal_inputs_produce_zero_series() {
        // 构造非平凡波动的收益序列；只要有非零方差即可走归一化分支
        let series: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let alpha = compute_vol_adjusted_alpha(&series, &series, 252, 0.20);
        assert_eq!(alpha.len(), series.len());
        for (i, v) in alpha.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "alpha[{i}] expected ~0, got {v} (long == bench should net to 0)"
            );
        }
    }

    /// 当 long 方差为 0（全 0 序列）时，按 long_alpha_stats 的边界口径退化为全 0 序列。
    #[test]
    fn vol_adjusted_alpha_zero_vol_returns_zero_series() {
        let long = vec![0.0_f64; 50];
        let bench: Vec<f64> = (0..50)
            .map(|i| if i % 3 == 0 { 0.005 } else { -0.002 })
            .collect();
        let alpha = compute_vol_adjusted_alpha(&long, &bench, 252, 0.20);
        assert_eq!(alpha.len(), 50);
        for (i, v) in alpha.iter().enumerate() {
            assert_eq!(*v, 0.0, "alpha[{i}] expected 0.0, got {v}");
        }
    }

    /// 长度一致性：返回序列长度必须等于 long.len()（也即 bench.len()）。
    #[test]
    fn vol_adjusted_alpha_length_matches_input() {
        let long: Vec<f64> = (0..37).map(|i| (i as f64) * 0.001 - 0.005).collect();
        let bench: Vec<f64> = (0..37).map(|i| (i as f64) * 0.0005 - 0.002).collect();
        let alpha = compute_vol_adjusted_alpha(&long, &bench, 252, 0.20);
        assert_eq!(alpha.len(), 37);
    }

    // ---------- T2: compute_yearly_metrics ----------

    /// 构造一个 YYYYMMDD 整数形式的 date_key（与 `DailyTotals::date_keys` 一致）。
    fn date_key(y: i32, m: u32, d: u32) -> i32 {
        y * 10000 + (m as i32) * 100 + (d as i32)
    }

    fn date_key_from_nd(nd: chrono::NaiveDate) -> i32 {
        nd.year() * 10000 + nd.month() as i32 * 100 + nd.day() as i32
    }

    /// 2020 满 130 天 + 2021 仅 30 天 + min_year_days=120
    /// → 2020 `is_complete_year=true`，2021 `is_complete_year=false`，但两年都返回。
    #[test]
    fn yearly_metrics_marks_incomplete_year() {
        let mut keys: Vec<i32> = Vec::new();
        let mut strat: Vec<f64> = Vec::new();
        let mut alpha: Vec<f64> = Vec::new();

        // 2020 年 130 天（从 1/2 起每天 +1，避开元旦）
        for i in 0..130 {
            let nd = chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap()
                + chrono::Duration::days(i as i64);
            keys.push(date_key_from_nd(nd));
            strat.push(0.001);
            alpha.push(0.0005);
        }
        // 2021 年 30 天
        for i in 0..30 {
            let nd = chrono::NaiveDate::from_ymd_opt(2021, 1, 4).unwrap()
                + chrono::Duration::days(i as i64);
            keys.push(date_key_from_nd(nd));
            strat.push(-0.001);
            alpha.push(-0.0005);
        }

        let metrics = compute_yearly_metrics(&keys, &strat, &alpha, 120);
        assert_eq!(metrics.len(), 2, "must keep both years; expected 2 entries");

        let y2020 = metrics.iter().find(|m| m.year == 2020).expect("2020");
        assert!(
            y2020.is_complete_year,
            "2020 should be complete (130 ≥ 120)"
        );
        assert_eq!(y2020.days, 130);

        let y2021 = metrics.iter().find(|m| m.year == 2021).expect("2021");
        assert!(
            !y2021.is_complete_year,
            "2021 should NOT be complete (30 < 120)"
        );
        assert_eq!(y2021.days, 30);
    }

    /// 已知日收益 → abs_return 必须用复利公式 (1+r1)*(1+r2)*...*(1+rn) - 1。
    #[test]
    fn yearly_metrics_uses_compound_formula() {
        // 2020-03-02 .. 2020-03-06，连续 5 个交易日
        let keys: Vec<i32> = (0..5)
            .map(|i| {
                let nd = chrono::NaiveDate::from_ymd_opt(2020, 3, 2).unwrap()
                    + chrono::Duration::days(i as i64);
                date_key_from_nd(nd)
            })
            .collect();
        let strat = vec![0.01_f64, 0.02, -0.01, 0.005, -0.002];
        let alpha = vec![0.005_f64, -0.003, 0.001, 0.002, -0.001];

        let metrics = compute_yearly_metrics(&keys, &strat, &alpha, 1);
        assert_eq!(metrics.len(), 1);
        let m = &metrics[0];
        assert_eq!(m.year, 2020);
        assert_eq!(m.days, 5);
        assert!(m.is_complete_year, "min_year_days=1 ≤ 5 days");

        let expected_abs = strat.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
        assert!(
            (m.abs_return - expected_abs).abs() < 1e-12,
            "abs_return: expected {expected_abs}, got {}",
            m.abs_return
        );
        let expected_alpha = alpha.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
        assert!(
            (m.alpha_return - expected_alpha).abs() < 1e-12,
            "alpha_return: expected {expected_alpha}, got {}",
            m.alpha_return
        );
    }

    /// year_passed 由 judge 主流程填，本算子默认输出 false。
    #[test]
    fn yearly_metrics_year_passed_defaults_to_false() {
        let keys = vec![date_key(2022, 6, 1)];
        let metrics = compute_yearly_metrics(&keys, &[0.01], &[0.005], 1);
        assert_eq!(metrics.len(), 1);
        assert!(!metrics[0].year_passed);
    }

    // ---------- T3: recent window + history max_dd ----------

    /// 构造 N 个连续交易日（用日历日代替）的 date_keys。
    fn consecutive_keys(start: chrono::NaiveDate, n: usize) -> Vec<i32> {
        (0..n)
            .map(|i| date_key_from_nd(start + chrono::Duration::days(i as i64)))
            .collect()
    }

    /// 500 天 + recent_days=252 → 取尾部 252 天，起止 date_key 落在最后 252 天上。
    #[test]
    fn recent_window_takes_tail_when_long_enough() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let keys = consecutive_keys(start, 500);
        let strat = vec![0.001_f64; 500];
        let alpha = vec![0.0005_f64; 500];
        let r = compute_recent_window(&keys, &strat, &alpha, 252, 252).unwrap();
        assert_eq!(r.actual_days, 252);
        let expected_start = start + chrono::Duration::days((500 - 252) as i64);
        let expected_end = start + chrono::Duration::days(499);
        assert_eq!(r.start_date, expected_start);
        assert_eq!(r.end_date, expected_end);
    }

    /// 100 天 + recent_days=252 → 取全部 100 天。
    #[test]
    fn recent_window_uses_all_when_short() {
        let start = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let keys = consecutive_keys(start, 100);
        let strat = vec![0.001_f64; 100];
        let alpha = vec![0.0005_f64; 100];
        let r = compute_recent_window(&keys, &strat, &alpha, 252, 252).unwrap();
        assert_eq!(r.actual_days, 100);
        assert_eq!(r.start_date, start);
        assert_eq!(r.end_date, start + chrono::Duration::days(99));
    }

    /// recent 窗口 abs_return / alpha_return 复利数值精确。
    #[test]
    fn recent_window_compound_formula_exact() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let keys = consecutive_keys(start, 5);
        let strat = vec![0.01_f64, -0.02, 0.015, 0.005, -0.01];
        let alpha = vec![0.003_f64, -0.001, 0.002, 0.0, -0.0005];
        let r = compute_recent_window(&keys, &strat, &alpha, 252, 252).unwrap();
        assert_eq!(r.actual_days, 5);
        let expected_abs = strat.iter().fold(1.0_f64, |acc, x| acc * (1.0 + x)) - 1.0;
        let expected_alpha = alpha.iter().fold(1.0_f64, |acc, x| acc * (1.0 + x)) - 1.0;
        assert!((r.abs_return - expected_abs).abs() < 1e-12);
        assert!((r.alpha_return - expected_alpha).abs() < 1e-12);
    }

    /// 全样本 max_dd 与 daily_performance 计算一致（取绝对值）。
    #[test]
    fn history_max_dd_full_matches_daily_performance() {
        let alpha = vec![0.05_f64, -0.10, -0.05, 0.02, 0.03];
        let dd = compute_history_max_dd_full(&alpha, 252).unwrap();
        let dp = daily_performance(&alpha, Some(252)).unwrap();
        assert!((dd - dp.max_drawdown.abs()).abs() < 1e-12);
    }

    /// 错开 recent 窗口后，历史 max_dd 只看前段。
    ///
    /// 构造：前 100 天恒为 +0.001（无回撤），尾 252 天恒为 -0.005（持续回撤）。
    /// recent_days=252 → excl_recent 部分只剩前 100 天，max_dd ≈ 0；
    /// 而全样本 max_dd 主要落在尾段，远 > 0。两者必不相等。
    #[test]
    fn history_max_dd_excl_recent_disjoints_from_recent_window() {
        let mut alpha: Vec<f64> = Vec::with_capacity(352);
        alpha.extend(std::iter::repeat(0.001_f64).take(100));
        alpha.extend(std::iter::repeat(-0.005_f64).take(252));

        let excl = compute_history_max_dd_excl_recent(&alpha, 252, 252)
            .unwrap()
            .unwrap();
        let full = compute_history_max_dd_full(&alpha, 252).unwrap();

        assert!(
            excl < 1e-6,
            "history excl_recent dd should be ~0 (前段无回撤), got {excl}"
        );
        assert!(
            full > 0.5,
            "full sample dd should be > 50% (尾段 252 天 × -0.5%), got {full}"
        );
        assert!(
            (excl - full).abs() > 0.1,
            "excl_recent and full must differ markedly"
        );
    }

    /// 序列长度 ≤ recent_days → excl_recent 部分为空 → None。
    #[test]
    fn history_max_dd_excl_recent_returns_none_when_too_short() {
        let alpha = vec![-0.01_f64; 200];
        let r = compute_history_max_dd_excl_recent(&alpha, 252, 252).unwrap();
        assert_eq!(r, None, "len=200 ≤ recent_days=252 must yield None");
    }

    // ---------- T4: judge ----------

    /// 构造 2 个完整自然年（各 130 天）的样例数据；long==bench → alpha=0 全程。
    /// 用户指定每年 strat_per_day_a / strat_per_day_b 来控制条件 A。
    fn build_two_year_samples(
        strat_per_day_a: f64,
        strat_per_day_b: f64,
    ) -> (Vec<i32>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut keys = Vec::new();
        let mut strat = Vec::new();
        let mut bench = Vec::new();
        let mut long = Vec::new();
        for i in 0..130 {
            let nd = chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap()
                + chrono::Duration::days(i as i64);
            keys.push(date_key_from_nd(nd));
            strat.push(strat_per_day_a);
            // long == bench → vol_adjusted alpha 全 0，history_dd_full = 0
            let same = 0.001 + 0.0005 * ((i % 3) as f64);
            bench.push(same);
            long.push(same);
        }
        for i in 0..130 {
            let nd = chrono::NaiveDate::from_ymd_opt(2021, 1, 4).unwrap()
                + chrono::Duration::days(i as i64);
            keys.push(date_key_from_nd(nd));
            strat.push(strat_per_day_b);
            let same = 0.001 + 0.0005 * ((i % 3) as f64);
            bench.push(same);
            long.push(same);
        }
        (keys, strat, bench, long)
    }

    /// History 全通过：两年都正绝对收益 + history_dd_full ≈ 0 < 0.20。
    #[test]
    fn history_passes_when_all_conditions_met() {
        let (k, s, b, l) = build_two_year_samples(0.001, 0.001);
        let r = judge(Mode::History, &k, &s, &b, &l, 252, 0.20, 0.20, 120, 252).unwrap();
        assert_eq!(r.get("mode").and_then(|v| v.as_str()), Some("history"));
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(true));
    }

    /// History 失败：2021 完整年绝对收益 < 0 且 alpha=0 不 > 0 → year_passed=false → is_good=false。
    #[test]
    fn history_fails_when_any_complete_year_both_metrics_negative() {
        let (k, s, b, l) = build_two_year_samples(0.001, -0.001);
        let r = judge(Mode::History, &k, &s, &b, &l, 252, 0.20, 0.20, 120, 252).unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            reason.contains("2021"),
            "reason should mention year 2021, got: {reason}"
        );
    }

    /// History 失败：无完整年（min_year_days=200，每年只有 130 天）。
    #[test]
    fn history_fails_when_no_complete_year() {
        let (k, s, b, l) = build_two_year_samples(0.001, 0.001);
        let r = judge(Mode::History, &k, &s, &b, &l, 252, 0.20, 0.20, 200, 252).unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(reason.contains("no complete year") || reason.contains("complete year"));
    }

    /// History 失败：max_dd 超阈。构造 long/bench 相反方向使归一化 alpha 大幅回撤。
    #[test]
    fn history_fails_when_max_dd_exceeds_threshold() {
        // 50 天 long 全负、bench 全正，归一化后 alpha 单调下行 → max_dd 接近持续累计
        let mut k = Vec::new();
        let mut s = Vec::new();
        let mut b = Vec::new();
        let mut l = Vec::new();
        for i in 0..150 {
            let nd = chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap()
                + chrono::Duration::days(i as i64);
            k.push(date_key_from_nd(nd));
            s.push(0.002); // 绝对收益正
            b.push(0.001 + 0.0005 * ((i % 3) as f64)); // 有方差
            l.push(-(0.001 + 0.0005 * ((i % 3) as f64))); // 反向
        }
        let r = judge(Mode::History, &k, &s, &b, &l, 252, 0.20, 0.20, 120, 252).unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        let dd = r
            .get("history_alpha_max_drawdown")
            .and_then(|v| v.as_f64())
            .expect("history_alpha_max_drawdown must be set");
        assert!(dd > 0.20, "history dd should exceed threshold, got {dd}");
    }

    /// Recent 模式契约：充足样本时返回的 dict 必含所有约定字段，且 history_window_empty=false。
    ///
    /// 注：vol_adjusted_alpha 用**全样本 std** 归一化，难以用合成数据精确构造满足
    /// "recent_dd < history_excl_dd 且 < threshold" 的 is_good=true 路径；完整端到端
    /// 通过路径留给 T6 Python e2e 用真实 backtest 数据覆盖。这里只校验字段契约。
    #[test]
    fn recent_returns_full_field_contract() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let mut k = Vec::new();
        let mut s = Vec::new();
        let mut b = Vec::new();
        let mut l = Vec::new();
        for i in 0..500 {
            k.push(date_key_from_nd(start + chrono::Duration::days(i as i64)));
            s.push(0.001);
            let bv = 0.001 + 0.0005 * ((i % 3) as f64);
            b.push(bv);
            l.push(bv);
        }
        let r = judge(Mode::Recent, &k, &s, &b, &l, 252, 0.20, 0.20, 120, 252).unwrap();
        assert_eq!(r.get("mode").and_then(|v| v.as_str()), Some("recent"));
        assert_eq!(
            r.get("history_window_empty").and_then(|v| v.as_bool()),
            Some(false)
        );
        for key in [
            "recent_start_date",
            "recent_end_date",
            "recent_actual_days",
            "recent_abs_return",
            "recent_alpha_return",
            "recent_alpha_max_drawdown",
            "history_alpha_max_drawdown_excl_recent",
            "cond_recent_return_passed",
            "cond_recent_dd_passed",
            "is_good",
            "reason",
        ] {
            assert!(r.contains_key(key), "missing key: {key}");
        }
        assert_eq!(
            r.get("recent_actual_days").and_then(|v| v.as_u64()),
            Some(252)
        );
    }

    /// Recent 失败：recent_dd 等于 history_excl_dd（严格不等的边界反例）。
    /// 构造方式：long==bench 全程 → 两段 alpha 都=0 → recent_dd==history_excl_dd==0 → 条件 D 不成立。
    #[test]
    fn recent_fails_when_dd_equals_history_excl() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let mut k = Vec::new();
        let mut s = Vec::new();
        let mut b = Vec::new();
        let mut l = Vec::new();
        for i in 0..500 {
            k.push(date_key_from_nd(start + chrono::Duration::days(i as i64)));
            s.push(0.001);
            let same = 0.001 + 0.0005 * ((i % 3) as f64);
            b.push(same);
            l.push(same);
        }
        let r = judge(Mode::Recent, &k, &s, &b, &l, 252, 0.20, 0.20, 120, 252).unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        let recent_dd = r
            .get("recent_alpha_max_drawdown")
            .and_then(|v| v.as_f64())
            .unwrap();
        let history_excl = r
            .get("history_alpha_max_drawdown_excl_recent")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!((recent_dd - history_excl).abs() < 1e-12);
    }

    /// Recent 失败：样本长度 ≤ recent_days → history 窗口为空 → is_good=false。
    #[test]
    fn recent_fails_when_history_window_empty() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let n = 200;
        let mut k = Vec::new();
        let mut s = Vec::new();
        let mut b = Vec::new();
        let mut l = Vec::new();
        for i in 0..n {
            k.push(date_key_from_nd(start + chrono::Duration::days(i as i64)));
            s.push(0.001);
            let same = 0.001 + 0.0005 * ((i % 3) as f64);
            b.push(same);
            l.push(same);
        }
        let r = judge(Mode::Recent, &k, &s, &b, &l, 252, 0.20, 0.20, 120, 252).unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(
            r.get("history_window_empty").and_then(|v| v.as_bool()),
            Some(true)
        );
        let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(reason.contains("history window empty"), "reason: {reason}");
    }
}
