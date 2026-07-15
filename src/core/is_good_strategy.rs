//! is_good_strategy: 评价策略能不能搞
//!
//! 提供两种判定模式：
//! - `history`：每个完整自然年（≥ `min_year_days`）满足三者之一即合格——
//!   绝对收益>0 / 波动率归一多头超额>0 / **当年**多头超额最大回撤<阈值；所有完整年都合格才进入
//!   全样本两道硬门——超额回撤 ≤ `max_alpha_dd_threshold`、Sharpe > `min_full_sharpe`。
//! - `recent` ：尾部 `recent_days` 天满足三者之一即可——绝对收益>0 / 波动率归一多头超额>0 /
//!   近期多头超额最大回撤<阈值；**且**近期最大回撤严格小于错开 recent 窗口后的历史最大回撤（唯一硬门）。
//!
//! ## 波动率归一化口径（重要）
//!
//! [`compute_vol_adjusted_alpha`] 计算的归一化 scale 基于**全样本** long/bench 序列
//! 的年化标准差，整段共用一组 (long_scale, bench_scale)。这意味着：
//!
//! - `Mode::History` 的逐年 `alpha_max_drawdown` 在全样本归一化序列上按年切片计算 — 与口径一致。
//! - `Mode::Recent` 的 `recent_alpha_max_drawdown` 与
//!   `history_alpha_max_drawdown_excl_recent` 都是在**同一条**全样本归一化序列上分别取
//!   尾部 / 头部计算，**不会**对 recent 窗口重新归一化。如果用户期望"recent 窗口按
//!   自身 vol 归一化"，需要单独跑一个截短样本的回测对象再调用本函数。
//!
//! ## 退化与错误
//!
//! - 输入序列含 NaN/Inf，或 long/bench 的年化波动率 < 1e-12（极端单值序列），
//!   归一化无法定义 → 返回 dict 标记 `alpha_degenerate=true`，所有 alpha 派生字段为
//!   `null`，`is_good=false`。
//! - 输入日期 / 序列长度不匹配、空输入、`recent_days=0` 等"用户错误"通过
//!   [`WbtError::InvalidInput`] 显式抛出。
//! - 业务口径与 [`crate::core::backtest::long_alpha_stats`] 中的波动率归一化保持一致。

use crate::core::errors::WbtError;
use crate::core::utils::std_inline;
use chrono::{Datelike, NaiveDate};
use serde_json::{Value, json};
use std::collections::HashMap;

/// 年度收益接近 0 时的浮点容差。`year_passed` / `cond_recent_return_passed` 使用。
const RETURN_EPSILON: f64 = 1e-9;

/// 年化波动率退化阈值。与 [`crate::core::backtest::long_alpha_stats`] 保持一致。
const VOL_EPSILON: f64 = 1e-12;

/// `is_good_strategy` 的判定模式。模块内部使用，不作为 SemVer 公开承诺。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    History,
    Recent,
}

/// 单个完整自然年的指标聚合。`year_passed` 字段由 [`judge`] 主流程根据
/// 业务条件（绝对收益>0 或 多头超额>0 或 当年超额回撤<阈值）回填，本算子内默认填 `false`。
/// `alpha_max_drawdown` 是**当年**多头超额日序列的最大回撤绝对值（在全样本归一化序列上按年切片）。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct YearMetric {
    pub year: i32,
    pub abs_return: f64,
    pub alpha_return: f64,
    pub alpha_max_drawdown: f64,
    pub days: usize,
    pub is_complete_year: bool,
    pub year_passed: bool,
}

/// 严格解码 YYYYMMDD 形式的 `date_key` 为 `NaiveDate`。无效 key 返回 `WbtError::InvalidInput`，
/// 不再像 [`crate::core::utils::date_key_to_naive_date`] 那样静默回落到 1970-01-01（避免污染年度聚合）。
fn parse_date_key_strict(dk: i32) -> Result<NaiveDate, WbtError> {
    let y = dk / 10000;
    let m = (dk / 100) % 100;
    let d = dk % 100;
    NaiveDate::from_ymd_opt(y, m as u32, d as u32).ok_or_else(|| {
        WbtError::InvalidInput(format!("invalid date_key {dk}: not a valid YYYYMMDD"))
    })
}

/// 直接从日收益序列计算最大回撤的绝对值，**不**依赖
/// [`crate::core::daily_performance`] 的早返回与四舍五入。
///
/// 口径：**单利**，与 [`crate::core::daily_performance::calc_underwater`] 一致（SKZ-195
/// 统一为单利）。累积收益 `cum = Σr`；峰值跟踪 `peak = max(cum)`；每步水下
/// `underwater = cum - peak`，最大回撤 = `max(peak - cum)`（收益空间的绝对回撤，非比例）。
/// 首个 bar 的 `peak` 即取 `cum`，故不在第 0 天记回撤，与 `calc_underwater` 同口径。
/// （历史上曾用复利净值 `nav = ∏(1+r)`、`dd = (peak - nav) / peak`。）
///
/// - 空输入返回 0。任一 r 不是有限值 → 返回 NaN（由调用方决策是否当作退化）。
fn local_max_drawdown_abs(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut cum = 0.0_f64;
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &r in returns {
        if !r.is_finite() {
            return f64::NAN;
        }
        cum += r;
        if cum > peak {
            peak = cum;
        }
        let dd = peak - cum;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// 全样本 Sharpe：`mean / std * sqrt(yearly_days)`，与 [`crate::core::daily_performance`]
/// 口径一致（ddof=0、不做 4 位四舍五入、不裁剪到 [-5, 10]；这里是判定门，不进展示）。
///
/// - 空 / 全等序列 → `std == 0` → 返回 `0.0`（调用方按"未达 Sharpe 阈值"判定）。
/// - 输入含 NaN/Inf → 返回 NaN。
fn full_sample_sharpe(returns: &[f64], yearly_days: usize) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let std = std_inline(returns);
    if !std.is_finite() {
        return f64::NAN;
    }
    // 浮点累计误差会让「数学上常数」的序列算出极小 std；与现有 `VOL_EPSILON` 同口径：
    // std 太小时视作"无波动"，Sharpe 退化为 0（"未达 Sharpe 阈值"路径）。
    if std < VOL_EPSILON {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    if !mean.is_finite() {
        return f64::NAN;
    }
    mean / std * (yearly_days as f64).sqrt()
}

/// 给定多头日收益与基准日收益，按目标年化波动率分别归一化后做差，得到波动率归一多头超额日序列。
///
/// 返回 `None` 表示**无法归一化**（输入含 NaN/Inf、长度不一致、或任一序列的年化波动率
/// < [`VOL_EPSILON`]）；此时调用方应将判定降级为"alpha 退化"而非视为零回撤通过。
pub(crate) fn compute_vol_adjusted_alpha(
    long: &[f64],
    bench: &[f64],
    yearly_days: usize,
    target_vol: f64,
) -> Option<Vec<f64>> {
    if long.len() != bench.len() {
        return None;
    }
    if !target_vol.is_finite() || target_vol <= 0.0 {
        return None;
    }
    if long.iter().any(|v| !v.is_finite()) || bench.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let yd_sqrt = (yearly_days as f64).sqrt();
    let long_vol = std_inline(long) * yd_sqrt;
    let bench_vol = std_inline(bench) * yd_sqrt;
    if !long_vol.is_finite()
        || !bench_vol.is_finite()
        || long_vol < VOL_EPSILON
        || bench_vol < VOL_EPSILON
    {
        return None;
    }
    let long_scale = target_vol / long_vol;
    let bench_scale = target_vol / bench_vol;
    Some(
        long.iter()
            .zip(bench.iter())
            .map(|(&l, &b)| l * long_scale - b * bench_scale)
            .collect(),
    )
}

/// 按年聚合策略绝对收益与波动率归一多头超额，输出每年的单利收益（`Σr`）、交易日数和"完整自然年"标记。
/// 口径与 [`crate::core::daily_performance`] 的绝对收益一致（SKZ-195 统一为单利）。
///
/// 输入要求：`date_keys` / `strategy_daily` / `alpha_daily` 等长；`date_keys` 是
/// YYYYMMDD 整数（与 `DailyTotals::date_keys` 一致）。长度不一致或 date_key 无效
/// 时返回 `WbtError::InvalidInput`，不再静默 panic / 回落到 1970。
pub(crate) fn compute_yearly_metrics(
    date_keys: &[i32],
    strategy_daily: &[f64],
    alpha_daily: &[f64],
    min_year_days: usize,
) -> Result<Vec<YearMetric>, WbtError> {
    use std::collections::BTreeMap;

    if date_keys.len() != strategy_daily.len() || date_keys.len() != alpha_daily.len() {
        return Err(WbtError::InvalidInput(format!(
            "compute_yearly_metrics length mismatch: date_keys={}, strategy_daily={}, alpha_daily={}",
            date_keys.len(),
            strategy_daily.len(),
            alpha_daily.len()
        )));
    }

    let mut buckets: BTreeMap<i32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (i, &dk) in date_keys.iter().enumerate() {
        let nd = parse_date_key_strict(dk)?;
        let year = nd.year();
        let entry = buckets.entry(year).or_default();
        entry.0.push(strategy_daily[i]);
        entry.1.push(alpha_daily[i]);
    }

    Ok(buckets
        .into_iter()
        .map(|(year, (strat, alpha))| {
            let days = strat.len();
            let abs_return: f64 = strat.iter().sum();
            let alpha_return: f64 = alpha.iter().sum();
            let alpha_max_drawdown = local_max_drawdown_abs(&alpha);
            YearMetric {
                year,
                abs_return,
                alpha_return,
                alpha_max_drawdown,
                days,
                is_complete_year: days >= min_year_days,
                year_passed: false,
            }
        })
        .collect())
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
/// 错误条件：空 `date_keys`、`recent_days == 0`、序列长度不一致、`date_keys` 含无效 YYYYMMDD。
/// `alpha_max_drawdown` 由 [`local_max_drawdown_abs`] 直接计算，不走 `daily_performance` 的早返回。
pub(crate) fn compute_recent_window(
    date_keys: &[i32],
    strategy_daily: &[f64],
    alpha_daily: &[f64],
    recent_days: usize,
) -> Result<RecentMetric, WbtError> {
    if date_keys.is_empty() {
        return Err(WbtError::InvalidInput(
            "compute_recent_window: date_keys is empty".into(),
        ));
    }
    if recent_days == 0 {
        return Err(WbtError::InvalidInput(
            "compute_recent_window: recent_days must be > 0".into(),
        ));
    }
    if date_keys.len() != strategy_daily.len() || date_keys.len() != alpha_daily.len() {
        return Err(WbtError::InvalidInput(format!(
            "compute_recent_window length mismatch: date_keys={}, strategy_daily={}, alpha_daily={}",
            date_keys.len(),
            strategy_daily.len(),
            alpha_daily.len()
        )));
    }

    let n = date_keys.len();
    let actual = n.min(recent_days);
    let start_idx = n - actual;
    let strat_w = &strategy_daily[start_idx..];
    let alpha_w = &alpha_daily[start_idx..];

    let abs_return: f64 = strat_w.iter().sum();
    let alpha_return: f64 = alpha_w.iter().sum();
    let alpha_max_drawdown = local_max_drawdown_abs(alpha_w);

    let start_date = parse_date_key_strict(date_keys[start_idx])?;
    let end_date = parse_date_key_strict(date_keys[n - 1])?;

    Ok(RecentMetric {
        start_date,
        end_date,
        actual_days: actual,
        abs_return,
        alpha_return,
        alpha_max_drawdown,
    })
}

/// 剔除尾部 `min(len, recent_days)` 天后计算历史 alpha 最大回撤（取绝对值）。
///
/// 用于 `recent` 模式条件 D，避免与 recent 窗口重叠。
/// `min_history_days` 为历史窗口必须达到的最少长度；不足时返回 `None`，由调用方
/// 判定 `is_good=false`（避免 head 长度极小时 max_dd≈0 导致条件 D 恒为假）。
pub(crate) fn compute_history_max_dd_excl_recent(
    alpha_daily: &[f64],
    recent_days: usize,
    min_history_days: usize,
) -> Option<f64> {
    let n = alpha_daily.len();
    let recent_actual = n.min(recent_days);
    let head_len = n.saturating_sub(recent_actual);
    if head_len < min_history_days || head_len == 0 {
        return None;
    }
    Some(local_max_drawdown_abs(&alpha_daily[..head_len]))
}

/// 顶层判定函数。组装两种模式所需的全部子指标，返回一个 `HashMap<String, Value>`，
/// 顶层 key 使用英文 snake_case（详见方案子文档的"返回值结构"）。
///
/// `long_daily` 应来自同 crate 内的 `aggregate_long_short_returns(...).0`；该 helper 是
/// 私有的，因此 `judge` 也保持 `pub(crate)`，避免暴露一个外部无法满足契约的 API。
#[allow(clippy::too_many_arguments)]
pub(crate) fn judge(
    mode: Mode,
    date_keys: &[i32],
    strategy_daily: &[f64],
    bench_daily: &[f64],
    long_daily: &[f64],
    yearly_days: usize,
    target_vol: f64,
    max_dd_threshold: f64,
    max_alpha_dd_threshold: f64,
    min_full_sharpe: f64,
    min_year_days: usize,
    recent_days: usize,
    min_history_days: usize,
) -> Result<HashMap<String, Value>, WbtError> {
    // ---- 入口验证 ----
    let n = date_keys.len();
    if n == 0 {
        return Err(WbtError::InvalidInput("date_keys is empty".into()));
    }
    if strategy_daily.len() != n || bench_daily.len() != n || long_daily.len() != n {
        return Err(WbtError::InvalidInput(format!(
            "input length mismatch: date_keys={}, strategy={}, bench={}, long={}",
            n,
            strategy_daily.len(),
            bench_daily.len(),
            long_daily.len()
        )));
    }
    if matches!(mode, Mode::Recent) && recent_days == 0 {
        return Err(WbtError::InvalidInput(
            "recent_days must be > 0 for recent mode".into(),
        ));
    }
    if !target_vol.is_finite() || target_vol <= 0.0 {
        return Err(WbtError::InvalidInput(format!(
            "target_vol must be positive and finite, got {target_vol}"
        )));
    }
    if !max_dd_threshold.is_finite() || max_dd_threshold <= 0.0 {
        return Err(WbtError::InvalidInput(format!(
            "max_dd_threshold must be positive and finite, got {max_dd_threshold}"
        )));
    }
    if !max_alpha_dd_threshold.is_finite() || max_alpha_dd_threshold <= 0.0 {
        return Err(WbtError::InvalidInput(format!(
            "max_alpha_dd_threshold must be positive and finite, got {max_alpha_dd_threshold}"
        )));
    }
    if !min_full_sharpe.is_finite() {
        return Err(WbtError::InvalidInput(format!(
            "min_full_sharpe must be finite, got {min_full_sharpe}"
        )));
    }
    for &dk in date_keys {
        parse_date_key_strict(dk)?;
    }
    if strategy_daily.iter().any(|v| !v.is_finite()) {
        return Err(WbtError::InvalidInput(
            "strategy_daily contains NaN/Inf".into(),
        ));
    }

    // ---- alpha 序列（含退化判定）----
    let alpha_opt = compute_vol_adjusted_alpha(long_daily, bench_daily, yearly_days, target_vol);
    let alpha_degenerate = alpha_opt.is_none();
    let alpha_daily = alpha_opt.unwrap_or_else(|| vec![0.0; n]);

    let mut out: HashMap<String, Value> = HashMap::new();
    let mut reasons: Vec<String> = Vec::new();
    if alpha_degenerate {
        reasons
            .push("alpha series degenerate (NaN/Inf or vol of long/bench below threshold)".into());
    }

    match mode {
        Mode::History => {
            out.insert("mode".into(), json!("history"));

            let mut yearly =
                compute_yearly_metrics(date_keys, strategy_daily, &alpha_daily, min_year_days)?;
            // 逐年三路 OR：绝对收益>ε 或 多头超额>ε 或 当年超额回撤<阈值。
            // alpha 退化时 alpha 派生两路不参与（避免"全 0 alpha → 回撤 0"凭空通过）。
            for m in yearly.iter_mut() {
                if m.is_complete_year {
                    m.year_passed = m.abs_return > RETURN_EPSILON
                        || (!alpha_degenerate
                            && (m.alpha_return > RETURN_EPSILON
                                || m.alpha_max_drawdown < max_dd_threshold));
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
                            "year {} all three metrics fail (abs_return={:.6}, alpha_return={:.6}, alpha_max_drawdown={:.6} ≥ threshold {:.6})",
                            m.year, m.abs_return, m.alpha_return, m.alpha_max_drawdown, max_dd_threshold
                        ));
                        ok = false;
                    }
                }
                ok
            };

            // 全样本两道硬门：超额回撤 ≤ 阈值、Sharpe > 阈值。退化时全部 NaN/无意义，
            // 沿用"alpha 退化即 is_good=false"的既有契约。
            let history_alpha_max_drawdown = if alpha_degenerate {
                f64::NAN
            } else {
                local_max_drawdown_abs(&alpha_daily)
            };
            let history_alpha_sharpe = if alpha_degenerate {
                f64::NAN
            } else {
                full_sample_sharpe(&alpha_daily, yearly_days)
            };

            let cond_history_dd =
                !alpha_degenerate && history_alpha_max_drawdown <= max_alpha_dd_threshold;
            let cond_history_sharpe = !alpha_degenerate && history_alpha_sharpe > min_full_sharpe;
            if !alpha_degenerate && !cond_history_dd {
                reasons.push(format!(
                    "history full-sample excess drawdown {:.6} > max_alpha_dd_threshold {:.6}",
                    history_alpha_max_drawdown, max_alpha_dd_threshold
                ));
            }
            if !alpha_degenerate && !cond_history_sharpe {
                reasons.push(format!(
                    "history full-sample Sharpe {:.6} ≤ min_full_sharpe {:.6}",
                    history_alpha_sharpe, min_full_sharpe
                ));
            }

            out.insert(
                "yearly_metrics".into(),
                Value::Array(yearly.iter().map(year_metric_to_value).collect()),
            );
            out.insert("complete_year_count".into(), json!(complete.len()));
            out.insert("alpha_degenerate".into(), json!(alpha_degenerate));
            out.insert("cond_yearly_passed".into(), json!(cond_yearly));
            out.insert(
                "history_alpha_max_drawdown".into(),
                if alpha_degenerate {
                    Value::Null
                } else {
                    json!(history_alpha_max_drawdown)
                },
            );
            out.insert(
                "history_alpha_sharpe".into(),
                if alpha_degenerate {
                    Value::Null
                } else {
                    json!(history_alpha_sharpe)
                },
            );
            out.insert("cond_history_dd_passed".into(), json!(cond_history_dd));
            out.insert(
                "cond_history_sharpe_passed".into(),
                json!(cond_history_sharpe),
            );
            out.insert(
                "is_good".into(),
                json!(!alpha_degenerate && cond_yearly && cond_history_dd && cond_history_sharpe),
            );
        }
        Mode::Recent => {
            out.insert("mode".into(), json!("recent"));

            let recent =
                compute_recent_window(date_keys, strategy_daily, &alpha_daily, recent_days)?;
            let history_dd_excl = if alpha_degenerate {
                None
            } else {
                compute_history_max_dd_excl_recent(&alpha_daily, recent_days, min_history_days)
            };

            // 收益侧三路 OR：绝对收益>ε 或 多头超额>ε 或 近期超额回撤<阈值。
            // 退化时 alpha 派生两路不参与，且整体 is_good 强制 false（见文档契约）。
            let cond_return_value = recent.abs_return > RETURN_EPSILON
                || recent.alpha_return > RETURN_EPSILON
                || recent.alpha_max_drawdown < max_dd_threshold;
            let cond_return = !alpha_degenerate && cond_return_value;
            if !alpha_degenerate && !cond_return {
                reasons.push(format!(
                    "recent all three metrics fail (abs_return={:.6}, alpha_return={:.6}, alpha_max_drawdown={:.6} ≥ threshold {:.6})",
                    recent.abs_return, recent.alpha_return, recent.alpha_max_drawdown, max_dd_threshold
                ));
            }

            // 唯一保留的硬门：近期最大回撤严格小于「剔除 recent 窗口后」的历史最大回撤。
            let history_window_short = history_dd_excl.is_none() && !alpha_degenerate;
            let (cond_dd, history_dd_excl_value) = if alpha_degenerate {
                (false, Value::Null)
            } else {
                match history_dd_excl {
                    Some(h) => {
                        let ok = recent.alpha_max_drawdown < h;
                        if !ok {
                            reasons.push(format!(
                                "recent_alpha_max_drawdown {:.6} ≥ history_excl {:.6}",
                                recent.alpha_max_drawdown, h
                            ));
                        }
                        (ok, json!(h))
                    }
                    None => {
                        reasons.push(format!(
                            "history window too short (head < min_history_days={min_history_days})"
                        ));
                        (false, Value::Null)
                    }
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
            out.insert(
                "recent_alpha_return".into(),
                if alpha_degenerate {
                    Value::Null
                } else {
                    json!(recent.alpha_return)
                },
            );
            out.insert(
                "recent_alpha_max_drawdown".into(),
                if alpha_degenerate {
                    Value::Null
                } else {
                    json!(recent.alpha_max_drawdown)
                },
            );
            out.insert(
                "history_alpha_max_drawdown_excl_recent".into(),
                history_dd_excl_value,
            );
            out.insert(
                "history_window_empty".into(),
                json!(alpha_degenerate || history_window_short),
            );
            out.insert("alpha_degenerate".into(), json!(alpha_degenerate));
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
    obj.insert("alpha_max_drawdown".into(), json!(m.alpha_max_drawdown));
    obj.insert("days".into(), json!(m.days));
    obj.insert("is_complete_year".into(), json!(m.is_complete_year));
    obj.insert("year_passed".into(), json!(m.year_passed));
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===========================================================================
    // Helpers
    // ===========================================================================

    fn date_key(y: i32, m: u32, d: u32) -> i32 {
        y * 10000 + (m as i32) * 100 + (d as i32)
    }

    fn date_key_from_nd(nd: chrono::NaiveDate) -> i32 {
        nd.year() * 10000 + nd.month() as i32 * 100 + nd.day() as i32
    }

    fn consecutive_keys(start: chrono::NaiveDate, n: usize) -> Vec<i32> {
        (0..n)
            .map(|i| date_key_from_nd(start + chrono::Duration::days(i as i64)))
            .collect()
    }

    /// 两年样本：long 与 bench 不同周期的小振荡，alpha 非退化。
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
            bench.push(0.0003 * ((i % 3) as f64 - 1.0));
            long.push(0.0003 * ((i % 5) as f64 - 2.0));
        }
        for i in 0..130 {
            let nd = chrono::NaiveDate::from_ymd_opt(2021, 1, 4).unwrap()
                + chrono::Duration::days(i as i64);
            keys.push(date_key_from_nd(nd));
            strat.push(strat_per_day_b);
            bench.push(0.0003 * ((i % 3) as f64 - 1.0));
            long.push(0.0003 * ((i % 5) as f64 - 2.0));
        }
        (keys, strat, bench, long)
    }

    // ===========================================================================
    // local_max_drawdown_abs
    // ===========================================================================

    #[test]
    fn local_max_drawdown_known_curve() {
        // 单利：[+0.05, -0.10, -0.05, 0.02, 0.03]
        // cum:  0.05, -0.05, -0.10, -0.08, -0.05
        // peak: 0.05（首个 bar），谷底 cum = -0.10 → 最大回撤 = 0.05 - (-0.10) = 0.15
        let alpha = vec![0.05_f64, -0.10, -0.05, 0.02, 0.03];
        let dd = local_max_drawdown_abs(&alpha);
        let expected = 0.05_f64 - (-0.10);
        assert!(
            (dd - expected).abs() < 1e-10,
            "expected {expected}, got {dd}"
        );
    }

    #[test]
    fn local_max_drawdown_empty_is_zero() {
        assert_eq!(local_max_drawdown_abs(&[]), 0.0);
    }

    #[test]
    fn local_max_drawdown_nan_input_returns_nan() {
        let alpha = vec![0.01, f64::NAN, 0.02];
        assert!(local_max_drawdown_abs(&alpha).is_nan());
    }

    /// F2 regression: 末端累计收益 ~= 0，但中途真实回撤必须照报。
    #[test]
    fn local_max_drawdown_survives_cumzero_with_real_dd() {
        // 单利 cum: 0.2, -0.3, 0.3667, 0.3667；末端 ~= 0.3667 但中途从 peak=0.2
        // 跌到 -0.3 → 回撤 = 0.2 - (-0.3) = 0.5。
        let alpha = vec![0.2_f64, -0.5, 2.0 / 3.0, 0.0];
        let dd = local_max_drawdown_abs(&alpha);
        assert!((dd - 0.5).abs() < 1e-10, "expected 0.5, got {dd}");
    }

    // ===========================================================================
    // full_sample_sharpe
    // ===========================================================================

    #[test]
    fn full_sample_sharpe_known_values() {
        // 均值 0.01、ddof=0 std ≈ 0.008165、yearly_days=252：
        // sharpe = 0.01 / 0.008165 * sqrt(252) ≈ 19.27
        let returns = [0.0_f64, 0.02, 0.01];
        let s = full_sample_sharpe(&returns, 252);
        let expected = (0.01_f64 / (2.0_f64 / 3.0 * 0.0001).sqrt()) * (252.0_f64).sqrt();
        assert!((s - expected).abs() < 1e-9, "expected {expected}, got {s}");
    }

    #[test]
    fn full_sample_sharpe_empty_returns_zero() {
        assert_eq!(full_sample_sharpe(&[], 252), 0.0);
    }

    #[test]
    fn full_sample_sharpe_constant_returns_zero() {
        assert_eq!(full_sample_sharpe(&[0.01; 50], 252), 0.0);
    }

    #[test]
    fn full_sample_sharpe_nan_input_returns_nan() {
        assert!(full_sample_sharpe(&[0.01, f64::NAN, 0.02], 252).is_nan());
    }

    // ===========================================================================
    // parse_date_key_strict
    // ===========================================================================

    #[test]
    fn parse_date_key_strict_valid() {
        let nd = parse_date_key_strict(20200601).unwrap();
        assert_eq!(nd, chrono::NaiveDate::from_ymd_opt(2020, 6, 1).unwrap());
    }

    #[test]
    fn parse_date_key_strict_invalid_errors() {
        assert!(matches!(
            parse_date_key_strict(0),
            Err(WbtError::InvalidInput(_))
        ));
        assert!(matches!(
            parse_date_key_strict(20200230),
            Err(WbtError::InvalidInput(_))
        ));
    }

    // ===========================================================================
    // compute_vol_adjusted_alpha
    // ===========================================================================

    #[test]
    fn vol_adjusted_alpha_equal_inputs_produce_zero_series() {
        let series: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let alpha = compute_vol_adjusted_alpha(&series, &series, 252, 0.20).unwrap();
        assert_eq!(alpha.len(), series.len());
        for (i, v) in alpha.iter().enumerate() {
            assert!(v.abs() < 1e-12, "alpha[{i}] expected ~0, got {v}");
        }
    }

    /// F3 fix: long zero-vol returns None.
    #[test]
    fn vol_adjusted_alpha_zero_long_vol_returns_none() {
        let long = vec![0.0_f64; 50];
        let bench: Vec<f64> = (0..50)
            .map(|i| if i % 3 == 0 { 0.005 } else { -0.002 })
            .collect();
        assert!(compute_vol_adjusted_alpha(&long, &bench, 252, 0.20).is_none());
    }

    /// F3 fix: bench zero-vol returns None.
    #[test]
    fn vol_adjusted_alpha_zero_bench_vol_returns_none() {
        let bench = vec![0.0_f64; 50];
        let long: Vec<f64> = (0..50)
            .map(|i| if i % 3 == 0 { 0.005 } else { -0.002 })
            .collect();
        assert!(compute_vol_adjusted_alpha(&long, &bench, 252, 0.20).is_none());
    }

    /// F4 fix: NaN inputs return None.
    #[test]
    fn vol_adjusted_alpha_nan_input_returns_none() {
        let mut long: Vec<f64> = (0..50).map(|i| 0.001 * (i as f64)).collect();
        long[10] = f64::NAN;
        let bench: Vec<f64> = (0..50).map(|i| 0.001 * (i as f64)).collect();
        assert!(compute_vol_adjusted_alpha(&long, &bench, 252, 0.20).is_none());
    }

    #[test]
    fn vol_adjusted_alpha_length_mismatch_returns_none() {
        let long = vec![0.01_f64; 10];
        let bench = vec![0.01_f64; 9];
        assert!(compute_vol_adjusted_alpha(&long, &bench, 252, 0.20).is_none());
    }

    #[test]
    fn vol_adjusted_alpha_length_matches_input() {
        let long: Vec<f64> = (0..37).map(|i| (i as f64) * 0.001 - 0.005).collect();
        let bench: Vec<f64> = (0..37).map(|i| (i as f64) * 0.0005 - 0.002).collect();
        let alpha = compute_vol_adjusted_alpha(&long, &bench, 252, 0.20).unwrap();
        assert_eq!(alpha.len(), 37);
    }

    // ===========================================================================
    // compute_yearly_metrics
    // ===========================================================================

    #[test]
    fn yearly_metrics_marks_incomplete_year() {
        let mut keys: Vec<i32> = Vec::new();
        let mut strat: Vec<f64> = Vec::new();
        let mut alpha: Vec<f64> = Vec::new();
        for i in 0..130 {
            let nd = chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap()
                + chrono::Duration::days(i as i64);
            keys.push(date_key_from_nd(nd));
            strat.push(0.001);
            alpha.push(0.0005);
        }
        for i in 0..30 {
            let nd = chrono::NaiveDate::from_ymd_opt(2021, 1, 4).unwrap()
                + chrono::Duration::days(i as i64);
            keys.push(date_key_from_nd(nd));
            strat.push(-0.001);
            alpha.push(-0.0005);
        }
        let metrics = compute_yearly_metrics(&keys, &strat, &alpha, 120).unwrap();
        assert_eq!(metrics.len(), 2);
        assert!(
            metrics
                .iter()
                .find(|m| m.year == 2020)
                .unwrap()
                .is_complete_year
        );
        assert!(
            !metrics
                .iter()
                .find(|m| m.year == 2021)
                .unwrap()
                .is_complete_year
        );
    }

    #[test]
    fn yearly_metrics_uses_simple_formula() {
        let keys: Vec<i32> = (0..5)
            .map(|i| {
                let nd = chrono::NaiveDate::from_ymd_opt(2020, 3, 2).unwrap()
                    + chrono::Duration::days(i as i64);
                date_key_from_nd(nd)
            })
            .collect();
        let strat = vec![0.01_f64, 0.02, -0.01, 0.005, -0.002];
        let alpha = vec![0.005_f64, -0.003, 0.001, 0.002, -0.001];
        let metrics = compute_yearly_metrics(&keys, &strat, &alpha, 1).unwrap();
        assert_eq!(metrics.len(), 1);
        let m = &metrics[0];
        assert_eq!(m.year, 2020);
        assert_eq!(m.days, 5);
        // 单利：逐日收益直接求和，与 daily_performance 绝对收益口径一致。
        let expected_abs: f64 = strat.iter().sum();
        assert!((m.abs_return - expected_abs).abs() < 1e-12);
        let expected_alpha: f64 = alpha.iter().sum();
        assert!((m.alpha_return - expected_alpha).abs() < 1e-12);
    }

    #[test]
    fn yearly_metrics_year_passed_defaults_to_false() {
        let keys = vec![date_key(2022, 6, 1)];
        let metrics = compute_yearly_metrics(&keys, &[0.01], &[0.005], 1).unwrap();
        assert_eq!(metrics.len(), 1);
        assert!(!metrics[0].year_passed);
    }

    /// F6 fix: length mismatch returns InvalidInput.
    #[test]
    fn yearly_metrics_length_mismatch_errors() {
        let keys = vec![date_key(2020, 1, 1), date_key(2020, 1, 2)];
        let r = compute_yearly_metrics(&keys, &[0.01], &[0.005, 0.001], 1);
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    /// F5 fix: invalid date_key returns InvalidInput.
    #[test]
    fn yearly_metrics_invalid_date_key_errors() {
        let keys = vec![20200230_i32];
        let r = compute_yearly_metrics(&keys, &[0.01], &[0.005], 1);
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    // ===========================================================================
    // compute_recent_window
    // ===========================================================================

    #[test]
    fn recent_window_takes_tail_when_long_enough() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let keys = consecutive_keys(start, 500);
        let strat = vec![0.001_f64; 500];
        let alpha = vec![0.0005_f64; 500];
        let r = compute_recent_window(&keys, &strat, &alpha, 252).unwrap();
        assert_eq!(r.actual_days, 252);
        assert_eq!(r.start_date, start + chrono::Duration::days(248));
        assert_eq!(r.end_date, start + chrono::Duration::days(499));
    }

    #[test]
    fn recent_window_uses_all_when_short() {
        let start = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let keys = consecutive_keys(start, 100);
        let strat = vec![0.001_f64; 100];
        let alpha = vec![0.0005_f64; 100];
        let r = compute_recent_window(&keys, &strat, &alpha, 252).unwrap();
        assert_eq!(r.actual_days, 100);
        assert_eq!(r.start_date, start);
        assert_eq!(r.end_date, start + chrono::Duration::days(99));
    }

    #[test]
    fn recent_window_simple_formula_exact() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let keys = consecutive_keys(start, 5);
        let strat = vec![0.01_f64, -0.02, 0.015, 0.005, -0.01];
        let alpha = vec![0.003_f64, -0.001, 0.002, 0.0, -0.0005];
        let r = compute_recent_window(&keys, &strat, &alpha, 252).unwrap();
        // 单利：窗口内逐日收益直接求和。
        let expected_abs: f64 = strat.iter().sum();
        let expected_alpha: f64 = alpha.iter().sum();
        assert!((r.abs_return - expected_abs).abs() < 1e-12);
        assert!((r.alpha_return - expected_alpha).abs() < 1e-12);
    }

    /// F1 fix: empty date_keys returns InvalidInput.
    #[test]
    fn recent_window_empty_date_keys_errors() {
        let r = compute_recent_window(&[], &[], &[], 252);
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    /// F1 fix: recent_days=0 returns InvalidInput.
    #[test]
    fn recent_window_zero_recent_days_errors() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let keys = consecutive_keys(start, 10);
        let r = compute_recent_window(&keys, &[0.001_f64; 10], &[0.001_f64; 10], 0);
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn recent_window_length_mismatch_errors() {
        let keys = vec![20200101_i32, 20200102];
        let r = compute_recent_window(&keys, &[0.001], &[0.001, 0.002], 5);
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    // ===========================================================================
    // compute_history_max_dd_excl_recent
    // ===========================================================================

    /// Disjoint check, min_history_days=0 disables floor.
    #[test]
    fn history_max_dd_excl_recent_disjoints_from_recent_window() {
        let mut alpha: Vec<f64> = Vec::with_capacity(352);
        alpha.extend(std::iter::repeat_n(0.001_f64, 100));
        alpha.extend(std::iter::repeat_n(-0.005_f64, 252));
        let excl = compute_history_max_dd_excl_recent(&alpha, 252, 0).unwrap();
        let full = local_max_drawdown_abs(&alpha);
        assert!(excl < 1e-6);
        assert!(full > 0.5);
        assert!((excl - full).abs() > 0.1);
    }

    #[test]
    fn history_max_dd_excl_recent_returns_none_when_no_head() {
        let alpha = vec![-0.01_f64; 200];
        assert_eq!(compute_history_max_dd_excl_recent(&alpha, 252, 0), None);
    }

    /// F7 fix: head shorter than min_history_days returns None.
    #[test]
    fn history_max_dd_excl_recent_returns_none_when_head_below_floor() {
        let alpha = vec![0.001_f64; 255];
        assert_eq!(compute_history_max_dd_excl_recent(&alpha, 252, 60), None);
    }

    #[test]
    fn history_max_dd_excl_recent_zero_floor_accepts_any_head() {
        let alpha = vec![0.001_f64; 255];
        assert!(compute_history_max_dd_excl_recent(&alpha, 252, 0).is_some());
    }

    // ===========================================================================
    // judge: input validation
    // ===========================================================================

    #[test]
    fn judge_empty_input_errors() {
        let r = judge(
            Mode::History,
            &[],
            &[],
            &[],
            &[],
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_length_mismatch_errors() {
        let r = judge(
            Mode::History,
            &[20200101_i32],
            &[0.01],
            &[0.01, 0.01],
            &[0.01],
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_invalid_target_vol_errors() {
        let keys = vec![20200101_i32];
        let r = judge(
            Mode::History,
            &keys,
            &[0.01],
            &[0.01],
            &[0.01],
            252,
            0.0,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_invalid_max_dd_threshold_errors() {
        let keys = vec![20200101_i32];
        let r = judge(
            Mode::History,
            &keys,
            &[0.01],
            &[0.01],
            &[0.01],
            252,
            0.20,
            -1.0,
            0.30,
            0.5,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_invalid_max_alpha_dd_threshold_errors() {
        let keys = vec![20200101_i32];
        let r = judge(
            Mode::History,
            &keys,
            &[0.01],
            &[0.01],
            &[0.01],
            252,
            0.20,
            0.20,
            -0.1,
            0.5,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_invalid_min_full_sharpe_errors() {
        let keys = vec![20200101_i32];
        let r = judge(
            Mode::History,
            &keys,
            &[0.01],
            &[0.01],
            &[0.01],
            252,
            0.20,
            0.20,
            0.30,
            f64::NAN,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_recent_zero_recent_days_errors() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let keys = consecutive_keys(start, 10);
        let r = judge(
            Mode::Recent,
            &keys,
            &[0.001_f64; 10],
            &[0.001_f64; 10],
            &[0.001_f64; 10],
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            0,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_strategy_with_nan_errors() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let keys = consecutive_keys(start, 10);
        let mut s = vec![0.001_f64; 10];
        s[3] = f64::NAN;
        let r = judge(
            Mode::History,
            &keys,
            &s,
            &[0.001_f64; 10],
            &[0.001_f64; 10],
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    #[test]
    fn judge_invalid_date_key_errors() {
        let r = judge(
            Mode::History,
            &[0_i32, 20200101],
            &[0.001, 0.002],
            &[0.001, 0.002],
            &[0.001, 0.002],
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        );
        assert!(matches!(r, Err(WbtError::InvalidInput(_))));
    }

    // ===========================================================================
    // judge: history mode
    // ===========================================================================

    /// 单年样本：strat 每日恒定；long/bench 由闭包按 index 生成（用于构造可控的 alpha）。
    fn year_block(
        year: i32,
        n: usize,
        strat_per_day: f64,
        long_fn: impl Fn(usize) -> f64,
        bench_fn: impl Fn(usize) -> f64,
    ) -> (Vec<i32>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let start = chrono::NaiveDate::from_ymd_opt(year, 1, 2).unwrap();
        let mut k = Vec::new();
        let (mut s, mut b, mut l) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            k.push(date_key_from_nd(start + chrono::Duration::days(i as i64)));
            s.push(strat_per_day);
            l.push(long_fn(i));
            b.push(bench_fn(i));
        }
        (k, s, b, l)
    }

    // long/bench 漂移方向相反、形状不同 → alpha 非退化。漂移远大于振荡时单调，
    // 用于"超额大幅下行 / 上行"这类需要明确符号与回撤的场景。
    fn drift_down_long(i: usize) -> f64 {
        -0.0008 + 0.0003 * ((i % 3) as f64 - 1.0)
    }
    fn drift_up_bench(i: usize) -> f64 {
        0.0008 + 0.0003 * ((i % 5) as f64 - 2.0)
    }

    /// history 三路 OR —— 仅靠「绝对收益>0」过关（超额下行：alpha_return<0 且当年回撤≥阈值）。
    /// 全样本硬门故意放宽（max_alpha_dd_threshold=1.0, min_full_sharpe=-1e9），让该用例只
    /// 校验「绝对收益>0 即可解锁年度通过 + 不被两道全样本硬门否决」的行为。
    #[test]
    fn history_year_passes_via_abs_return_only() {
        let (k, s, b, l) = year_block(2020, 130, 0.001, drift_down_long, drift_up_bench);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            1.0,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        let ym = r.get("yearly_metrics").and_then(|v| v.as_array()).unwrap();
        let y = &ym[0];
        assert!(y["abs_return"].as_f64().unwrap() > 0.0);
        assert!(y["alpha_return"].as_f64().unwrap() <= 0.0);
        assert!(y["alpha_max_drawdown"].as_f64().unwrap() >= 0.20);
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(true));
    }

    /// history 三路 OR —— 仅靠「当年超额回撤<阈值」过关（绝对/超额收益均为 0）。
    /// alpha 序列恒为 0 → 全样本 Sharpe=0，门槛取 0.0 强制等于下限、应被 strict > 否决。
    /// 因此把全样本硬门放空（min_full_sharpe=-1e9）来隔离「年度 OR」单点行为。
    #[test]
    fn history_year_passes_via_year_dd_only() {
        // strat=0 → abs_return=0；long==bench → alpha≡0 → alpha_return=0、year_dd=0<阈值。
        let osc = |i: usize| 0.0003 * ((i % 3) as f64 - 1.0);
        let (k, s, b, l) = year_block(2020, 130, 0.0, osc, osc);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            1.0,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        let ym = r.get("yearly_metrics").and_then(|v| v.as_array()).unwrap();
        let y = &ym[0];
        assert!(y["abs_return"].as_f64().unwrap().abs() < 1e-9);
        assert!(y["alpha_return"].as_f64().unwrap().abs() < 1e-9);
        assert!(y["alpha_max_drawdown"].as_f64().unwrap() < 0.20);
        assert_eq!(
            r.get("alpha_degenerate").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(r.get("reason").and_then(|v| v.as_str()), Some(""));
    }

    /// history 三路 OR —— 仅靠「波动率归一多头超额>0」过关（abs=0，当年回撤≥极小阈值）。
    /// alpha 上行 → 全样本 Sharpe > 0.5 触发会让该用例失败；用 min_full_sharpe=-1e9 旁路。
    #[test]
    fn history_year_passes_via_alpha_return_only() {
        // 小漂移、相对振荡较大 → alpha 上行但有下行日 → alpha_return>0 且 year_dd>1e-9。
        let long = |i: usize| 0.0002 + 0.0006 * ((i % 3) as f64 - 1.0);
        let bench = |i: usize| -0.0002 + 0.0006 * ((i % 5) as f64 - 2.0);
        let (k, s, b, l) = year_block(2020, 130, 0.0, long, bench);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            1e-9,
            1.0,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        let ym = r.get("yearly_metrics").and_then(|v| v.as_array()).unwrap();
        let y = &ym[0];
        assert!(y["abs_return"].as_f64().unwrap().abs() < 1e-9);
        assert!(y["alpha_return"].as_f64().unwrap() > 0.0);
        assert!(y["alpha_max_drawdown"].as_f64().unwrap() >= 1e-9);
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(true));
    }

    /// 某完整年三路全败 → is_good=false，reason 点名该年；两道全样本硬门仍按字面行为走
    /// （这里用 max=1.0 / Sharpe=-1e9 旁路，避免被双硬门再次阻断）。
    #[test]
    fn history_fails_when_year_fails_all_three() {
        // strat<0（abs<0）；超额下行（alpha_return<0 且 year_dd≥阈值）。
        let (k, s, b, l) = year_block(2020, 130, -0.001, drift_down_long, drift_up_bench);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            1.0,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            reason.contains("2020") && reason.contains("all three"),
            "reason should name failing year and all-three, got: {reason}"
        );
        // 全样本硬门已恢复返回对应 key。
        assert!(r.contains_key("history_alpha_max_drawdown"));
        assert!(r.contains_key("cond_history_dd_passed"));
        assert!(r.contains_key("history_alpha_sharpe"));
        assert!(r.contains_key("cond_history_sharpe_passed"));
    }

    /// 逐年 AND：一年合格、一年三路全败 → 整体不通过。
    #[test]
    fn history_requires_all_complete_years_pass() {
        let (mut k, mut s, mut b, mut l) =
            year_block(2020, 130, 0.001, drift_down_long, drift_up_bench); // 年1：靠 abs 过
        let (k2, s2, b2, l2) = year_block(2021, 130, -0.001, drift_down_long, drift_up_bench); // 年2：全败
        k.extend(k2);
        s.extend(s2);
        b.extend(b2);
        l.extend(l2);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            1.0,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        assert_eq!(
            r.get("complete_year_count").and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(
            r.get("cond_yearly_passed").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn history_fails_when_no_complete_year() {
        let (k, s, b, l) = build_two_year_samples(0.001, 0.001);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            1.0,
            1.0,
            -1e9,
            500,
            252,
            0,
        )
        .unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(reason.contains("no complete year") || reason.contains("complete year"));
    }

    /// alpha 退化（long vol == 0）→ is_good=false；两道全样本硬门对应 key 一律为 None
    /// （与既有「alpha 派生字段退化时为 null」契约一致）。
    #[test]
    fn history_alpha_degenerate_is_not_good() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap();
        let keys = consecutive_keys(start, 130);
        let s = vec![0.001_f64; 130]; // abs_return>0，但退化时强制 is_good=false
        let long = vec![0.0_f64; 130]; // long vol == 0 -> 退化
        let bench: Vec<f64> = (0..130)
            .map(|i| 0.001 + 0.0005 * ((i % 3) as f64))
            .collect();
        let r = judge(
            Mode::History,
            &keys,
            &s,
            &bench,
            &long,
            252,
            0.20,
            1.0,
            0.30,
            0.5,
            120,
            252,
            0,
        )
        .unwrap();
        assert_eq!(
            r.get("alpha_degenerate").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        assert!(matches!(
            r.get("history_alpha_max_drawdown"),
            Some(Value::Null)
        ));
        assert!(matches!(r.get("history_alpha_sharpe"), Some(Value::Null)));
        assert_eq!(
            r.get("cond_history_dd_passed").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(
            r.get("cond_history_sharpe_passed")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    /// 年度收益接近 0 时用 RETURN_EPSILON：abs≈0、超额下行 → 三路全败 → is_good=false。
    #[test]
    fn history_year_passed_uses_epsilon() {
        let (k, s, b, l) = year_block(2020, 130, 1e-12, drift_down_long, drift_up_bench);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            1.0,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
    }

    // ===========================================================================
    // judge: history 全样本硬门
    // ===========================================================================

    /// 全样本超额回撤 = X。阈值取 X → `dd <= threshold` 通过；阈值取 X - 1e-6 → 不通过。
    /// 用"先大跌再修复"的可控形态把 dd 锁在阈值附近。
    #[test]
    fn history_full_dd_threshold_boundary_is_le() {
        let n = 130usize;
        // long: 前 5 天单边 -0.10（资产腰折 dd=0.5），后 125 天 +0.005 修复（仍保持 0.5 峰值差）。
        // 加 1e-6 噪声避免 long_vol=0；bench 用 0 ± small。
        let long: Vec<f64> = (0..n)
            .map(|i| if i < 5 { -0.10 } else { 0.005 })
            .map(|x| x + 1e-6)
            .collect();
        let bench: Vec<f64> = (0..n).map(|i| 0.001 * ((i % 5) as f64 - 2.0)).collect();
        let strat = vec![0.0_f64; n];
        let keys = consecutive_keys(chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(), n);

        // 第一次：以阈值 1.0 跑 → 必过；同时拿到实际 dd。
        let r = judge(
            Mode::History,
            &keys,
            &strat,
            &bench,
            &long,
            252,
            0.20,
            0.20,
            1.0,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        let dd = r
            .get("history_alpha_max_drawdown")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(
            dd > 0.0 && dd < 1.0,
            "fixture should yield a positive DD < 1.0, got {dd}"
        );
        assert_eq!(
            r.get("cond_history_dd_passed").and_then(|v| v.as_bool()),
            Some(true)
        );

        // 阈值取 dd → `<=` 通过；阈值取 dd - 1e-6 → 不通过。
        let r_eq = judge(
            Mode::History,
            &keys,
            &strat,
            &bench,
            &long,
            252,
            0.20,
            0.20,
            dd,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        assert_eq!(
            r_eq.get("cond_history_dd_passed").and_then(|v| v.as_bool()),
            Some(true),
            "dd {dd} == threshold should pass (`dd <= threshold`)"
        );
        let r_lt = judge(
            Mode::History,
            &keys,
            &strat,
            &bench,
            &long,
            252,
            0.20,
            0.20,
            (dd - 1e-6).max(0.0),
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        assert_eq!(
            r_lt.get("cond_history_dd_passed").and_then(|v| v.as_bool()),
            Some(false),
            "dd {dd} > threshold by 1e-6 should fail"
        );
    }

    /// 全样本超额回撤 > 阈值 → 硬门否决、reason 给出具体值。
    #[test]
    fn history_full_dd_above_threshold_fails() {
        // long 单边下行 -0.005（带极小噪声避免 long_vol=0），bench 微振。alpha 整体下行、
        // 全样本 dd > 0.30。
        let n = 130usize;
        let keys = consecutive_keys(chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(), n);
        let strat = vec![0.0_f64; n];
        let long: Vec<f64> = (0..n)
            .map(|i| -0.005 + 1e-6 * (((i % 3) as f64) - 1.0))
            .collect();
        let bench: Vec<f64> = (0..n).map(|i| 0.001 * ((i % 5) as f64 - 2.0)).collect();
        let r = judge(
            Mode::History,
            &keys,
            &strat,
            &bench,
            &long,
            252,
            0.20,
            0.20,
            0.30,
            -1e9,
            120,
            252,
            0,
        )
        .unwrap();
        let dd = r
            .get("history_alpha_max_drawdown")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(dd > 0.30, "fixture should produce dd > 0.30, got {dd}");
        assert_eq!(
            r.get("cond_history_dd_passed").and_then(|v| v.as_bool()),
            Some(false)
        );
        let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            reason.contains("history full-sample excess drawdown"),
            "{reason}"
        );
    }

    /// 全样本 Sharpe ≤ 阈值 → 硬门否决；反之通过。Sharpe 是严格大于。
    #[test]
    fn history_full_sharpe_gate_is_strict_greater() {
        // long 单边 -0.002（带极小噪声避免 long_vol=0），bench 微振。alpha 均值负 → Sharpe 负。
        let n = 130usize;
        let keys = consecutive_keys(chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(), n);
        let strat = vec![0.0_f64; n];
        let long: Vec<f64> = (0..n)
            .map(|i| -0.002 + 1e-6 * (((i % 3) as f64) - 1.0))
            .collect();
        let bench: Vec<f64> = (0..n).map(|i| 0.001 * ((i % 5) as f64 - 2.0)).collect();
        let r = judge(
            Mode::History,
            &keys,
            &strat,
            &bench,
            &long,
            252,
            0.20,
            0.20,
            1.0,
            0.5,
            120,
            252,
            0,
        )
        .unwrap();
        let sharpe = r
            .get("history_alpha_sharpe")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(
            sharpe.is_finite(),
            "Sharpe should be finite on non-degenerate data"
        );
        assert!(
            sharpe <= 0.5,
            "fixture should produce Sharpe <= 0.5, got {sharpe}"
        );
        assert_eq!(
            r.get("cond_history_sharpe_passed")
                .and_then(|v| v.as_bool()),
            Some(false)
        );

        // 把 min_full_sharpe 调到 Sharpe 之下 → Sharpe gate 通过。
        let r2 = judge(
            Mode::History,
            &keys,
            &strat,
            &bench,
            &long,
            252,
            0.20,
            0.20,
            1.0,
            sharpe - 1.0,
            120,
            252,
            0,
        )
        .unwrap();
        assert_eq!(
            r2.get("cond_history_sharpe_passed")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    /// Sharpe 严格大于 0.5 时，硬门应通过（验证「>」逻辑）。
    #[test]
    fn history_full_sharpe_above_default_passes() {
        // long 上行（0.01 + 噪声），bench 微振下行。alpha 整体上行 → Sharpe 应 > 0.5。
        let n = 252usize;
        let keys = consecutive_keys(chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(), n);
        let strat = vec![0.0_f64; n];
        let long: Vec<f64> = (0..n)
            .map(|i| 0.01 + 0.001 * (((i % 7) as f64) - 3.0))
            .collect();
        let bench: Vec<f64> = (0..n).map(|i| 0.0005 * (((i % 5) as f64) - 2.0)).collect();
        let r = judge(
            Mode::History,
            &keys,
            &strat,
            &bench,
            &long,
            252,
            0.20,
            0.20,
            1.0,
            0.5,
            120,
            252,
            0,
        )
        .unwrap();
        let sharpe = r
            .get("history_alpha_sharpe")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(
            sharpe.is_finite(),
            "Sharpe should be finite on non-degenerate data"
        );
        assert!(
            sharpe > 0.5,
            "fixture should produce Sharpe > 0.5, got {sharpe}"
        );
        assert_eq!(
            r.get("cond_history_sharpe_passed")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    /// 默认参数下，年度 OR 通过 + 双硬门中 Sharpe 不达标 → is_good=false 且 reason 点名 Sharpe。
    #[test]
    fn history_default_sharpe_gate_rejects_low_quality() {
        let (k, s, b, l) = year_block(2020, 130, 0.001, drift_down_long, drift_up_bench);
        let r = judge(
            Mode::History,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            0,
        )
        .unwrap();
        // 年内绝对收益 > 0，所以 cond_yearly_passed=true；但 alpha 漂移向下，Sharpe 应为负。
        let sharpe = r
            .get("history_alpha_sharpe")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(
            sharpe < 0.5,
            "fixture should produce Sharpe < 0.5, got {sharpe}"
        );
        assert_eq!(
            r.get("cond_history_sharpe_passed")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
        let reason = r.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(reason.contains("Sharpe"), "{reason}");
    }

    // ===========================================================================
    // judge: recent mode
    // ===========================================================================

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
            b.push(0.0003 * ((i % 3) as f64 - 1.0));
            l.push(0.0003 * ((i % 5) as f64 - 2.0));
        }
        let r = judge(
            Mode::Recent,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        )
        .unwrap();
        assert_eq!(r.get("mode").and_then(|v| v.as_str()), Some("recent"));
        assert_eq!(
            r.get("alpha_degenerate").and_then(|v| v.as_bool()),
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
            "history_window_empty",
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

    #[test]
    fn recent_fails_when_alpha_degenerate() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let mut k = Vec::new();
        let mut s = Vec::new();
        let mut b = Vec::new();
        let l = vec![0.0_f64; 500]; // long vol == 0 -> degenerate
        for i in 0..500 {
            k.push(date_key_from_nd(start + chrono::Duration::days(i as i64)));
            s.push(0.001);
            b.push(0.001 + 0.0005 * ((i % 3) as f64));
        }
        let r = judge(
            Mode::Recent,
            &k,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        )
        .unwrap();
        assert_eq!(
            r.get("alpha_degenerate").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
    }

    /// F8 fix: short sample -> history_alpha_max_drawdown_excl_recent is Null.
    #[test]
    fn recent_history_dd_excl_recent_is_null_when_short() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let keys = consecutive_keys(start, 200);
        let s = vec![0.001_f64; 200];
        let b: Vec<f64> = (0..200).map(|i| 0.0003 * ((i % 3) as f64 - 1.0)).collect();
        let l: Vec<f64> = (0..200).map(|i| 0.0003 * ((i % 5) as f64 - 2.0)).collect();
        let r = judge(
            Mode::Recent,
            &keys,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        )
        .unwrap();
        assert_eq!(
            r.get("history_window_empty").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(matches!(
            r.get("history_alpha_max_drawdown_excl_recent"),
            Some(Value::Null)
        ));
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
    }

    /// F7 fix: head below min_history_days flips history_window_empty=true.
    #[test]
    fn recent_history_short_below_min_history_days_is_window_empty() {
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let keys = consecutive_keys(start, 255);
        let s = vec![0.001_f64; 255];
        let b: Vec<f64> = (0..255).map(|i| 0.0003 * ((i % 3) as f64 - 1.0)).collect();
        let l: Vec<f64> = (0..255).map(|i| 0.0003 * ((i % 5) as f64 - 2.0)).collect();
        let r = judge(
            Mode::Recent,
            &keys,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        )
        .unwrap();
        assert_eq!(
            r.get("history_window_empty").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(matches!(
            r.get("history_alpha_max_drawdown_excl_recent"),
            Some(Value::Null)
        ));
    }

    /// recent 三路 OR —— 仅靠「近期超额回撤<阈值」过关（abs<0、alpha_return≈0），
    /// 且近期回撤严格小于历史回撤这一硬门成立 → is_good=true。
    #[test]
    fn recent_passes_via_dd_branch_only() {
        // 头段 148 天超额大幅下行（历史回撤大）；尾段 252 天 long/bench≡0 → 近期 alpha≡0。
        let head = 148usize;
        let total = head + 252;
        let start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let keys = consecutive_keys(start, total);
        let s = vec![-0.0001_f64; total]; // 近期 abs_return<0
        let l: Vec<f64> = (0..total)
            .map(|i| if i < head { drift_down_long(i) } else { 0.0 })
            .collect();
        let b: Vec<f64> = (0..total)
            .map(|i| if i < head { drift_up_bench(i) } else { 0.0 })
            .collect();
        let r = judge(
            Mode::Recent,
            &keys,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        )
        .unwrap();
        assert_eq!(
            r.get("alpha_degenerate").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert!(r.get("recent_abs_return").and_then(|v| v.as_f64()).unwrap() < 0.0);
        assert!(
            r.get("recent_alpha_max_drawdown")
                .and_then(|v| v.as_f64())
                .unwrap()
                < 0.20
        );
        assert_eq!(
            r.get("cond_recent_return_passed").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            r.get("cond_recent_dd_passed").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(true));
    }

    /// 历史回撤硬门是严格小于：近期回撤恰好等于历史回撤时判 False。
    #[test]
    fn recent_strict_history_rejects_equal_dd() {
        // 头段 [0,252) 与尾段 [252,504) 取相同的周期化形状 → 两窗 alpha 逐点相等 →
        // recent_dd == history_excl，严格 `<` 应为 false。
        let total = 504usize;
        let start = chrono::NaiveDate::from_ymd_opt(2019, 1, 1).unwrap();
        let keys = consecutive_keys(start, total);
        let s = vec![0.001_f64; total]; // 收益侧 OR 通过（abs_return>0）
        let l: Vec<f64> = (0..total)
            .map(|i| 0.0002 + 0.0006 * (((i % 252) % 3) as f64 - 1.0))
            .collect();
        let b: Vec<f64> = (0..total)
            .map(|i| -0.0002 + 0.0006 * (((i % 252) % 5) as f64 - 2.0))
            .collect();
        let r = judge(
            Mode::Recent,
            &keys,
            &s,
            &b,
            &l,
            252,
            0.20,
            0.20,
            0.30,
            0.5,
            120,
            252,
            60,
        )
        .unwrap();
        assert_eq!(
            r.get("cond_recent_return_passed").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            r.get("cond_recent_dd_passed").and_then(|v| v.as_bool()),
            Some(false),
            "recent_dd == history_excl must NOT pass the strict `<` gate"
        );
        assert_eq!(r.get("is_good").and_then(|v| v.as_bool()), Some(false));
    }
}
