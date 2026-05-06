//! Top-N drawdown analytics.
//!
//! Reuses the underwater helpers from `daily_performance.rs`
//! (`calc_underwater`, `calc_underwater_valley`, `calc_underwater_peak`,
//! `calc_underwater_recovery`) to walk the cumulative-return curve,
//! repeatedly pull the deepest unrecorded drawdown, and return a
//! DataFrame with the rs-czsc-compatible schema:
//!
//!   回撤开始 / 回撤结束 / 回撤修复 / 净值回撤 / 回撤天数 / 恢复天数 / 新高间隔
//!
//! `回撤修复` / `恢复天数` / `新高间隔` are nullable (drawdown not yet
//! recovered as of the last sample). `净值回撤` is the cumulative-return
//! delta at the valley (a non-positive f64).

use crate::core::daily_performance::{
    calc_underwater, calc_underwater_peak, calc_underwater_recovery, calc_underwater_valley,
};
use crate::core::errors::WbtError;
use anyhow::anyhow;
use chrono::NaiveDate;
use polars::{df, frame::DataFrame};

/// Identify the top-N drawdown windows in a return series.
pub fn top_drawdowns(
    returns: &[f64],
    dates: &[NaiveDate],
    top: Option<usize>,
) -> Result<DataFrame, WbtError> {
    if returns.len() != dates.len() {
        return Err(WbtError::Unexpected(anyhow!(
            "returns.len() ({}) must equal dates.len() ({})",
            returns.len(),
            dates.len()
        )));
    }
    if returns.is_empty() {
        return Err(WbtError::Unexpected(anyhow!(
            "returns must not be empty"
        )));
    }

    let top = top.unwrap_or(10);
    let mut underwater = calc_underwater(returns);

    let mut drawdown_start_dates = Vec::with_capacity(top);
    let mut drawdown_end_dates = Vec::with_capacity(top);
    let mut drawdowns = Vec::with_capacity(top);
    let mut drawdown_days = Vec::with_capacity(top);
    let mut recovery_dates = Vec::with_capacity(top);
    let mut recovery_days = Vec::with_capacity(top);
    let mut new_high_interval = Vec::with_capacity(top);

    for _ in 0..top {
        let valley = match calc_underwater_valley(&underwater) {
            Some(v) => v,
            None => break,
        };
        let peak = calc_underwater_peak(&underwater, valley);
        let recovery = calc_underwater_recovery(&underwater, valley);

        drawdown_start_dates.push(dates[peak]);
        drawdown_end_dates.push(dates[valley]);
        drawdowns.push(underwater[valley]);
        let dd_days = (dates[valley] - dates[peak]).num_days();
        drawdown_days.push(dd_days);

        if let Some(rec) = recovery {
            recovery_dates.push(Some(dates[rec]));
            let rec_days = (dates[rec] - dates[valley]).num_days();
            recovery_days.push(Some(rec_days));
            new_high_interval.push(Some(dd_days + rec_days));
            underwater[peak..rec].fill(0.0);
        } else {
            recovery_dates.push(None);
            recovery_days.push(None);
            new_high_interval.push(None);
            underwater[peak..returns.len()].fill(0.0);
        }
    }

    let df = df!(
        "回撤开始" => drawdown_start_dates,
        "回撤结束" => drawdown_end_dates,
        "回撤修复" => recovery_dates,
        "净值回撤" => drawdowns,
        "回撤天数" => drawdown_days,
        "恢复天数" => recovery_days,
        "新高间隔" => new_high_interval,
    )
    .map_err(WbtError::Polars)?;
    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_drawdowns_matches_rs_czsc_fixture() {
        // Same fixture as rs-czsc/crates/czsc-utils/src/top_drawdowns.rs::test_top_drawdowns
        let dates: Vec<NaiveDate> = (1..=9)
            .map(|d| NaiveDate::from_ymd_opt(2024, 12, d).unwrap())
            .collect();
        let returns = [1.0, 2.0, -3.0, -1.0, 5.0, 1.0, -7.0, 6.0, 16.0];
        let df = top_drawdowns(&returns, &dates, Some(2)).unwrap();
        assert_eq!(df.height(), 2);

        let dd = df.column("净值回撤").unwrap().f64().unwrap();
        assert_eq!(dd.get(0), Some(-7.0));
        assert_eq!(dd.get(1), Some(-4.0));

        let dd_days = df.column("回撤天数").unwrap().i64().unwrap();
        assert_eq!(dd_days.get(0), Some(1));
        assert_eq!(dd_days.get(1), Some(2));

        let rec_days = df.column("恢复天数").unwrap().i64().unwrap();
        assert_eq!(rec_days.get(0), Some(2));
        assert_eq!(rec_days.get(1), Some(1));

        let nhi = df.column("新高间隔").unwrap().i64().unwrap();
        assert_eq!(nhi.get(0), Some(3));
        assert_eq!(nhi.get(1), Some(3));
    }

    #[test]
    fn top_drawdowns_handles_unrecovered_tail() {
        let dates: Vec<NaiveDate> = (1..=9)
            .map(|d| NaiveDate::from_ymd_opt(2024, 12, d).unwrap())
            .collect();
        let returns = [1.0, 2.0, -3.0, -1.0, 5.0, 1.0, -7.0, -6.0, -16.0];
        let df = top_drawdowns(&returns, &dates, Some(10)).unwrap();
        // The deepest drawdown (extracted first) is unrecovered →
        // 回撤修复 / 恢复天数 / 新高间隔 should be null on row 0.
        assert!(df.height() >= 1);
        let recovery = df.column("回撤修复").unwrap();
        assert!(recovery.get(0).unwrap().is_null());
        let recovery_days = df.column("恢复天数").unwrap();
        assert!(recovery_days.get(0).unwrap().is_null());
    }

    #[test]
    fn top_drawdowns_rejects_length_mismatch() {
        let dates = vec![NaiveDate::from_ymd_opt(2024, 12, 1).unwrap()];
        let returns = [1.0, 2.0];
        let res = top_drawdowns(&returns, &dates, Some(1));
        assert!(res.is_err());
    }
}
