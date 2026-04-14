//! 从 daily_return 宽表计算年度收益长表
//!
//! 参考：vista/utils/yearly_return.py 中的 `calculate_yearly_returns`
//! 核心公式：每年复利收益 = (1+r1)*(1+r2)*...*(1+rn) - 1

use crate::core::errors::WbtError;
use polars::prelude::*;

/// 从 daily_return 宽表计算年度收益长表
///
/// # 参数
/// - `wide_df`: 宽表，必须包含 `date` 列（`DataType::Date`），其余列为各 symbol 的日收益率（`f64`）。
///   约定：策略整体收益以 `total` 列存在，会作为 `symbol = "total"` 出现在结果中。
/// - `min_days`: 每年最少交易日数量；不足的 `(year, symbol)` 组合会被跳过。
///
/// # 返回
/// 长表 `[year: i32, symbol: String, return: f64]`，按 `(year, symbol)` 升序排列。
/// 若没有任何组合满足 `min_days`，返回与 schema 一致的空 DataFrame。
pub fn compute_yearly_returns(wide_df: &DataFrame, min_days: usize) -> Result<DataFrame, WbtError> {
    use chrono::{Datelike, NaiveDate};
    use std::collections::BTreeMap;

    // 1. date 列 → year 向量
    let date_ca = wide_df.column("date")?.as_materialized_series().date()?.clone();
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let mut years: Vec<i32> = Vec::with_capacity(date_ca.len());
    for opt in date_ca.physical().iter() {
        let days =
            opt.ok_or_else(|| WbtError::NoneValue("date column contains null".into()))?;
        let nd = epoch + chrono::Duration::days(days as i64);
        years.push(nd.year());
    }

    // 2. 所有非 date 列
    let symbol_cols: Vec<String> = wide_df
        .get_column_names()
        .iter()
        .filter_map(|n| {
            let s = n.as_str();
            if s == "date" { None } else { Some(s.to_string()) }
        })
        .collect();

    // 3. 按 (year, symbol) 聚合 + min_days 过滤 + 复利
    let mut out_years: Vec<i32> = Vec::new();
    let mut out_symbols: Vec<String> = Vec::new();
    let mut out_returns: Vec<f64> = Vec::new();

    for sym in &symbol_cols {
        let ca = wide_df.column(sym)?.as_materialized_series().f64()?.clone();
        let mut by_year: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
        for (i, opt_r) in ca.iter().enumerate() {
            if let Some(r) = opt_r {
                by_year.entry(years[i]).or_default().push(r);
            }
        }
        for (year, rs) in by_year {
            if rs.len() >= min_days {
                let yearly_ret = rs.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
                out_years.push(year);
                out_symbols.push(sym.clone());
                out_returns.push(yearly_ret);
            }
        }
    }

    // 4. (year asc, symbol asc) 排序
    let mut idx: Vec<usize> = (0..out_years.len()).collect();
    idx.sort_by(|&a, &b| {
        out_years[a]
            .cmp(&out_years[b])
            .then_with(|| out_symbols[a].cmp(&out_symbols[b]))
    });
    let sorted_years: Vec<i32> = idx.iter().map(|&i| out_years[i]).collect();
    let sorted_symbols: Vec<String> = idx.iter().map(|&i| out_symbols[i].clone()).collect();
    let sorted_returns: Vec<f64> = idx.iter().map(|&i| out_returns[i]).collect();

    DataFrame::new(vec![
        Series::new("year".into(), sorted_years).into_column(),
        Series::new("symbol".into(), sorted_symbols).into_column(),
        Series::new("return".into(), sorted_returns).into_column(),
    ])
    .map_err(WbtError::Polars)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一张测试用宽表：`date` 是 `Date` 类型，其他列是 `f64`
    fn wide(dates: &[&str], symbol_cols: &[(&str, Vec<Option<f64>>)]) -> DataFrame {
        let date_epoch_days: Vec<i32> = dates
            .iter()
            .map(|s| {
                let nd = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap();
                let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                (nd - epoch).num_days() as i32
            })
            .collect();

        let mut cols: Vec<Column> = Vec::with_capacity(symbol_cols.len() + 1);
        cols.push(
            Series::new("date".into(), date_epoch_days)
                .cast(&DataType::Date)
                .unwrap()
                .into_column(),
        );
        for (name, values) in symbol_cols {
            cols.push(Series::new((*name).into(), values.clone()).into_column());
        }
        DataFrame::new(cols).unwrap()
    }

    fn row(df: &DataFrame, i: usize) -> (i32, String, f64) {
        let y = df
            .column("year")
            .unwrap()
            .as_materialized_series()
            .i32()
            .unwrap()
            .get(i)
            .unwrap();
        let s = df
            .column("symbol")
            .unwrap()
            .as_materialized_series()
            .str()
            .unwrap()
            .get(i)
            .unwrap()
            .to_string();
        let r = df
            .column("return")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .get(i)
            .unwrap();
        (y, s, r)
    }

    #[test]
    fn single_year_single_symbol_compound_return() {
        // 三天日收益：0.01, 0.02, -0.01
        // 年度复利 = 1.01 * 1.02 * 0.99 - 1 = 0.019898
        let df = wide(
            &["2020-03-01", "2020-03-02", "2020-03-03"],
            &[("A", vec![Some(0.01), Some(0.02), Some(-0.01)])],
        );
        let out = compute_yearly_returns(&df, 1).unwrap();
        assert_eq!(out.height(), 1);
        let (y, s, r) = row(&out, 0);
        assert_eq!(y, 2020);
        assert_eq!(s, "A");
        let expected = 1.01_f64 * 1.02 * 0.99 - 1.0;
        assert!(
            (r - expected).abs() < 1e-12,
            "expected {expected}, got {r}"
        );
    }

    #[test]
    fn multi_year_separates_by_year() {
        // 跨两年，每年两天
        let df = wide(
            &["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"],
            &[("A", vec![Some(0.10), Some(0.10), Some(-0.05), Some(-0.05)])],
        );
        let out = compute_yearly_returns(&df, 1).unwrap();
        assert_eq!(out.height(), 2);

        let (y0, _, r0) = row(&out, 0);
        let (y1, _, r1) = row(&out, 1);
        assert_eq!(y0, 2020);
        assert_eq!(y1, 2021);
        assert!((r0 - (1.10_f64 * 1.10 - 1.0)).abs() < 1e-12);
        assert!((r1 - (0.95_f64 * 0.95 - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn min_days_filters_short_years() {
        // A 在 2020 只有 2 天，2021 有 3 天；min_days=3 → 2020 被过滤
        let df = wide(
            &[
                "2020-12-30",
                "2020-12-31",
                "2021-01-04",
                "2021-01-05",
                "2021-01-06",
            ],
            &[(
                "A",
                vec![Some(0.01), Some(0.02), Some(0.03), Some(0.04), Some(0.05)],
            )],
        );
        let out = compute_yearly_returns(&df, 3).unwrap();
        assert_eq!(out.height(), 1);
        let (y, _, _) = row(&out, 0);
        assert_eq!(y, 2021);
    }

    #[test]
    fn total_column_emitted_as_symbol() {
        // total 列也当作普通 symbol 处理
        let df = wide(
            &["2020-01-02", "2020-01-03"],
            &[
                ("A", vec![Some(0.01), Some(-0.01)]),
                ("total", vec![Some(0.005), Some(-0.005)]),
            ],
        );
        let out = compute_yearly_returns(&df, 1).unwrap();
        assert_eq!(out.height(), 2);
        let symbols: Vec<String> = (0..out.height()).map(|i| row(&out, i).1).collect();
        assert!(symbols.contains(&"A".to_string()));
        assert!(symbols.contains(&"total".to_string()));
    }

    #[test]
    fn output_sorted_by_year_then_symbol() {
        // 多 symbol + 跨年 → 验证排序 (year asc, symbol asc)
        let df = wide(
            &["2020-06-01", "2021-06-01"],
            &[
                ("B", vec![Some(0.02), Some(0.02)]),
                ("A", vec![Some(0.01), Some(0.01)]),
            ],
        );
        let out = compute_yearly_returns(&df, 1).unwrap();
        assert_eq!(out.height(), 4);
        let rows: Vec<(i32, String)> =
            (0..4).map(|i| (row(&out, i).0, row(&out, i).1)).collect();
        assert_eq!(
            rows,
            vec![
                (2020, "A".into()),
                (2020, "B".into()),
                (2021, "A".into()),
                (2021, "B".into()),
            ]
        );
    }

    #[test]
    fn empty_when_all_below_min_days() {
        let df = wide(
            &["2020-01-02"],
            &[("A", vec![Some(0.01)])],
        );
        let out = compute_yearly_returns(&df, 10).unwrap();
        assert_eq!(out.height(), 0);
        // schema 仍然正确
        assert_eq!(
            out.get_column_names()
                .iter()
                .map(|s| s.as_str().to_string())
                .collect::<Vec<_>>(),
            vec!["year", "symbol", "return"]
        );
    }

    #[test]
    fn none_values_excluded_from_day_count() {
        // A 在 2020 有 2 天非空 + 1 天 None；min_days=2 应通过（只数非空）
        // 年度收益只对非空天数复利
        let df = wide(
            &["2020-01-02", "2020-01-03", "2020-01-04"],
            &[("A", vec![Some(0.01), None, Some(0.02)])],
        );
        let out = compute_yearly_returns(&df, 2).unwrap();
        assert_eq!(out.height(), 1);
        let (_, _, r) = row(&out, 0);
        let expected = 1.01_f64 * 1.02 - 1.0;
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn none_values_and_min_days_boundary() {
        // min_days=3 但只有 2 天非空 → 过滤
        let df = wide(
            &["2020-01-02", "2020-01-03", "2020-01-04"],
            &[("A", vec![Some(0.01), None, Some(0.02)])],
        );
        let out = compute_yearly_returns(&df, 3).unwrap();
        assert_eq!(out.height(), 0);
    }
}
