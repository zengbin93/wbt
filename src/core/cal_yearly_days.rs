use chrono::{Datelike, NaiveDate};
use std::collections::{BTreeSet, HashMap};

/// 计算年度交易日数量（与 czsc.eda.cal_yearly_days 行为完全一致）。
///
/// 业务规则（全部在 Rust 内）：
/// 1. 输入为空 → panic（PyO3 wrapper 转 PyException）；
/// 2. 去重后样本跨度 < 365 天 → 通过 `log::warn!` 提示，返回 252；
/// 3. 否则按自然年聚合取交易日数 max，钳制到 [1, 365]。
pub fn cal_yearly_days(dates: &[NaiveDate]) -> i64 {
    assert!(!dates.is_empty(), "输入的日期数量必须大于0");

    let dedup: BTreeSet<NaiveDate> = dates.iter().copied().collect();
    let min = *dedup.iter().next().unwrap();
    let max = *dedup.iter().next_back().unwrap();

    if (max - min).num_days() < 365 {
        log::warn!("时间跨度小于一年，直接返回 252");
        return 252;
    }

    let mut per_year: HashMap<i32, i64> = HashMap::new();
    for d in &dedup {
        *per_year.entry(d.year()).or_insert(0) += 1;
    }
    per_year.values().copied().max().unwrap_or(0).min(365)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    #[should_panic(expected = "输入的日期数量必须大于0")]
    fn empty_input_panics() {
        cal_yearly_days(&[]);
    }

    #[test]
    fn span_less_than_one_year_returns_252() {
        let dates: Vec<NaiveDate> = (1..=200).map(|i| d(2024, 1, 1) + chrono::Duration::days(i)).collect();
        assert_eq!(cal_yearly_days(&dates), 252);
    }

    #[test]
    fn multi_year_returns_max_year_count() {
        // 2022 有 260 个交易日，2023 有 250 个，跨度 > 365
        let mut dates = Vec::new();
        for i in 0..260 {
            dates.push(d(2022, 1, 1) + chrono::Duration::days(i));
        }
        for i in 0..250 {
            dates.push(d(2023, 1, 1) + chrono::Duration::days(i));
        }
        // 加一个 2024 年初的日期保证 max-min > 365
        dates.push(d(2024, 2, 1));
        assert_eq!(cal_yearly_days(&dates), 260);
    }

    #[test]
    fn clamped_to_365() {
        // 极端情况：连续每天都有数据（>365 天/年时仍钳到 365）
        let dates: Vec<NaiveDate> = (0..800).map(|i| d(2020, 1, 1) + chrono::Duration::days(i)).collect();
        assert!(cal_yearly_days(&dates) <= 365);
    }
}
