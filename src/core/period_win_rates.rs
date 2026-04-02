use crate::core::utils::RoundToNthDigit;
use chrono::{Datelike, NaiveDate};

// ---------------------------------------------------------------------------
// PeriodWinRates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PeriodWinRates {
    pub week: f64,
    pub month: f64,
    pub quarter: f64,
    pub year: f64,
}

impl Default for PeriodWinRates {
    fn default() -> Self {
        PeriodWinRates {
            week: 0.0,
            month: 0.0,
            quarter: 0.0,
            year: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Period key types — (year, discriminator) pairs that change when a new
// natural period starts.
// ---------------------------------------------------------------------------

#[inline]
fn date_key_to_naive_date(dk: i32) -> NaiveDate {
    let y = dk / 10000;
    let m = (dk / 100) % 100;
    let d = dk % 100;
    NaiveDate::from_ymd_opt(y, m as u32, d as u32)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
}

#[inline]
fn week_key(nd: NaiveDate) -> (i32, u32) {
    let iw = nd.iso_week();
    (iw.year(), iw.week())
}

#[inline]
fn month_key(nd: NaiveDate) -> (i32, u32) {
    (nd.year(), nd.month())
}

#[inline]
fn quarter_key(nd: NaiveDate) -> (i32, u32) {
    let q = (nd.month() - 1) / 3 + 1;
    (nd.year(), q)
}

#[inline]
fn year_key(nd: NaiveDate) -> i32 {
    nd.year()
}

// ---------------------------------------------------------------------------
// Generic single-pass accumulator for sorted period data
// ---------------------------------------------------------------------------

/// Accumulates daily returns into period sums using a running-key approach.
/// Returns (wins, total) where wins = periods with sum > 0.
fn period_win_count<K, F>(date_keys: &[i32], returns: &[f64], key_fn: F) -> (usize, usize)
where
    K: PartialEq,
    F: Fn(NaiveDate) -> K,
{
    if date_keys.is_empty() {
        return (0, 0);
    }

    let mut wins = 0usize;
    let mut total = 0usize;
    let mut current_key = key_fn(date_key_to_naive_date(date_keys[0]));
    let mut period_sum = returns[0];

    for i in 1..date_keys.len() {
        let nd = date_key_to_naive_date(date_keys[i]);
        let key = key_fn(nd);
        if key != current_key {
            total += 1;
            if period_sum > 0.0 {
                wins += 1;
            }
            current_key = key;
            period_sum = returns[i];
        } else {
            period_sum += returns[i];
        }
    }
    // flush final period
    total += 1;
    if period_sum > 0.0 {
        wins += 1;
    }

    (wins, total)
}

// ---------------------------------------------------------------------------
// Year win rate with min-days filter
// ---------------------------------------------------------------------------

fn year_win_rate(date_keys: &[i32], returns: &[f64], yearly_days: i64) -> f64 {
    if date_keys.is_empty() {
        return 0.0;
    }

    let min_days = yearly_days / 2;

    // Collect per-year (sum, count) in a single pass
    let mut wins = 0usize;
    let mut total = 0usize;
    let mut current_year = year_key(date_key_to_naive_date(date_keys[0]));
    let mut period_sum = returns[0];
    let mut period_count = 1usize;

    let flush = |wins: &mut usize, total: &mut usize, sum: f64, count: usize| {
        if count as i64 >= min_days {
            *total += 1;
            if sum > 0.0 {
                *wins += 1;
            }
        }
    };

    for i in 1..date_keys.len() {
        let nd = date_key_to_naive_date(date_keys[i]);
        let year = year_key(nd);
        if year != current_year {
            flush(&mut wins, &mut total, period_sum, period_count);
            current_year = year;
            period_sum = returns[i];
            period_count = 1;
        } else {
            period_sum += returns[i];
            period_count += 1;
        }
    }
    flush(&mut wins, &mut total, period_sum, period_count);

    if total == 0 {
        return 0.0;
    }
    (wins as f64 / total as f64).round_to_4_digit()
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn period_win_rates(
    date_keys: &[i32],
    returns: &[f64],
    yearly_days: i64,
) -> PeriodWinRates {
    if date_keys.is_empty() {
        return PeriodWinRates::default();
    }

    let rate = |wins: usize, total: usize| -> f64 {
        if total == 0 {
            0.0
        } else {
            (wins as f64 / total as f64).round_to_4_digit()
        }
    };

    let (ww, wt) = period_win_count(date_keys, returns, week_key);
    let (mw, mt) = period_win_count(date_keys, returns, month_key);
    let (qw, qt) = period_win_count(date_keys, returns, quarter_key);

    PeriodWinRates {
        week: rate(ww, wt),
        month: rate(mw, mt),
        quarter: rate(qw, qt),
        year: year_win_rate(date_keys, returns, yearly_days),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- empty input ---
    #[test]
    fn empty_input_returns_default() {
        let result = period_win_rates(&[], &[], 252);
        assert_eq!(result, PeriodWinRates::default());
    }

    // --- single day ---
    // One day → one week, one month, one quarter (all win rate = 1.0 if return > 0)
    // Year has only 1 day which is < yearly_days/2, so it is filtered out → 0.0
    #[test]
    fn single_positive_day() {
        let result = period_win_rates(&[20240101], &[0.01], 252);
        assert_eq!(result.week, 1.0);
        assert_eq!(result.month, 1.0);
        assert_eq!(result.quarter, 1.0);
        assert_eq!(result.year, 0.0); // filtered: 1 day < 252/2 = 126
    }

    #[test]
    fn single_negative_day() {
        let result = period_win_rates(&[20240101], &[-0.01], 252);
        assert_eq!(result.week, 0.0);
        assert_eq!(result.month, 0.0);
        assert_eq!(result.quarter, 0.0);
        assert_eq!(result.year, 0.0);
    }

    // --- two weeks with opposite returns → week = 0.5 ---
    #[test]
    fn two_weeks_opposite_returns() {
        // Week 1 of 2024: Jan 1-5, Week 2: Jan 8-12
        let date_keys = [20240101, 20240102, 20240103, 20240108, 20240109, 20240110];
        let returns = [0.01, 0.01, 0.01, -0.01, -0.01, -0.01];
        let result = period_win_rates(&date_keys, &returns, 252);
        assert_eq!(result.week, 0.5); // week1: +0.03 → win; week2: -0.03 → loss
    }

    // --- year filter: short year excluded ---
    #[test]
    fn year_filter_excludes_short_year() {
        // 2023 has 1 day (< 252/2=126), 2024 has 200 days
        // Build 200 dates in 2024 from Jan 2 onward (weekdays only approx.)
        let mut date_keys = vec![20231229i32]; // 1 day in 2023
        let mut returns = vec![-0.01f64];
        // Add 200 positive-return days in 2024
        for i in 1..=200i32 {
            // Use a simple sequential key: 20240101 + offset (not all valid dates, but
            // our code just parses via from_ymd_opt, fallback to 1970-01-01 on invalid)
            // Use real dates: Jan has 31, Feb has 29 (2024 leap), Mar has 31, ...
            // Easier: use 20240000 + day-of-year style won't work. Use explicit list.
            // Actually just use days from 20240102 cycling through valid dates.
            let month = ((i - 1) / 28) + 1;
            let day = ((i - 1) % 28) + 1;
            if month <= 12 {
                let dk = 20240000 + month * 100 + day;
                date_keys.push(dk);
                returns.push(0.01);
            }
        }
        let result = period_win_rates(&date_keys, &returns, 252);
        // 2023 filtered out (1 day < 126); 2024 included and all positive → year = 1.0
        assert_eq!(result.year, 1.0);
    }

    // --- year filter: sufficient year included ---
    #[test]
    fn year_filter_includes_sufficient_year() {
        // Build a year with exactly yearly_days/2 days (boundary: included)
        let yearly_days = 10i64;
        let min_days = (yearly_days / 2) as usize; // 5

        // 5 positive days in 2024
        let date_keys: Vec<i32> = (1..=min_days as i32)
            .map(|d| 20240100 + d) // 20240101..20240105
            .collect();
        let returns: Vec<f64> = vec![0.01; min_days];

        let result = period_win_rates(&date_keys, &returns, yearly_days);
        assert_eq!(result.year, 1.0); // included, all positive
    }

    #[test]
    fn year_filter_excludes_exactly_below_threshold() {
        let yearly_days = 10i64;
        let below_min = (yearly_days / 2 - 1) as usize; // 4

        let date_keys: Vec<i32> = (1..=below_min as i32)
            .map(|d| 20240100 + d)
            .collect();
        let returns: Vec<f64> = vec![0.01; below_min];

        let result = period_win_rates(&date_keys, &returns, yearly_days);
        assert_eq!(result.year, 0.0); // excluded
    }

    // --- quarter win rate with mixed quarters ---
    #[test]
    fn quarter_mixed() {
        // Q1 2024: Jan = positive
        // Q2 2024: Apr = negative
        // Q3 2024: Jul = positive
        let date_keys = [
            20240102, 20240103, // Q1 positive
            20240401, 20240402, // Q2 negative
            20240701, 20240702, // Q3 positive
        ];
        let returns = [0.01, 0.01, -0.01, -0.01, 0.01, 0.01];
        let result = period_win_rates(&date_keys, &returns, 252);
        // 2 wins out of 3 quarters = 0.6667
        assert_eq!(result.quarter, 0.6667);
    }

    // --- month win rate ---
    #[test]
    fn month_two_months_one_win() {
        // Jan positive, Feb negative
        let date_keys = [20240102, 20240103, 20240201, 20240202];
        let returns = [0.01, 0.01, -0.01, -0.01];
        let result = period_win_rates(&date_keys, &returns, 252);
        assert_eq!(result.month, 0.5);
    }

    // --- period exactly zero sum counts as loss ---
    #[test]
    fn period_zero_sum_is_loss() {
        // One week: +0.01 and -0.01 → sum = 0.0 → loss (not > 0)
        let date_keys = [20240101, 20240102];
        let returns = [0.01, -0.01];
        let result = period_win_rates(&date_keys, &returns, 252);
        assert_eq!(result.week, 0.0);
        assert_eq!(result.month, 0.0);
    }
}
