use chrono::NaiveDate;
use strum_macros::{AsRefStr, Display, EnumString};

#[derive(Debug, Clone, Copy, PartialEq, EnumString, AsRefStr, Display)]
pub enum WeightType {
    #[strum(serialize = "ts")]
    TS,
    #[strum(serialize = "cs")]
    CS,
}

pub trait RoundToNthDigit {
    fn round_to_nth_digit(&self, nth: usize) -> Self;
    fn round_to_2_digit(&self) -> Self;
    fn round_to_3_digit(&self) -> Self;
    fn round_to_4_digit(&self) -> Self;
}

impl RoundToNthDigit for f64 {
    fn round_to_nth_digit(&self, nth: usize) -> f64 {
        let scale = match nth {
            2 => 100.0,
            3 => 1000.0,
            4 => 10000.0,
            _ => 10_f64.powi(nth as i32),
        };
        (self * scale).round() / scale
    }
    fn round_to_2_digit(&self) -> f64 {
        self.round_to_nth_digit(2)
    }
    fn round_to_3_digit(&self) -> f64 {
        self.round_to_nth_digit(3)
    }
    fn round_to_4_digit(&self) -> f64 {
        self.round_to_nth_digit(4)
    }
}

/// 将 YYYYMMDD 整数 date_key 转换为 NaiveDate
pub(crate) fn date_key_to_naive_date(dk: i32) -> NaiveDate {
    let y = dk / 10000;
    let m = (dk / 100) % 100;
    let d = dk % 100;
    NaiveDate::from_ymd_opt(y, m as u32, d as u32).unwrap_or_else(|| {
        debug_assert!(false, "date_key_to_naive_date: invalid dk={dk}");
        NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
    })
}

/// 纯数值 pearson 相关系数 (ddof=1，与 Polars pearson_corr ddof=1 一致)
#[inline]
#[allow(dead_code)]
pub(crate) fn pearson_corr_inline(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = xs.iter().sum::<f64>() / nf;
    let mean_y = ys.iter().sum::<f64>() / nf;
    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 { 0.0 } else { cov / denom }
}

/// 纯数值标准差 (ddof=0，与 Polars .std(0) 一致)
#[inline]
pub(crate) fn std_inline(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nf;
    var.sqrt()
}

pub fn min_max(x: f64, min_val: f64, max_val: f64) -> f64 {
    if x < min_val {
        min_val
    } else if x > max_val {
        max_val
    } else {
        x
    }
}

pub trait Quantile {
    fn quantile(&self, q: f64) -> Option<f64>;
}

impl Quantile for [f64] {
    fn quantile(&self, q: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&q) {
            return None;
        }
        let n = self.len();
        if n == 0 {
            return None;
        }
        let mut sorted = self.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = q * (n as f64 - 1.0);
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let fraction = pos - lower as f64;
        if lower == upper {
            Some(sorted[lower])
        } else {
            Some(sorted[lower] + fraction * (sorted[upper] - sorted[lower]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- WeightType ---
    #[test]
    fn weight_type_from_str() {
        assert_eq!("ts".parse::<WeightType>().unwrap(), WeightType::TS);
        assert_eq!("cs".parse::<WeightType>().unwrap(), WeightType::CS);
        assert!("invalid".parse::<WeightType>().is_err());
    }

    #[test]
    fn weight_type_display() {
        assert_eq!(WeightType::TS.to_string(), "ts");
        assert_eq!(WeightType::CS.to_string(), "cs");
    }

    // --- RoundToNthDigit ---
    #[test]
    fn round_to_nth_digit_basic() {
        assert_eq!(1.2345f64.round_to_nth_digit(2), 1.23);
        assert_eq!(1.2355f64.round_to_nth_digit(2), 1.24);
        assert_eq!(1.2345f64.round_to_nth_digit(3), 1.235);
        assert_eq!(1.2345f64.round_to_nth_digit(4), 1.2345);
        assert_eq!(1.2345f64.round_to_nth_digit(0), 1.0);
        assert_eq!(1.5f64.round_to_nth_digit(0), 2.0);
    }

    #[test]
    fn round_to_nth_digit_negative() {
        assert_eq!((-1.2345f64).round_to_nth_digit(2), -1.23);
        assert_eq!((-1.2355f64).round_to_nth_digit(2), -1.24);
    }

    #[test]
    fn round_to_nth_digit_zero() {
        assert_eq!(0.0f64.round_to_nth_digit(2), 0.0);
    }

    #[test]
    fn round_to_2_3_4_digit() {
        assert_eq!(1.2345f64.round_to_2_digit(), 1.23);
        assert_eq!(1.2345f64.round_to_3_digit(), 1.235);
        assert_eq!(1.23456f64.round_to_4_digit(), 1.2346);
    }

    // --- Quantile ---
    #[test]
    fn quantile_empty() {
        let v: Vec<f64> = vec![];
        assert_eq!(v.quantile(0.5), None);
    }

    #[test]
    fn quantile_single() {
        assert_eq!([42.0].quantile(0.0), Some(42.0));
        assert_eq!([42.0].quantile(0.5), Some(42.0));
        assert_eq!([42.0].quantile(1.0), Some(42.0));
    }

    #[test]
    fn quantile_sorted() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(v.quantile(0.0), Some(1.0));
        assert_eq!(v.quantile(1.0), Some(5.0));
        assert_eq!(v.quantile(0.5), Some(3.0));
        assert_eq!(v.quantile(0.25), Some(2.0));
    }

    #[test]
    fn quantile_unsorted() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(v.quantile(0.5), Some(3.0));
    }

    #[test]
    fn quantile_out_of_range() {
        let v = [1.0, 2.0];
        assert_eq!(v.quantile(-0.1), None);
        assert_eq!(v.quantile(1.1), None);
    }

    // --- date_key_to_naive_date ---
    #[test]
    fn date_key_known_dates() {
        assert_eq!(
            date_key_to_naive_date(20250101),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()
        );
        assert_eq!(
            date_key_to_naive_date(19700101),
            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
        );
        assert_eq!(
            date_key_to_naive_date(20001231),
            NaiveDate::from_ymd_opt(2000, 12, 31).unwrap()
        );
    }

    // --- pearson_corr_inline ---
    #[test]
    fn pearson_perfect_positive() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_corr_inline(&xs, &ys) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_perfect_negative() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson_corr_inline(&xs, &ys) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_constant_returns_zero() {
        let xs = [3.0, 3.0, 3.0];
        let ys = [1.0, 2.0, 3.0];
        assert_eq!(pearson_corr_inline(&xs, &ys), 0.0);
    }

    #[test]
    fn pearson_single_element() {
        assert_eq!(pearson_corr_inline(&[1.0], &[2.0]), 0.0);
    }

    // --- std_inline ---
    #[test]
    fn std_all_same() {
        assert_eq!(std_inline(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn std_known_values() {
        let result = std_inline(&[1.0, 2.0, 3.0]);
        assert!((result - (2.0_f64 / 3.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn std_empty() {
        assert_eq!(std_inline(&[]), 0.0);
    }

    #[test]
    fn std_single() {
        assert_eq!(std_inline(&[42.0]), 0.0);
    }

    // --- min_max ---
    #[test]
    fn min_max_in_range() {
        assert_eq!(min_max(5.0, 0.0, 10.0), 5.0);
    }

    #[test]
    fn min_max_below() {
        assert_eq!(min_max(-1.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn min_max_above() {
        assert_eq!(min_max(15.0, 0.0, 10.0), 10.0);
    }
}
