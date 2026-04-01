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
        let scale = 10_f64.powi(nth as i32);
        (self * scale).round() / scale
    }
    fn round_to_2_digit(&self) -> f64 { self.round_to_nth_digit(2) }
    fn round_to_3_digit(&self) -> f64 { self.round_to_nth_digit(3) }
    fn round_to_4_digit(&self) -> f64 { self.round_to_nth_digit(4) }
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
    if x < min_val { min_val } else if x > max_val { max_val } else { x }
}

pub trait Quantile {
    fn quantile(&self, q: f64) -> Option<f64>;
}

impl Quantile for [f64] {
    fn quantile(&self, q: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&q) { return None; }
        let n = self.len();
        if n == 0 { return None; }
        let mut sorted = self.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = q * (n as f64 - 1.0);
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let fraction = pos - lower as f64;
        if lower == upper { Some(sorted[lower]) }
        else { Some(sorted[lower] + fraction * (sorted[upper] - sorted[lower])) }
    }
}
