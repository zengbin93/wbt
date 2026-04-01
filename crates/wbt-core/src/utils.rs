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
