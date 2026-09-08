//! Shared population-volatility normalization for stats, verdicts and Python curves.
use crate::core::utils::std_inline;

pub(crate) const VOL_EPSILON: f64 = 1e-12;

/// None means normalization is undefined; callers choose their presentation fallback.
pub(crate) fn normalize_returns(
    returns: &[f64],
    yearly_days: usize,
    target_vol: f64,
) -> Option<Vec<f64>> {
    if returns.is_empty()
        || yearly_days == 0
        || !target_vol.is_finite()
        || target_vol <= 0.0
        || returns.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let vol = std_inline(returns) * (yearly_days as f64).sqrt();
    if !vol.is_finite() || vol < VOL_EPSILON {
        return None;
    }
    let scale = target_vol / vol;
    let normalized: Vec<_> = returns.iter().map(|v| v * scale).collect();
    normalized
        .iter()
        .all(|v| v.is_finite())
        .then_some(normalized)
}

/// Normalize long and benchmark separately over the full sample, then subtract.
pub(crate) fn compute_vol_adjusted_alpha(
    long: &[f64],
    bench: &[f64],
    yearly_days: usize,
    target_vol: f64,
) -> Option<Vec<f64>> {
    if long.len() != bench.len() {
        return None;
    }
    let long = normalize_returns(long, yearly_days, target_vol)?;
    let bench = normalize_returns(bench, yearly_days, target_vol)?;
    let alpha: Vec<_> = long.iter().zip(bench.iter()).map(|(l, b)| l - b).collect();
    alpha.iter().all(|v| v.is_finite()).then_some(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_uses_population_vol_and_parameters() {
        // std([0, 2]) = 1; annual vol at four days is 2.
        assert_eq!(normalize_returns(&[0.0, 2.0], 4, 0.4), Some(vec![0.0, 0.4]));
        assert_eq!(
            compute_vol_adjusted_alpha(&[0.0, 2.0], &[2.0, 0.0], 4, 0.4),
            Some(vec![-0.4, 0.4])
        );
        assert_eq!(
            compute_vol_adjusted_alpha(&[0.0, 2.0], &[2.0, 0.0], 16, 0.8),
            Some(vec![-0.4, 0.4])
        );
    }

    #[test]
    fn degeneracy_is_symmetric_and_explicit() {
        let valid = [0.01, -0.02];
        for bad in [
            vec![],
            vec![1.0],
            vec![0.0, 0.0],
            vec![1e-15, -1e-15],
            vec![f64::NAN, 0.0],
            vec![f64::INFINITY, 0.0],
            vec![f64::MAX, -f64::MAX],
        ] {
            assert!(compute_vol_adjusted_alpha(&bad, &valid, 252, 0.2).is_none());
            assert!(compute_vol_adjusted_alpha(&valid, &bad, 252, 0.2).is_none());
        }
        for target in [0.0, -0.2, f64::NAN, f64::INFINITY] {
            assert!(normalize_returns(&valid, 252, target).is_none());
        }
        assert!(normalize_returns(&valid, 0, 0.2).is_none());
        assert!(normalize_returns(&[1.0, 2.0], 1, f64::MAX).is_none());
    }
}
