use crate::core::errors::WbtError;
use crate::core::utils::{RoundToNthDigit, min_max};
use serde::Serialize;

// ---------------------------------------------------------------------------
// Underwater / drawdown helpers (inlined from top_drawdowns.rs)
// ---------------------------------------------------------------------------

pub(crate) fn calc_underwater(returns: &[f64]) -> Vec<f64> {
    let mut sum = 0.0;
    let mut sum_max_so_far = f64::NEG_INFINITY;
    returns
        .iter()
        .map(|&r| {
            sum += r;
            sum_max_so_far = sum_max_so_far.max(sum);
            sum - sum_max_so_far
        })
        .collect()
}

pub(crate) fn calc_underwater_valley(underwater: &[f64]) -> Option<usize> {
    underwater
        .iter()
        .enumerate()
        .filter(|&(_, &val)| !val.is_nan() && val != 0.0)
        .min_by(|&(_, val1), &(_, val2)| val1.partial_cmp(val2).unwrap())
        .map(|(i, _)| i)
}

pub(crate) fn calc_underwater_peak(underwater: &[f64], valley: usize) -> usize {
    underwater
        .iter()
        .enumerate()
        .rev()
        .skip(underwater.len() - 1 - valley)
        .find(|&(_, &x)| x == 0.0)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub(crate) fn calc_underwater_recovery(underwater: &[f64], valley: usize) -> Option<usize> {
    underwater
        .iter()
        .enumerate()
        .skip(valley)
        .find(|&(_, &x)| x == 0.0)
        .map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// DailyPerformance
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Serialize)]
pub struct DailyPerformance {
    pub absolute_return: f64,
    pub annual_returns: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub daily_win_rate: f64,
    pub daily_profit_loss_ratio: f64,
    pub daily_win_probability: f64,
    pub annual_volatility: f64,
    pub downside_volatility: f64,
    pub non_zero_coverage: f64,
    pub break_even_point: f64,
    pub new_high_interval: f64,
    pub new_high_ratio: f64,
    pub drawdown_risk: f64,
    pub annual_lin_reg_cumsum_return: Option<f64>,
    pub length_adjusted_average_max_drawdown: f64,
}

impl Default for DailyPerformance {
    fn default() -> DailyPerformance {
        DailyPerformance {
            absolute_return: 0.0,
            annual_returns: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            daily_win_rate: 0.0,
            daily_profit_loss_ratio: 0.0,
            daily_win_probability: 0.0,
            annual_volatility: 0.0,
            downside_volatility: 0.0,
            non_zero_coverage: 0.0,
            break_even_point: 0.0,
            new_high_interval: 0.0,
            new_high_ratio: 0.0,
            drawdown_risk: 0.0,
            annual_lin_reg_cumsum_return: None,
            length_adjusted_average_max_drawdown: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn daily_performance(
    daily_returns: &[f64],
    yearly_days: Option<usize>,
) -> Result<DailyPerformance, WbtError> {
    if daily_returns.is_empty() {
        return Ok(DailyPerformance::default());
    }

    let total_days = daily_returns.len() as f64;
    let yearly_days = yearly_days.unwrap_or(252) as f64;

    let mut cum_return = 0.0;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut max_cum_return: f64 = 0.0;
    let mut zero_drawdown_count = 0;
    let mut current_interval = 0;
    let mut new_high_interval: i32 = 0;
    let mut win_count = 0;
    let mut cum_win = 0.0;
    let mut cum_loss = 0.0;
    let mut neg_count = 0.0;
    let mut neg_mean = 0.0;
    let mut neg_m2 = 0.0;
    let mut zero_count = 0;
    let mut lr_sum_xy = 0.0;
    let mut lr_sum_cum_return = 0.0;

    for (i, &daily_return) in daily_returns.iter().enumerate() {
        let delta = daily_return - mean;
        mean += delta / (i as f64 + 1.0);
        let delta2 = daily_return - mean;
        m2 += delta * delta2;

        cum_return += daily_return;
        lr_sum_cum_return += cum_return;
        lr_sum_xy += (i as f64) * cum_return;

        if cum_return > max_cum_return {
            max_cum_return = cum_return;
            new_high_interval = new_high_interval.max(current_interval);
            current_interval = 0;
        }
        current_interval += 1;

        if max_cum_return - cum_return <= 0.0 {
            zero_drawdown_count += 1;
        }

        match daily_return {
            d if d > 0.0 => {
                win_count += 1;
                cum_win += d;
            }
            d if d < 0.0 => {
                cum_loss += daily_return;
                neg_count += 1.0;
                let neg_delta = d - neg_mean;
                neg_mean += neg_delta / neg_count;
                let neg_delta2 = d - neg_mean;
                neg_m2 += neg_delta * neg_delta2;
            }
            _ => {
                win_count += 1;
                zero_count += 1;
            }
        };
    }

    if cum_return.abs() < f64::EPSILON {
        return Ok(DailyPerformance::default());
    }

    let lr_sum_x = (total_days - 1.0) * total_days / 2.0;
    let lr_sum_x_squared = (total_days - 1.0) * total_days * (2.0 * total_days - 1.0) / 6.0;
    let lr_denominator = total_days * lr_sum_x_squared - lr_sum_x * lr_sum_x;
    let annual_lr_cumsum_slope = if lr_denominator.abs() > f64::EPSILON {
        let slope =
            (1.0 / lr_denominator) * (total_days * lr_sum_xy - lr_sum_x * lr_sum_cum_return);
        Some((slope * yearly_days).round_to_4_digit())
    } else {
        None
    };

    let (max_drawdown, length_adjusted_average_max_drawdown) =
        daily_performance_drawdown(5, daily_returns, yearly_days);

    let variance = m2 / total_days;
    let std_val = variance.sqrt();
    if std_val < f64::EPSILON {
        return Ok(DailyPerformance::default());
    }

    let sharpe_ratio = mean / std_val * yearly_days.sqrt();
    let new_high_ratio = (zero_drawdown_count as f64 / total_days).round_to_4_digit();
    let annual_returns = (mean * yearly_days).round_to_4_digit();
    let calmar_ratio = if max_drawdown < f64::EPSILON {
        10.0
    } else {
        annual_returns / max_drawdown
    }
    .round_to_4_digit();
    let daily_win_rate = (win_count as f64 / total_days).round_to_4_digit();
    let loss_count = total_days as usize - win_count;
    let daily_mean_loss = if loss_count > 0 {
        cum_loss / loss_count as f64
    } else {
        0.0
    };
    let daily_mean_win = if win_count > 0 {
        cum_win / win_count as f64
    } else {
        0.0
    };
    let daily_profit_loss_ratio = if daily_mean_loss.abs() > f64::EPSILON {
        daily_mean_win / daily_mean_loss.abs()
    } else {
        5.0
    }
    .round_to_4_digit();
    let daily_win_probability =
        (daily_profit_loss_ratio * daily_win_rate - (1.0 - daily_win_rate)).round_to_4_digit();

    let annual_volatility = (std_val * yearly_days.sqrt()).round_to_4_digit();

    let drawdown_risk = (max_drawdown / annual_volatility).round_to_4_digit();

    let downside_volatility = if neg_count > 0.0 {
        let neg_variance = neg_m2 / neg_count;
        let neg_std_dev = neg_variance.sqrt();
        neg_std_dev * yearly_days.sqrt()
    } else {
        0.0
    }
    .round_to_4_digit();

    let absolute_return = cum_return.round_to_4_digit();

    let non_zero_coverage = ((total_days - zero_count as f64) / total_days).round_to_4_digit();

    let sharpe_ratio = (min_max(sharpe_ratio, -5.0, 10.0)).round_to_4_digit();
    let calmar_ratio = (min_max(calmar_ratio, -10.0, 20.0)).round_to_4_digit();

    let mut sorted_daily_returns = daily_returns.to_vec();
    sorted_daily_returns.sort_by(|a, b| a.total_cmp(b));

    let break_even_index = sorted_daily_returns
        .iter()
        .scan(0.0, |sum, &daily_return| {
            *sum += daily_return;
            Some(*sum)
        })
        .position(|cum_sum| cum_sum > 0.0);

    let break_even_point = match break_even_index {
        Some(idx) => ((idx + 1) as f64 / total_days).round_to_4_digit(),
        None => 1.0,
    };

    Ok(DailyPerformance {
        absolute_return,
        annual_returns,
        sharpe_ratio,
        max_drawdown: max_drawdown.round_to_4_digit(),
        calmar_ratio,
        daily_win_rate,
        daily_profit_loss_ratio,
        daily_win_probability,
        annual_volatility,
        downside_volatility,
        non_zero_coverage,
        break_even_point,
        new_high_interval: new_high_interval as f64,
        new_high_ratio,
        drawdown_risk,
        annual_lin_reg_cumsum_return: annual_lr_cumsum_slope,
        length_adjusted_average_max_drawdown: length_adjusted_average_max_drawdown
            .round_to_4_digit(),
    })
}

// ---------------------------------------------------------------------------
// Drawdown extraction (top-N)
// ---------------------------------------------------------------------------

pub(crate) fn daily_performance_drawdown(
    top_n_drawdown: usize,
    daily_returns: &[f64],
    yearly_days: f64,
) -> (f64, f64) {
    let total_days = daily_returns.len();
    let mut underwater = calc_underwater(daily_returns);
    let mut top_n_drawdown_days_sum = 0;
    let mut max_drawdown = 0.0;
    for _ in 0..top_n_drawdown {
        let valley = calc_underwater_valley(&underwater);
        if valley.is_none() {
            break;
        }
        let valley = valley.unwrap();
        let peak = calc_underwater_peak(&underwater, valley);
        let recovery = calc_underwater_recovery(&underwater, valley);
        let drawdown = -underwater[valley];
        max_drawdown = if max_drawdown > drawdown {
            max_drawdown
        } else {
            drawdown
        };
        top_n_drawdown_days_sum += valley.abs_diff(peak);
        if let Some(recovery) = recovery {
            underwater[peak..recovery].fill(0.0);
        } else {
            underwater[peak..total_days].fill(0.0);
        }
    }
    let length_adjusted_average_max_drawdown =
        top_n_drawdown_days_sum as f64 / top_n_drawdown as f64 / yearly_days;
    (max_drawdown, length_adjusted_average_max_drawdown)
}
