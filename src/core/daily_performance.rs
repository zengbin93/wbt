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

#[derive(Debug, Clone, PartialEq, Serialize)]
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
    let mut max_cum_return: f64 = f64::NEG_INFINITY;
    let mut zero_drawdown_count = 0;
    // 新高间隔 = 最长「严格水下」连续 bar 数（cum_return < running_max）。
    // 语义与 czsc 漏洞对照文档「方法二」保持一致：只数水下天数本身，
    // 不把新高当日或起始 bar 计入。
    let mut current_uw_streak: i32 = 0;
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

        if cum_return >= max_cum_return {
            max_cum_return = cum_return;
            current_uw_streak = 0;
            zero_drawdown_count += 1;
        } else {
            current_uw_streak += 1;
            new_high_interval = new_high_interval.max(current_uw_streak);
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- calc_underwater ---
    #[test]
    fn underwater_basic() {
        let returns = [0.1, -0.05, 0.02, -0.15, 0.03];
        let uw = calc_underwater(&returns);
        assert_eq!(uw.len(), 5);
        assert!((uw[0] - 0.0).abs() < 1e-10);
        assert!((uw[1] - (-0.05)).abs() < 1e-10);
        assert!((uw[2] - (-0.03)).abs() < 1e-10);
        assert!((uw[3] - (-0.18)).abs() < 1e-10);
        assert!((uw[4] - (-0.15)).abs() < 1e-10);
    }

    #[test]
    fn underwater_all_positive() {
        let returns = [0.01, 0.02, 0.03];
        let uw = calc_underwater(&returns);
        assert!(uw.iter().all(|&x| x == 0.0));
    }

    // --- calc_underwater_valley ---
    #[test]
    fn valley_finds_min() {
        let uw = [0.0, -0.05, -0.03, -0.18, -0.15];
        assert_eq!(calc_underwater_valley(&uw), Some(3));
    }

    #[test]
    fn valley_all_zero() {
        let uw = [0.0, 0.0, 0.0];
        assert_eq!(calc_underwater_valley(&uw), None);
    }

    // --- calc_underwater_peak ---
    #[test]
    fn peak_before_valley() {
        let uw = [0.0, -0.05, -0.03, -0.18, -0.15];
        assert_eq!(calc_underwater_peak(&uw, 3), 0);
    }

    // --- calc_underwater_recovery ---
    #[test]
    fn recovery_finds_zero_after_valley() {
        let uw = [0.0, -0.05, 0.0, -0.01, 0.0];
        assert_eq!(calc_underwater_recovery(&uw, 1), Some(2));
    }

    #[test]
    fn recovery_none_when_no_recovery() {
        let uw = [0.0, -0.05, -0.03, -0.01, -0.02];
        assert_eq!(calc_underwater_recovery(&uw, 1), None);
    }

    // --- daily_performance ---
    #[test]
    fn daily_performance_empty_returns_default() {
        let dp = daily_performance(&[], None).unwrap();
        assert_eq!(dp, DailyPerformance::default());
    }

    #[test]
    fn daily_performance_all_zero_returns_default() {
        let dp = daily_performance(&[0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(dp, DailyPerformance::default());
    }

    #[test]
    fn daily_performance_known_values() {
        // returns = [0.01, -0.005, 0.02], yearly_days=252
        //
        // cum = [0.01, 0.005, 0.025]
        // absolute_return = 0.025
        // mean = 0.025/3 = 0.008333, std(ddof=0) = 0.010274
        // sharpe_raw = 0.008333/0.010274 * sqrt(252) = 12.876 -> capped at 10.0
        // annual_returns = 0.008333 * 252 = 2.1
        //
        // underwater: [0, -0.005, 0], max_drawdown = 0.005
        // calmar_raw = 2.1/0.005 = 420 -> capped at 20.0
        //
        // win_count = 2 (0.01>0, 0.02>0), loss_count = 1 (-0.005<0)
        // daily_win_rate = 2/3 = 0.6667
        // cum_win = 0.03, cum_loss = -0.005
        // mean_win = 0.03/2 = 0.015, mean_loss = -0.005/1 = -0.005
        // daily_profit_loss_ratio = 0.015/0.005 = 3.0
        //
        // new_high_interval = 最长水下连续天数:
        //   cum = [0.01, 0.005, 0.025], running_max = [0.01, 0.01, 0.025]
        //   underwater = [F, T, F] → 最长水下段 = 1
        // zero_drawdown_count = 2 (day0 and day2), ratio = 2/3
        //
        // sorted: [-0.005, 0.01, 0.02], cumsum: [-0.005, 0.005, 0.025]
        // first > 0 at index 1 => break_even_point = 2/3
        let returns = [0.01, -0.005, 0.02];
        let dp = daily_performance(&returns, Some(252)).unwrap();

        assert_eq!(dp.absolute_return, 0.025);
        assert_eq!(dp.annual_returns, 2.1);
        assert_eq!(dp.sharpe_ratio, 10.0);
        assert_eq!(dp.max_drawdown, 0.005);
        assert_eq!(dp.calmar_ratio, 20.0);
        assert_eq!(dp.daily_win_rate, 0.6667);
        assert_eq!(dp.daily_profit_loss_ratio, 3.0);
        assert_eq!(dp.annual_volatility, 0.1631);
        assert_eq!(dp.non_zero_coverage, 1.0);
        assert_eq!(dp.new_high_interval, 1.0);
        assert_eq!(dp.new_high_ratio, 0.6667);
        assert_eq!(dp.break_even_point, 0.6667);
        assert_eq!(dp.drawdown_risk, 0.0307);
    }

    #[test]
    fn daily_performance_constant_returns_default() {
        // Constant returns have zero std => returns default
        // This is a design decision: when std=0, all metrics are zeroed out
        let returns: Vec<f64> = (0..100).map(|_| 0.001).collect();
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp, DailyPerformance::default());
    }

    #[test]
    fn daily_performance_negative_returns_known() {
        // returns = [-0.01, -0.02, 0.005]
        // cum = [-0.01, -0.03, -0.025]
        // calc_underwater starts with sum_max = -inf
        //   day0: sum=-0.01, max=-0.01, uw=0
        //   day1: sum=-0.03, max=-0.01, uw=-0.02
        //   day2: sum=-0.025, max=-0.01, uw=-0.015
        // max_drawdown = 0.02 (from peak -0.01 to valley -0.03)
        //
        // win=1 (0.005>0), loss=2 (-0.01,-0.02 <0)
        // daily_win_rate = 1/3 = 0.3333
        let returns = [-0.01, -0.02, 0.005];
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp.absolute_return, -0.025);
        assert!(dp.annual_returns < 0.0);
        assert!(dp.sharpe_ratio < 0.0);
        assert_eq!(dp.max_drawdown, 0.02);
        assert_eq!(dp.daily_win_rate, 0.3333);
    }

    /// All negative returns: sorted cumsum never goes positive → break_even_point = 1.0
    #[test]
    fn break_even_all_negative() {
        let returns = [-0.01, -0.02, -0.03];
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp.break_even_point, 1.0); // Can never break even
    }

    /// Known linear-regression slope on cumsum [0.01, 0.005, 0.025] with n=3, yearly=252.
    ///
    /// x = [0, 1, 2], y = cumsum
    /// lr_sum_x = 3, lr_sum_x2 = 5
    /// lr_sum_xy = 0*0.01 + 1*0.005 + 2*0.025 = 0.055
    /// lr_sum_y  = 0.01 + 0.005 + 0.025 = 0.04
    /// denom = 3*5 - 3*3 = 6
    /// slope = (3*0.055 - 3*0.04) / 6 = 0.045/6 = 0.0075
    /// annualised = 0.0075 * 252 = 1.89
    #[test]
    fn annual_lin_reg_known() {
        let returns = [0.01, -0.005, 0.02];
        let dp = daily_performance(&returns, Some(252)).unwrap();
        let slope = dp
            .annual_lin_reg_cumsum_return
            .expect("annual_lin_reg_cumsum_return should be Some for these returns");
        assert!(
            (slope - 1.89).abs() < 1e-3,
            "expected slope ≈ 1.89, got {slope}"
        );
    }

    #[test]
    fn daily_performance_yearly_days_proportional() {
        let returns: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.002 } else { -0.001 })
            .collect();
        let dp252 = daily_performance(&returns, Some(252)).unwrap();
        let dp365 = daily_performance(&returns, Some(365)).unwrap();
        // annual_returns = mean * yearly_days, so ratio should be 365/252
        let ratio = dp365.annual_returns as f64 / dp252.annual_returns as f64;
        assert!((ratio - 365.0 / 252.0).abs() < 0.01);
    }

    /// 尾段水下（last new high 之后再也未创新高）必须纳入 new_high_interval。
    /// 2 天新高 + 5 天水下：cum = [.01,.03,.025,.022,.020,.019,.018]，
    /// underwater = [F,F,T,T,T,T,T]，最长水下段 = 5。
    /// 修复前仅捕获两次新高之间的 interval(=1)，漏掉 5 天尾段。
    #[test]
    fn new_high_interval_includes_trailing_underwater() {
        let returns = [0.01, 0.02, -0.005, -0.003, -0.002, -0.001, -0.001];
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp.new_high_interval, 5.0);
    }

    /// 早期内部水下 + 尾段更长水下：取最长段（与 czsc bug 文件「方法二」同口径）。
    #[test]
    fn new_high_interval_trailing_longer_than_interior() {
        // 3 天新高 → 2 天内部水下 → 1 天新高 → 6 天尾段水下
        // underwater 最长段 = 6（尾段）
        let returns = [
            0.01, 0.01, 0.01, -0.002, -0.001, 0.02, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001,
        ];
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp.new_high_interval, 6.0);
    }

    /// 无回撤场景（每天创新高）：从未水下，新高间隔 = 0。
    #[test]
    fn new_high_interval_all_new_highs() {
        let returns = [0.01, 0.02, 0.03, 0.04];
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp.new_high_interval, 0.0);
    }

    /// V 型回到 peak（cum 严格等于 running_max，tie）应该终止水下段，不计入。
    /// cum = [0.05, 0.03, 0.01, 0.03, 0.05]，running_max 末端回到 0.05，最长水下 = 3。
    #[test]
    fn new_high_interval_tie_at_peak_terminates_streak() {
        let returns = [0.05, -0.02, -0.02, 0.02, 0.02];
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp.new_high_interval, 3.0);
    }

    /// 极长单段水下：1 个新高 + N 个严格水下 bar，新高间隔 = N。
    #[test]
    fn new_high_interval_long_single_underwater_segment() {
        let mut returns = vec![0.10];
        returns.extend(std::iter::repeat_n(-0.0001, 100));
        let dp = daily_performance(&returns, Some(252)).unwrap();
        assert_eq!(dp.new_high_interval, 100.0);
    }

    // --- daily_performance_drawdown ---
    #[test]
    fn drawdown_all_positive() {
        let returns: Vec<f64> = (0..50).map(|_| 0.01).collect();
        let (max_dd, _) = daily_performance_drawdown(5, &returns, 252.0);
        assert_eq!(max_dd, 0.0);
    }

    #[test]
    fn drawdown_known_sequence() {
        let returns = [0.1, -0.05, 0.1, -0.15, 0.1];
        let (max_dd, _) = daily_performance_drawdown(5, &returns, 252.0);
        assert!((max_dd - 0.15).abs() < 1e-10);
    }
}
