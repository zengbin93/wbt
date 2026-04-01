use crate::errors::WbtError;
use crate::native_engine::PairsSoA;
use crate::trade_dir::TradeDir;
use crate::utils::RoundToNthDigit;
use polars::prelude::*;
use serde::Serialize;

#[derive(Serialize)]
pub struct EvaluatePairs {
    /// 交易方向
    pub trade_direction: TradeDir,
    /// 交易次数
    pub trade_count: usize,
    /// 累计收益
    pub total_profit: f64,
    /// 单笔收益
    pub single_trade_profit: f64,
    /// 盈利次数
    pub win_trade_count: usize,
    /// 累计盈利
    pub sum_win: f64,
    /// 单笔盈利
    pub win_one: f64,
    /// 亏损次数
    pub loss_trade_count: usize,
    /// 累计亏损
    pub sum_loss: f64,
    /// 单笔亏损
    pub loss_one: f64,
    /// 交易胜率
    pub win_rate: f64,
    /// 累计盈亏比
    pub total_profit_loss_ratio: f64,
    /// 单笔盈亏比
    pub single_profit_loss_ratio: f64,
    /// 盈亏平衡点
    pub break_even_point: f64,
    /// 持仓K线数
    pub position_k_days: f64,
}

impl Default for EvaluatePairs {
    fn default() -> EvaluatePairs {
        EvaluatePairs {
            trade_direction: TradeDir::LongShort,
            trade_count: 0,
            total_profit: 0.0,
            single_trade_profit: 0.0,
            win_trade_count: 0,
            sum_win: 0.0,
            win_one: 0.0,
            loss_trade_count: 0,
            sum_loss: 0.0,
            loss_one: 0.0,
            win_rate: 0.0,
            total_profit_loss_ratio: 0.0,
            single_profit_loss_ratio: 0.0,
            break_even_point: 0.0,
            position_k_days: 0.0,
        }
    }
}

/// 评估交易对性能 — 纯数值循环版本（零 Polars 操作）
#[allow(dead_code)]
pub fn evaluate_pairs(
    pairs: &DataFrame,
    trade_dir: TradeDir,
) -> Result<EvaluatePairs, WbtError> {
    if pairs.is_empty() {
        return Ok(EvaluatePairs::default());
    }

    // 获取底层数据切片
    let profit_col = pairs.column("盈亏比例")?.as_materialized_series().f64()?;
    let count_col = match pairs.column("持仓数量") {
        Ok(s) => s.as_materialized_series().clone(),
        Err(_) => Series::new("持仓数量".into(), vec![1.0f64; pairs.height()]),
    };
    let count_f64 = count_col.cast(&DataType::Float64)?;
    let count_ca = count_f64.f64()?;

    let hold_bars_col = pairs.column("持仓K线数")?.as_materialized_series().clone();

    // 如果需要按方向过滤，获取方向列
    let dir_col = pairs.column("交易方向")?.as_materialized_series().str()?;
    let dir_filter = match trade_dir {
        TradeDir::Long => Some("多头"),
        TradeDir::Short => Some("空头"),
        TradeDir::LongShort => None,
    };

    // 单次遍历计算所有统计量
    let mut trade_count = 0.0f64;
    let mut win_trade_count = 0.0f64;
    let mut sum_win = 0.0f64;
    let mut loss_trade_count = 0.0f64;
    let mut sum_loss = 0.0f64;
    let mut sum_hold_bars = 0.0f64;

    // 收集 (profit, count) 对用于 break_even_point 计算
    let n = pairs.height();
    let mut profit_count_pairs: Vec<(f64, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        // 方向过滤
        if let Some(filter_str) = dir_filter {
            match dir_col.get(i) {
                Some(d) if d == filter_str => {}
                _ => continue,
            }
        }

        let p = profit_col.get(i).unwrap_or(0.0);
        let c = count_ca.get(i).unwrap_or(1.0);
        if c <= 0.0 {
            continue;
        }

        trade_count += c;

        if p >= 0.0 {
            win_trade_count += c;
            sum_win += p * c;
        } else {
            loss_trade_count += c;
            sum_loss += p * c;
        }

        // 持仓K线数
        if let Ok(hb_f) = hold_bars_col.f64() {
            sum_hold_bars += hb_f.get(i).unwrap_or(0.0) * c;
        } else if let Ok(hb_i) = hold_bars_col.i64() {
            sum_hold_bars += (hb_i.get(i).unwrap_or(0) as f64) * c;
        }

        profit_count_pairs.push((p, c));
    }

    if trade_count <= 0.0 {
        return Ok(EvaluatePairs::default());
    }

    let position_k_days = sum_hold_bars / trade_count;
    let win_one = if win_trade_count > 0.0 {
        sum_win / win_trade_count
    } else {
        0.0
    };
    let loss_one = if loss_trade_count > 0.0 {
        sum_loss / loss_trade_count
    } else {
        0.0
    };
    let win_rate = win_trade_count / trade_count;

    // Break-even point 计算 (排序后遍历)
    profit_count_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut sum = 0.0;
    let mut seen = 0.0;
    let mut break_even_point = 1.0;
    let mut found = false;

    for (p, c) in &profit_count_pairs {
        if *c <= 0.0 {
            continue;
        }

        if !found {
            if *p <= 0.0 {
                sum += p * c;
                seen += c;
                if sum >= 0.0 {
                    break_even_point = seen / trade_count;
                    found = true;
                }
            } else {
                let need = -sum / p;
                let mut k = need.ceil();
                if k < 1.0 {
                    k = 1.0;
                }
                if k > *c {
                    k = *c;
                }

                sum += p * k;
                seen += k;
                if sum >= 0.0 {
                    break_even_point = seen / trade_count;
                    found = true;
                }

                if k < *c {
                    sum += p * (*c - k);
                    seen += *c - k;
                }
            }
        } else {
            sum += p * c;
            seen += c;
        }
    }

    if sum <= 0.0 {
        break_even_point = 1.0;
    }

    let total_profit_loss_ratio = if sum_loss == 0.0 {
        0.0
    } else {
        sum_win / sum_loss.abs()
    };
    let single_profit_loss_ratio = if loss_one == 0.0 {
        0.0
    } else {
        win_one / loss_one.abs()
    };

    Ok(EvaluatePairs {
        trade_direction: trade_dir,
        trade_count: trade_count as usize,
        total_profit: sum.round_to_2_digit(),
        single_trade_profit: (sum / trade_count).round_to_2_digit(),
        win_trade_count: win_trade_count as usize,
        sum_win: sum_win.round_to_2_digit(),
        win_one: win_one.round_to_4_digit(),
        loss_trade_count: loss_trade_count as usize,
        sum_loss: sum_loss.round_to_2_digit(),
        loss_one: loss_one.round_to_4_digit(),
        win_rate: win_rate.round_to_4_digit(),
        total_profit_loss_ratio: total_profit_loss_ratio.round_to_4_digit(),
        single_profit_loss_ratio: single_profit_loss_ratio.round_to_4_digit(),
        break_even_point: break_even_point.round_to_4_digit(),
        position_k_days: position_k_days.round_to_2_digit(),
    })
}

/// 评估交易对性能 — 从 PairsSoA 直接读取（零 DataFrame 构建）
pub fn evaluate_pairs_soa(
    pairs: &PairsSoA,
    trade_dir: TradeDir,
) -> Result<EvaluatePairs, WbtError> {
    let n = pairs.profit_bps.len();
    if n == 0 {
        return Ok(EvaluatePairs::default());
    }

    let dir_filter = match trade_dir {
        TradeDir::Long => Some("多头"),
        TradeDir::Short => Some("空头"),
        TradeDir::LongShort => None,
    };

    let mut trade_count = 0.0f64;
    let mut win_trade_count = 0.0f64;
    let mut sum_win = 0.0f64;
    let mut loss_trade_count = 0.0f64;
    let mut sum_loss = 0.0f64;
    let mut sum_hold_bars = 0.0f64;
    let mut profit_count_pairs: Vec<(f64, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        // 方向过滤
        if let Some(filter_str) = dir_filter {
            if pairs.dirs[i] != filter_str {
                continue;
            }
        }

        let p = pairs.profit_bps[i];
        let c = pairs.counts[i] as f64;
        if c <= 0.0 {
            continue;
        }

        trade_count += c;

        if p >= 0.0 {
            win_trade_count += c;
            sum_win += p * c;
        } else {
            loss_trade_count += c;
            sum_loss += p * c;
        }

        sum_hold_bars += (pairs.hold_bars[i] as f64) * c;
        profit_count_pairs.push((p, c));
    }

    if trade_count <= 0.0 {
        return Ok(EvaluatePairs::default());
    }

    let position_k_days = sum_hold_bars / trade_count;
    let win_one = if win_trade_count > 0.0 {
        sum_win / win_trade_count
    } else {
        0.0
    };
    let loss_one = if loss_trade_count > 0.0 {
        sum_loss / loss_trade_count
    } else {
        0.0
    };
    let win_rate = win_trade_count / trade_count;

    // Break-even point (排序后遍历)
    profit_count_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut sum = 0.0;
    let mut seen = 0.0;
    let mut break_even_point = 1.0;
    let mut found = false;

    for (p, c) in &profit_count_pairs {
        if *c <= 0.0 {
            continue;
        }
        if !found {
            if *p <= 0.0 {
                sum += p * c;
                seen += c;
                if sum >= 0.0 {
                    break_even_point = seen / trade_count;
                    found = true;
                }
            } else {
                let need = -sum / p;
                let mut k = need.ceil();
                if k < 1.0 {
                    k = 1.0;
                }
                if k > *c {
                    k = *c;
                }
                sum += p * k;
                seen += k;
                if sum >= 0.0 {
                    break_even_point = seen / trade_count;
                    found = true;
                }
                if k < *c {
                    sum += p * (*c - k);
                    seen += *c - k;
                }
            }
        } else {
            sum += p * c;
            seen += c;
        }
    }

    if sum <= 0.0 {
        break_even_point = 1.0;
    }

    let total_profit_loss_ratio = if sum_loss == 0.0 {
        0.0
    } else {
        sum_win / sum_loss.abs()
    };
    let single_profit_loss_ratio = if loss_one == 0.0 {
        0.0
    } else {
        win_one / loss_one.abs()
    };

    Ok(EvaluatePairs {
        trade_direction: trade_dir,
        trade_count: trade_count as usize,
        total_profit: sum.round_to_2_digit(),
        single_trade_profit: (sum / trade_count).round_to_2_digit(),
        win_trade_count: win_trade_count as usize,
        sum_win: sum_win.round_to_2_digit(),
        win_one: win_one.round_to_4_digit(),
        loss_trade_count: loss_trade_count as usize,
        sum_loss: sum_loss.round_to_2_digit(),
        loss_one: loss_one.round_to_4_digit(),
        win_rate: win_rate.round_to_4_digit(),
        total_profit_loss_ratio: total_profit_loss_ratio.round_to_4_digit(),
        single_profit_loss_ratio: single_profit_loss_ratio.round_to_4_digit(),
        break_even_point: break_even_point.round_to_4_digit(),
        position_k_days: position_k_days.round_to_2_digit(),
    })
}
