//! pairs 聚合去重 + 每年最赚/最亏关键交易。
//!
//! 业务口径（见飞书设计文档 §九 / §十一）：
//! - **聚合键**：`(symbol, 开仓时间, 平仓时间)`。引擎按 LIFO 撮合，一次平仓可能命中多个
//!   lot，从而产出多条共享同一 `(open_dt, close_dt)` 的部分成交记录；按此键聚合即去重。
//! - **count**：被合并的原始开平记录数量。
//! - **盈亏比例**：按成交量（`counts`）加权平均的 `profit_bp`，单位 BP（与 `pairs` 的
//!   `盈亏比例` 列一致）。
//! - **持仓K线数**：同组共享同一值（由开/平 bar 决定），取其值，**不求和**。
//! - **分年依据**：按**平仓时间**的年份。

use crate::core::errors::WbtError;
use crate::core::native_engine::{PairsSoA, dt_to_date_key_fast};
use polars::prelude::*;
use std::collections::BTreeMap;

/// 一条聚合后的开平记录。
#[derive(Debug, Clone, PartialEq)]
pub struct AggRow {
    pub sym_id: u32,
    pub open_dt: i64,
    pub close_dt: i64,
    pub dir: &'static str,
    pub open_price: f64,
    pub close_price: f64,
    pub hold_bars: i64,
    /// 成交量加权平均盈亏比例（BP）
    pub profit_bp: f64,
    /// 被合并的原始记录数量
    pub count: i64,
}

/// 按 `(sym_id, open_dt, close_dt)` 聚合 `PairsSoA`，结果按该键升序稳定排列。
pub fn aggregate_pairs(pairs: &PairsSoA) -> Vec<AggRow> {
    // 累加器：(profit*vol 之和, vol 之和, 记录数, hold_bars, dir, open_px, close_px)
    struct Acc {
        sum_pw: f64,
        sum_w: f64,
        count: i64,
        hold_bars: i64,
        dir: &'static str,
        open_price: f64,
        close_price: f64,
    }

    let mut groups: BTreeMap<(u32, i64, i64), Acc> = BTreeMap::new();

    for i in 0..pairs.sym_ids.len() {
        let key = (pairs.sym_ids[i], pairs.open_dts[i], pairs.close_dts[i]);
        let w = pairs.counts[i] as f64;
        let entry = groups.entry(key).or_insert_with(|| Acc {
            sum_pw: 0.0,
            sum_w: 0.0,
            count: 0,
            hold_bars: pairs.hold_bars[i],
            dir: pairs.dirs[i],
            open_price: pairs.open_prices[i],
            close_price: pairs.close_prices[i],
        });
        entry.sum_pw += pairs.profit_bps[i] * w;
        entry.sum_w += w;
        entry.count += 1;
        // 同组 hold_bars 理论一致；取较大值以防御异常输入。
        if pairs.hold_bars[i] > entry.hold_bars {
            entry.hold_bars = pairs.hold_bars[i];
        }
    }

    groups
        .into_iter()
        .map(|((sym_id, open_dt, close_dt), a)| {
            let profit_bp = if a.sum_w != 0.0 {
                // 与引擎 profit_bp 一致，保留两位小数
                (a.sum_pw / a.sum_w * 100.0).round() / 100.0
            } else {
                0.0
            };
            AggRow {
                sym_id,
                open_dt,
                close_dt,
                dir: a.dir,
                open_price: a.open_price,
                close_price: a.close_price,
                hold_bars: a.hold_bars,
                profit_bp,
                count: a.count,
            }
        })
        .collect()
}

/// 在聚合结果上按平仓年份分组，取每年 best/worst 各 `top` 笔。
///
/// 返回 `(year, kind, &AggRow)`，`kind` ∈ {"best", "worst"}；按 `(year, kind, 排名)` 顺序排列。
fn select_key_trades(
    agg: &[AggRow],
    top: usize,
    time_unit: TimeUnit,
) -> Vec<(i32, &'static str, usize)> {
    // year -> agg 行下标列表
    let mut by_year: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, row) in agg.iter().enumerate() {
        let year = dt_to_date_key_fast(row.close_dt, time_unit) / 10000;
        by_year.entry(year).or_default().push(idx);
    }

    let mut out: Vec<(i32, &'static str, usize)> = Vec::new();
    for (year, mut idxs) in by_year {
        // best：profit 降序；tiebreak open_dt 升序保证稳定
        idxs.sort_by(|&a, &b| {
            agg[b]
                .profit_bp
                .total_cmp(&agg[a].profit_bp)
                .then(agg[a].open_dt.cmp(&agg[b].open_dt))
        });
        let mut best: Vec<usize> = Vec::new();
        for &i in idxs.iter().take(top) {
            best.push(i);
            out.push((year, "best", i));
        }
        // worst：profit 升序；剔除已入 best 的交易，避免同一笔同时是最赚和最亏
        idxs.sort_by(|&a, &b| {
            agg[a]
                .profit_bp
                .total_cmp(&agg[b].profit_bp)
                .then(agg[a].open_dt.cmp(&agg[b].open_dt))
        });
        for &i in idxs.iter().filter(|i| !best.contains(i)).take(top) {
            out.push((year, "worst", i));
        }
    }
    out
}

fn agg_rows_to_columns(
    pairs: &PairsSoA,
    rows: impl Iterator<Item = (Option<(i32, &'static str)>, AggRow)>,
) -> Result<DataFrame, WbtError> {
    let mut years: Vec<i32> = Vec::new();
    let mut kinds: Vec<&'static str> = Vec::new();
    let mut symbols: Vec<String> = Vec::new();
    let mut open_dts: Vec<i64> = Vec::new();
    let mut close_dts: Vec<i64> = Vec::new();
    let mut dirs: Vec<&'static str> = Vec::new();
    let mut open_px: Vec<f64> = Vec::new();
    let mut close_px: Vec<f64> = Vec::new();
    let mut hold_bars: Vec<i64> = Vec::new();
    let mut profits: Vec<f64> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();

    let mut has_year = false;
    for (meta, r) in rows {
        if let Some((y, kind)) = meta {
            has_year = true;
            years.push(y);
            kinds.push(kind);
        }
        symbols.push(pairs.symbol_dict[r.sym_id as usize].clone());
        open_dts.push(r.open_dt);
        close_dts.push(r.close_dt);
        dirs.push(r.dir);
        open_px.push(r.open_price);
        close_px.push(r.close_price);
        hold_bars.push(r.hold_bars);
        profits.push(r.profit_bp);
        counts.push(r.count);
    }

    let open_series = Series::new("开仓时间".into(), &open_dts)
        .cast(&DataType::Datetime(pairs.time_unit, None))
        .map_err(WbtError::Polars)?;
    let close_series = Series::new("平仓时间".into(), &close_dts)
        .cast(&DataType::Datetime(pairs.time_unit, None))
        .map_err(WbtError::Polars)?;

    let mut cols: Vec<Column> = Vec::new();
    if has_year {
        cols.push(Series::new("year".into(), &years).into_column());
        cols.push(Series::new("kind".into(), &kinds).into_column());
    }
    cols.push(Series::new("symbol".into(), &symbols).into_column());
    cols.push(Series::new("交易方向".into(), &dirs).into_column());
    cols.push(open_series.into_column());
    cols.push(close_series.into_column());
    cols.push(Series::new("开仓价格".into(), &open_px).into_column());
    cols.push(Series::new("平仓价格".into(), &close_px).into_column());
    cols.push(Series::new("持仓K线数".into(), &hold_bars).into_column());
    cols.push(Series::new("盈亏比例".into(), &profits).into_column());
    cols.push(Series::new("count".into(), &counts).into_column());

    DataFrame::new_infer_height(cols).map_err(WbtError::Polars)
}

/// 由已聚合的 `AggRow` 切片构建聚合开平记录表（供 WeightBacktest 缓存复用）。
pub fn agg_to_df(pairs: &PairsSoA, agg: &[AggRow]) -> Result<DataFrame, WbtError> {
    agg_rows_to_columns(pairs, agg.iter().map(|r| (None, r.clone())))
}

/// 由已聚合的 `AggRow` 切片选取每年 best/worst 各 `top` 笔（含 `year` / `kind` 列）。
pub fn key_trades_to_df(
    pairs: &PairsSoA,
    agg: &[AggRow],
    top: usize,
) -> Result<DataFrame, WbtError> {
    let selected = select_key_trades(agg, top, pairs.time_unit);
    agg_rows_to_columns(
        pairs,
        selected
            .into_iter()
            .map(|(y, kind, i)| (Some((y, kind)), agg[i].clone())),
    )
}

/// 聚合去重后的开平记录表（聚合 + 物化，便于单测/单次调用）。
pub fn aggregated_pairs_df(pairs: &PairsSoA) -> Result<DataFrame, WbtError> {
    agg_to_df(pairs, &aggregate_pairs(pairs))
}

/// 每年最赚/最亏各 `top` 笔关键交易（聚合 + 物化，便于单测/单次调用）。
pub fn key_trades_df(pairs: &PairsSoA, top: usize) -> Result<DataFrame, WbtError> {
    key_trades_to_df(pairs, &aggregate_pairs(pairs), top)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个最小 PairsSoA。每条记录用 (sym_id, open_dt, close_dt, hold_bars, profit_bp, count)。
    fn make_pairs(rows: &[(u32, i64, i64, i64, f64, i64)], symbol_dict: Vec<String>) -> PairsSoA {
        PairsSoA {
            sym_ids: rows.iter().map(|r| r.0).collect(),
            dirs: rows.iter().map(|_| "多头").collect(),
            open_dts: rows.iter().map(|r| r.1).collect(),
            close_dts: rows.iter().map(|r| r.2).collect(),
            open_prices: rows.iter().map(|_| 100.0).collect(),
            close_prices: rows.iter().map(|_| 110.0).collect(),
            hold_bars: rows.iter().map(|r| r.3).collect(),
            event_seqs: rows.iter().map(|_| "开多->平多").collect(),
            profit_bps: rows.iter().map(|r| r.4).collect(),
            counts: rows.iter().map(|r| r.5).collect(),
            time_unit: TimeUnit::Milliseconds,
            symbol_dict,
        }
    }

    // 2024-06-03 与 2025-06-03 的毫秒时间戳
    const DT_2024: i64 = 1_717_372_800_000; // 2024-06-03 00:00:00 UTC
    const DT_2024B: i64 = 1_717_459_200_000; // 2024-06-04
    const DT_2025: i64 = 1_748_908_800_000; // 2025-06-03 00:00:00 UTC

    #[test]
    fn aggregate_empty_is_empty() {
        let pairs = make_pairs(&[], vec!["A".into()]);
        assert!(aggregate_pairs(&pairs).is_empty());
        let df = aggregated_pairs_df(&pairs).unwrap();
        assert_eq!(df.height(), 0);
        // schema 仍齐全
        assert!(df.column("count").is_ok());
        assert!(df.column("盈亏比例").is_ok());
    }

    #[test]
    fn aggregate_merges_same_open_close() {
        // 同一 (sym, open, close) 的两条部分成交：profit 100/200，量 1/3
        let pairs = make_pairs(
            &[
                (0, DT_2024, DT_2024B, 5, 100.0, 1),
                (0, DT_2024, DT_2024B, 5, 200.0, 3),
            ],
            vec!["A".into()],
        );
        let agg = aggregate_pairs(&pairs);
        assert_eq!(agg.len(), 1, "同 key 应合并为一条");
        let r = &agg[0];
        assert_eq!(r.count, 2, "count = 合并的记录数");
        assert_eq!(r.hold_bars, 5, "hold_bars 取共享值，不求和");
        // 量加权平均 = (100*1 + 200*3) / (1+3) = 175
        assert!((r.profit_bp - 175.0).abs() < 1e-9, "got {}", r.profit_bp);
    }

    #[test]
    fn aggregate_keeps_distinct_close() {
        let pairs = make_pairs(
            &[
                (0, DT_2024, DT_2024B, 5, 100.0, 1),
                (0, DT_2024, DT_2025, 5, 200.0, 1),
            ],
            vec!["A".into()],
        );
        assert_eq!(aggregate_pairs(&pairs).len(), 2, "平仓时间不同应保留两条");
    }

    #[test]
    fn key_trades_best_worst_per_year() {
        // 2024 年 3 笔：profit 300/-50/10；top=2
        let pairs = make_pairs(
            &[
                (0, DT_2024, DT_2024B, 1, 300.0, 1),
                (1, DT_2024, DT_2024B, 1, -50.0, 1),
                (2, DT_2024, DT_2024B, 1, 10.0, 1),
            ],
            vec!["A".into(), "B".into(), "C".into()],
        );
        let agg = aggregate_pairs(&pairs);
        let sel = select_key_trades(&agg, 2, TimeUnit::Milliseconds);
        let best: Vec<f64> = sel
            .iter()
            .filter(|(_, k, _)| *k == "best")
            .map(|(_, _, i)| agg[*i].profit_bp)
            .collect();
        let worst: Vec<f64> = sel
            .iter()
            .filter(|(_, k, _)| *k == "worst")
            .map(|(_, _, i)| agg[*i].profit_bp)
            .collect();
        assert_eq!(best, vec![300.0, 10.0], "best 降序取前 2");
        // worst 升序取前 2，但 10.0 已入 best 被剔除 → 仅剩 -50.0（去重避免同笔既最赚又最亏）
        assert_eq!(worst, vec![-50.0], "worst 剔除已入 best 的交易");
    }

    #[test]
    fn key_trades_top_exceeds_count() {
        // 当年仅 2 笔，top=3 → 各取 2 笔，不补空、不 panic
        let pairs = make_pairs(
            &[
                (0, DT_2024, DT_2024B, 1, 100.0, 1),
                (1, DT_2024, DT_2024B, 1, -20.0, 1),
            ],
            vec!["A".into(), "B".into()],
        );
        let agg = aggregate_pairs(&pairs);
        let sel = select_key_trades(&agg, 3, TimeUnit::Milliseconds);
        // 2 笔均入 best；worst 剔除 best 后为空 → 不重复、不补空、不 panic
        assert_eq!(sel.iter().filter(|(_, k, _)| *k == "best").count(), 2);
        assert_eq!(sel.iter().filter(|(_, k, _)| *k == "worst").count(), 0);
    }

    #[test]
    fn key_trades_groups_by_close_year() {
        let pairs = make_pairs(
            &[
                (0, DT_2024, DT_2024B, 1, 100.0, 1),
                (1, DT_2024, DT_2025, 1, 200.0, 1),
            ],
            vec!["A".into(), "B".into()],
        );
        let agg = aggregate_pairs(&pairs);
        let sel = select_key_trades(&agg, 5, TimeUnit::Milliseconds);
        let years: std::collections::BTreeSet<i32> = sel.iter().map(|(y, _, _)| *y).collect();
        assert_eq!(years, [2024, 2025].into_iter().collect(), "按平仓年份分组");
    }

    #[test]
    fn key_trades_df_has_year_kind_columns() {
        let pairs = make_pairs(&[(0, DT_2024, DT_2024B, 1, 100.0, 1)], vec!["A".into()]);
        let df = key_trades_df(&pairs, 3).unwrap();
        assert!(df.column("year").is_ok());
        assert!(df.column("kind").is_ok());
        assert!(df.column("symbol").is_ok());
        assert_eq!(df.height(), 1, "1 笔 → 仅 best 1（worst 去重后为空）");
    }
}
