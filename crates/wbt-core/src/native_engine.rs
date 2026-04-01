use crate::errors::WbtError;
use crate::report::SymbolsReport;
use crate::trade_dir::{TradeAction, TradeDir};
use chrono::Datelike;
use polars::prelude::*;
use rayon::prelude::*;

/// 极速纳秒/微秒/毫秒时间戳 -> 整数 YYYYMMDD 的转换 (仅用于冷路径)
#[inline(always)]
pub fn dt_to_date_key_fast(dt_val: i64, tu: TimeUnit) -> i32 {
    let secs = match tu {
        TimeUnit::Nanoseconds => dt_val / 1_000_000_000,
        TimeUnit::Microseconds => dt_val / 1_000_000,
        TimeUnit::Milliseconds => dt_val / 1_000,
    };
    let dt = chrono::DateTime::from_timestamp(secs, 0)
        .unwrap_or_default()
        .naive_utc();
    let d = dt.date();
    d.year() * 10000 + d.month() as i32 * 100 + d.day() as i32
}

/// 极速纯整数日序号 — 零 chrono 开销，用于热循环日边界检测
#[inline(always)]
pub fn dt_to_day_ordinal(dt_val: i64, tu: TimeUnit) -> i32 {
    let secs = match tu {
        TimeUnit::Nanoseconds => dt_val / 1_000_000_000,
        TimeUnit::Microseconds => dt_val / 1_000_000,
        TimeUnit::Milliseconds => dt_val / 1_000,
    };
    (secs / 86400) as i32
}

/// 极速计算距离 1970-01-01 的绝对天数（用于数组全量映射 O(1)）
#[inline(always)]
pub fn dt_to_days_since_epoch(dt_val: i64, tu: TimeUnit) -> i32 {
    let secs = match tu {
        TimeUnit::Nanoseconds => dt_val / 1_000_000_000,
        TimeUnit::Microseconds => dt_val / 1_000_000,
        TimeUnit::Milliseconds => dt_val / 1_000,
    };
    (secs / 86400) as i32
}

/// 内联聚合的每日等权收益数据
pub struct DailyTotals {
    pub date_keys: Vec<i32>,
    pub totals: Vec<f64>,
    /// 每日等权 n1b（基准收益）
    pub n1b_totals: Vec<f64>,
    pub start_date_key: i32,
    pub end_date_key: i32,
    /// long/short weight 计数
    pub long_count: u64,
    pub short_count: u64,
    pub total_weight_rows: u64,
    /// 未四舍五入的日均值（用于 alpha 统计计算，精度对齐 Polars mean）
    pub strategy_means: Vec<f64>,
    pub benchmark_means: Vec<f64>,
}

/// 全局 Daily 数据的 SoA 存储（延迟物化为 DataFrame）
pub struct DailysSoA {
    pub sym_ids: Vec<u32>,
    pub date_ticks: Vec<i64>,
    pub n1b: Vec<f64>,
    pub edge: Vec<f64>,
    pub ret: Vec<f64>,
    pub cost: Vec<f64>,
    pub turnover: Vec<f64>,
    pub long_edge: Vec<f64>,
    pub short_edge: Vec<f64>,
    pub long_cost: Vec<f64>,
    pub short_cost: Vec<f64>,
    pub long_turnover: Vec<f64>,
    pub short_turnover: Vec<f64>,
    pub long_return: Vec<f64>,
    pub short_return: Vec<f64>,
    pub time_unit: TimeUnit,
    pub symbol_dict: Vec<String>,
}

impl DailysSoA {
    /// 按需构建 DataFrame（仅在 Python 端调用 .dailys() 时触发）
    pub fn to_dataframe(&self) -> Result<DataFrame, WbtError> {
        let mut sym_builder = StringChunkedBuilder::new(PlSmallStr::from("symbol"), self.sym_ids.len());
        for &id in &self.sym_ids {
            sym_builder.append_value(&self.symbol_dict[id as usize]);
        }
        let sym_series = sym_builder.finish().into_series();

        let date_as_days: Vec<i32> = self
            .date_ticks
            .iter()
            .map(|&ticks| dt_to_days_since_epoch(ticks, self.time_unit))
            .collect();
        let date_series = Series::new("date".into(), date_as_days)
            .cast(&DataType::Date)
            .map_err(WbtError::Polars)?;

        Ok(DataFrame::new(vec![
            sym_series.into_column(),
            date_series.into_column(),
            Series::new("n1b".into(), &self.n1b).into_column(),
            Series::new("edge".into(), &self.edge).into_column(),
            Series::new("return".into(), &self.ret).into_column(),
            Series::new("cost".into(), &self.cost).into_column(),
            Series::new("turnover".into(), &self.turnover).into_column(),
            Series::new("long_edge".into(), &self.long_edge).into_column(),
            Series::new("short_edge".into(), &self.short_edge).into_column(),
            Series::new("long_cost".into(), &self.long_cost).into_column(),
            Series::new("short_cost".into(), &self.short_cost).into_column(),
            Series::new("long_turnover".into(), &self.long_turnover).into_column(),
            Series::new("short_turnover".into(), &self.short_turnover).into_column(),
            Series::new("long_return".into(), &self.long_return).into_column(),
            Series::new("short_return".into(), &self.short_return).into_column(),
        ])?)
    }
}

/// 全局 Pairs 数据的 SoA 存储（延迟物化为 DataFrame）
pub struct PairsSoA {
    pub sym_ids: Vec<u32>,
    pub dirs: Vec<&'static str>,
    pub open_dts: Vec<i64>,
    pub close_dts: Vec<i64>,
    pub open_prices: Vec<f64>,
    pub close_prices: Vec<f64>,
    pub hold_bars: Vec<i64>,
    pub event_seqs: Vec<&'static str>,
    pub profit_bps: Vec<f64>,
    pub counts: Vec<i64>,
    pub time_unit: TimeUnit,
    pub symbol_dict: Vec<String>,
}

impl PairsSoA {
    /// 按需构建 DataFrame（仅在 Python 端调用 .pairs() 时触发）
    pub fn to_dataframe(&self) -> Result<DataFrame, WbtError> {
        use chrono::DateTime;
        let mut hold_days = Vec::with_capacity(self.open_dts.len());
        for (open_dt, close_dt) in self.open_dts.iter().zip(self.close_dts.iter()) {
            let (o_secs, o_nano, c_secs, c_nano) = match self.time_unit {
                TimeUnit::Milliseconds => (
                    open_dt / 1000,
                    (*open_dt % 1000) as u32 * 1_000_000,
                    close_dt / 1000,
                    (*close_dt % 1000) as u32 * 1_000_000,
                ),
                TimeUnit::Microseconds => (
                    open_dt / 1000000,
                    (*open_dt % 1000000) as u32 * 1000,
                    close_dt / 1000000,
                    (*close_dt % 1000000) as u32 * 1000,
                ),
                TimeUnit::Nanoseconds => (
                    open_dt / 1_000_000_000,
                    (*open_dt % 1_000_000_000) as u32,
                    close_dt / 1_000_000_000,
                    (*close_dt % 1_000_000_000) as u32,
                ),
            };
            let o_dt = DateTime::from_timestamp(o_secs, o_nano).unwrap_or_default();
            let c_dt = DateTime::from_timestamp(c_secs, c_nano).unwrap_or_default();
            hold_days.push((c_dt - o_dt).num_days());
        }
        let mut sym_builder = StringChunkedBuilder::new(PlSmallStr::from("symbol"), self.sym_ids.len());
        for &id in &self.sym_ids {
            sym_builder.append_value(&self.symbol_dict[id as usize]);
        }
        let sym_series = sym_builder.finish().into_series();

        let open_dt_series = Series::new("开仓时间".into(), &self.open_dts)
            .cast(&DataType::Datetime(self.time_unit, None))
            .map_err(WbtError::Polars)?;
        let close_dt_series = Series::new("平仓时间".into(), &self.close_dts)
            .cast(&DataType::Datetime(self.time_unit, None))
            .map_err(WbtError::Polars)?;

        Ok(DataFrame::new(vec![
            sym_series.into_column(),
            Series::new("交易方向".into(), &self.dirs).into_column(),
            open_dt_series.into_column(),
            close_dt_series.into_column(),
            Series::new("开仓价格".into(), &self.open_prices).into_column(),
            Series::new("平仓价格".into(), &self.close_prices).into_column(),
            Series::new("持仓K线数".into(), &self.hold_bars).into_column(),
            Series::new("事件序列".into(), &self.event_seqs).into_column(),
            Series::new("盈亏比例".into(), &self.profit_bps).into_column(),
            Series::new("持仓数量".into(), &self.counts).into_column(),
            Series::new("持仓天数".into(), &hold_days).into_column(),
        ])?)
    }
}

/// 单个品种产出的局部 Data Struct of Arrays
struct SymbolDailysSoA {
    pub date_ticks: Vec<i64>,
    pub n1b: Vec<f64>,
    pub edge: Vec<f64>,
    pub ret: Vec<f64>,
    pub cost: Vec<f64>,
    pub turnover: Vec<f64>,
    pub long_edge: Vec<f64>,
    pub short_edge: Vec<f64>,
    pub long_cost: Vec<f64>,
    pub short_cost: Vec<f64>,
    pub long_turnover: Vec<f64>,
    pub short_turnover: Vec<f64>,
    pub long_return: Vec<f64>,
    pub short_return: Vec<f64>,
}

impl SymbolDailysSoA {
    fn new(cap: usize) -> Self {
        Self {
            date_ticks: Vec::with_capacity(cap),
            n1b: Vec::with_capacity(cap),
            edge: Vec::with_capacity(cap),
            ret: Vec::with_capacity(cap),
            cost: Vec::with_capacity(cap),
            turnover: Vec::with_capacity(cap),
            long_edge: Vec::with_capacity(cap),
            short_edge: Vec::with_capacity(cap),
            long_cost: Vec::with_capacity(cap),
            short_cost: Vec::with_capacity(cap),
            long_turnover: Vec::with_capacity(cap),
            short_turnover: Vec::with_capacity(cap),
            long_return: Vec::with_capacity(cap),
            short_return: Vec::with_capacity(cap),
        }
    }
}

/// 动态扩容配置的栈内撮合单元 (Struct of Arrays 动态栈)
pub struct LotsSoA {
    pub is_long: Vec<bool>,
    pub dt_ticks: Vec<i64>,
    pub bar_id: Vec<i64>,
    pub price: Vec<f64>,
    pub volume: Vec<i64>,
    pub action: Vec<TradeAction>,
    pub head: usize,
}

impl LotsSoA {
    pub fn new(capacity: usize) -> Self {
        Self {
            is_long: Vec::with_capacity(capacity),
            dt_ticks: Vec::with_capacity(capacity),
            bar_id: Vec::with_capacity(capacity),
            price: Vec::with_capacity(capacity),
            volume: Vec::with_capacity(capacity),
            action: Vec::with_capacity(capacity),
            head: 0,
        }
    }

    #[inline(always)]
    pub fn push(
        &mut self,
        is_long: bool,
        dt_ticks: i64,
        bar_id: i64,
        price: f64,
        volume: i64,
        action: TradeAction,
    ) {
        if self.head < self.is_long.len() {
            self.is_long[self.head] = is_long;
            self.dt_ticks[self.head] = dt_ticks;
            self.bar_id[self.head] = bar_id;
            self.price[self.head] = price;
            self.volume[self.head] = volume;
            self.action[self.head] = action;
        } else {
            self.is_long.push(is_long);
            self.dt_ticks.push(dt_ticks);
            self.bar_id.push(bar_id);
            self.price.push(price);
            self.volume.push(volume);
            self.action.push(action);
        }
        self.head += 1;
    }
}

impl Default for LotsSoA {
    fn default() -> Self {
        Self::new(128)
    }
}

/// 稀疏质点对
struct SymbolPairsSoA {
    pub dirs: Vec<&'static str>,
    pub open_dts: Vec<i64>,
    pub close_dts: Vec<i64>,
    pub open_prices: Vec<f64>,
    pub close_prices: Vec<f64>,
    pub hold_bars: Vec<i64>,
    pub event_seqs: Vec<&'static str>,
    pub profit_bps: Vec<f64>,
    pub counts: Vec<i64>,
    pub dirs_enum: Vec<TradeDir>,
}

impl SymbolPairsSoA {
    fn new(cap: usize) -> Self {
        Self {
            dirs: Vec::with_capacity(cap),
            open_dts: Vec::with_capacity(cap),
            close_dts: Vec::with_capacity(cap),
            open_prices: Vec::with_capacity(cap),
            close_prices: Vec::with_capacity(cap),
            hold_bars: Vec::with_capacity(cap),
            event_seqs: Vec::with_capacity(cap),
            profit_bps: Vec::with_capacity(cap),
            counts: Vec::with_capacity(cap),
            dirs_enum: Vec::with_capacity(cap),
        }
    }
}

pub struct NativeEngine;

impl NativeEngine {
    #[allow(clippy::type_complexity)]
    pub fn process_all(
        dfw: &DataFrame,
        symbols: &[String],
        digits: i64,
        fee_rate: f64,
        weight_type_is_ts: bool,
    ) -> Result<
        (
            Vec<SymbolsReport>,
            DailysSoA,
            PairsSoA,
            DailyTotals,
            Vec<String>,
        ),
        WbtError,
    > {
        let mut order_map: hashbrown::HashMap<&str, u32> =
            hashbrown::HashMap::with_capacity(symbols.len());
        let mut symbol_dict = Vec::with_capacity(symbols.len());
        for (idx, sym) in symbols.iter().enumerate() {
            order_map.insert(sym.as_str(), idx as u32);
            symbol_dict.push(sym.clone());
        }

        let dt_col = dfw.column("dt")?.as_materialized_series().datetime()?;
        let time_unit = dt_col.time_unit();
        let dt_ca = dt_col.rechunk();
        let dt_slice = dt_ca.cont_slice().unwrap();

        let weight_col = dfw.column("weight")?.as_materialized_series().f64()?;
        let weight_ca = weight_col.rechunk();
        let weight_slice = weight_ca.cont_slice().unwrap();

        let price_col = dfw.column("price")?.as_materialized_series().f64()?;
        let price_ca = price_col.rechunk();
        let price_slice = price_ca.cont_slice().unwrap();

        let sym_id_col = dfw.column("sym_id")?.as_materialized_series().u32()?.rechunk();
        let sym_id_slice = sym_id_col.cont_slice().unwrap();

        // O(N) 极速扫描切片分界点
        let mut block_bounds: Vec<(u32, usize, usize)> = Vec::with_capacity(symbols.len());
        if !sym_id_slice.is_empty() {
            let mut current_sym = sym_id_slice[0];
            let mut start_idx = 0;
            for (i, &sym_id) in sym_id_slice.iter().enumerate().skip(1) {
                if sym_id != current_sym {
                    block_bounds.push((current_sym, start_idx, i));
                    current_sym = sym_id_slice[i];
                    start_idx = i;
                }
            }
            block_bounds.push((current_sym, start_idx, sym_id_slice.len()));
        }

        let digits_pow10 = 10_f64.powi(digits as i32);

        let processed: Vec<(u32, SymbolDailysSoA, SymbolPairsSoA)> = block_bounds
            .into_par_iter()
            .map(|(sym_id, start, end)| {
                let dt_blk = &dt_slice[start..end];
                let w_blk = &weight_slice[start..end];
                let p_blk = &price_slice[start..end];

                let (dailys, pairs) = Self::process_symbol_chunk(
                    dt_blk,
                    w_blk,
                    p_blk,
                    digits_pow10,
                    fee_rate,
                    time_unit,
                );
                (sym_id, dailys, pairs)
            })
            .collect();

        // === 后处理: 全量数据合并 ===
        let total_dailys = processed
            .iter()
            .map(|(_, d, _)| d.date_ticks.len())
            .sum::<usize>();
        let total_pairs = processed
            .iter()
            .map(|(_, _, p)| p.dirs.len())
            .sum::<usize>();

        let mut d_sym = Vec::with_capacity(total_dailys);
        let mut d_date = Vec::with_capacity(total_dailys);
        let mut d_n1b = Vec::with_capacity(total_dailys);
        let mut d_edge = Vec::with_capacity(total_dailys);
        let mut d_return = Vec::with_capacity(total_dailys);
        let mut d_cost = Vec::with_capacity(total_dailys);
        let mut d_turnover = Vec::with_capacity(total_dailys);
        let mut d_long_edge = Vec::with_capacity(total_dailys);
        let mut d_short_edge = Vec::with_capacity(total_dailys);
        let mut d_long_cost = Vec::with_capacity(total_dailys);
        let mut d_short_cost = Vec::with_capacity(total_dailys);
        let mut d_long_turnover = Vec::with_capacity(total_dailys);
        let mut d_short_turnover = Vec::with_capacity(total_dailys);
        let mut d_long_return = Vec::with_capacity(total_dailys);
        let mut d_short_return = Vec::with_capacity(total_dailys);

        let mut p_sym = Vec::with_capacity(total_pairs);
        let mut p_dirs = Vec::with_capacity(total_pairs);
        let mut p_open_dts = Vec::with_capacity(total_pairs);
        let mut p_close_dts = Vec::with_capacity(total_pairs);
        let mut p_open_prices = Vec::with_capacity(total_pairs);
        let mut p_close_prices = Vec::with_capacity(total_pairs);
        let mut p_hold_bars = Vec::with_capacity(total_pairs);
        let mut p_event_seqs = Vec::with_capacity(total_pairs);
        let mut p_profit_bps = Vec::with_capacity(total_pairs);
        let mut p_counts = Vec::with_capacity(total_pairs);

        const MAX_DAYS: usize = 60000;
        let mut global_sum_by_day = vec![0.0f64; MAX_DAYS];
        let mut global_n1b_by_day = vec![0.0f64; MAX_DAYS];
        let mut global_active_by_day = vec![0u32; MAX_DAYS];
        let mut min_abs_day = 50000i32;
        let mut max_abs_day = -10000i32;
        let mut total_long_count = 0u64;
        let mut total_short_count = 0u64;
        let mut total_weight_rows = 0u64;

        let mut reports = Vec::with_capacity(symbol_dict.len());

        for (sym_id, d, p) in processed {
            let n_daily = d.date_ticks.len();
            let valid_n = n_daily;
            if valid_n > 0 {
                d_sym.extend(std::iter::repeat(sym_id).take(valid_n));
                d_date.extend_from_slice(&d.date_ticks[0..]);
                d_n1b.extend_from_slice(&d.n1b[0..]);
                d_edge.extend_from_slice(&d.edge[0..]);
                d_return.extend_from_slice(&d.ret[0..]);
                d_cost.extend_from_slice(&d.cost[0..]);
                d_turnover.extend_from_slice(&d.turnover[0..]);
                d_long_edge.extend_from_slice(&d.long_edge[0..]);
                d_short_edge.extend_from_slice(&d.short_edge[0..]);
                d_long_cost.extend_from_slice(&d.long_cost[0..]);
                d_short_cost.extend_from_slice(&d.short_cost[0..]);
                d_long_turnover.extend_from_slice(&d.long_turnover[0..]);
                d_short_turnover.extend_from_slice(&d.short_turnover[0..]);
                d_long_return.extend_from_slice(&d.long_return[0..]);
                d_short_return.extend_from_slice(&d.short_return[0..]);
            }

            for i in 0..n_daily {
                let ticks = d.date_ticks[i];
                let ret = d.ret[i];
                let n1b_val = d.n1b[i];

                let abs_day = dt_to_days_since_epoch(ticks, time_unit);
                let idx_signed = abs_day as i64 + 10000;
                if idx_signed >= 0 && (idx_signed as usize) < MAX_DAYS {
                    let idx = idx_signed as usize;
                    global_sum_by_day[idx] += ret;
                    global_n1b_by_day[idx] += n1b_val;
                    global_active_by_day[idx] += 1;
                    if abs_day < min_abs_day {
                        min_abs_day = abs_day;
                    }
                    if abs_day > max_abs_day {
                        max_abs_day = abs_day;
                    }
                }
            }

            let n_pairs = p.dirs.len();
            p_sym.extend(std::iter::repeat(sym_id).take(n_pairs));
            p_dirs.extend_from_slice(&p.dirs);
            p_open_dts.extend_from_slice(&p.open_dts);
            p_close_dts.extend_from_slice(&p.close_dts);
            p_open_prices.extend_from_slice(&p.open_prices);
            p_close_prices.extend_from_slice(&p.close_prices);
            p_hold_bars.extend_from_slice(&p.hold_bars);
            p_event_seqs.extend_from_slice(&p.event_seqs);
            p_profit_bps.extend_from_slice(&p.profit_bps);
            p_counts.extend_from_slice(&p.counts);

            reports.push(SymbolsReport {
                symbol: symbol_dict[sym_id as usize].clone(),
                daily: DataFrame::empty(),
                pair: DataFrame::empty(),
            });
        }

        // --- O(N) 一次性日均收敛提取 + n1b 基准收益 ---
        let cap = (max_abs_day - min_abs_day + 1).max(0) as usize;
        let mut out_date_keys = Vec::with_capacity(cap);
        let mut out_totals = Vec::with_capacity(cap);
        let mut out_n1b = Vec::with_capacity(cap);
        let mut out_strategy_means = Vec::with_capacity(cap);
        let mut out_benchmark_means = Vec::with_capacity(cap);

        for &w in weight_slice {
            total_weight_rows += 1;
            if w > 0.0 {
                total_long_count += 1;
            } else if w < 0.0 {
                total_short_count += 1;
            }
        }

        if min_abs_day <= max_abs_day {
            for ad in min_abs_day..=max_abs_day {
                let idx = ad as usize + 10000;
                let cnt = global_active_by_day[idx];
                if cnt > 0 {
                    let raw_sum = global_sum_by_day[idx];
                    let raw_n1b = global_n1b_by_day[idx];
                    let d = chrono::DateTime::from_timestamp((ad as i64) * 86400, 0)
                        .unwrap_or_default()
                        .naive_utc()
                        .date();
                    let date_key = d.year() * 10000 + d.month() as i32 * 100 + d.day() as i32;
                    let cnt_f = cnt as f64;
                    let (v, n) = if weight_type_is_ts {
                        (raw_sum / cnt_f, raw_n1b / cnt_f)
                    } else {
                        (raw_sum, raw_n1b)
                    };
                    out_date_keys.push(date_key);
                    out_totals.push((v * 10000.0).round() / 10000.0);
                    out_n1b.push((n * 10000.0).round() / 10000.0);
                    out_strategy_means.push(raw_sum / cnt_f);
                    out_benchmark_means.push(raw_n1b / cnt_f);
                }
            }
        }

        let daily_totals = DailyTotals {
            start_date_key: out_date_keys.first().copied().unwrap_or(0),
            end_date_key: out_date_keys.last().copied().unwrap_or(0),
            date_keys: out_date_keys,
            totals: out_totals,
            n1b_totals: out_n1b,
            long_count: total_long_count,
            short_count: total_short_count,
            total_weight_rows,
            strategy_means: out_strategy_means,
            benchmark_means: out_benchmark_means,
        };

        let dailys_soa = DailysSoA {
            sym_ids: d_sym,
            date_ticks: d_date,
            n1b: d_n1b,
            edge: d_edge,
            ret: d_return,
            cost: d_cost,
            turnover: d_turnover,
            long_edge: d_long_edge,
            short_edge: d_short_edge,
            long_cost: d_long_cost,
            short_cost: d_short_cost,
            long_turnover: d_long_turnover,
            short_turnover: d_short_turnover,
            long_return: d_long_return,
            short_return: d_short_return,
            time_unit,
            symbol_dict: symbol_dict.clone(),
        };

        let pairs_soa = PairsSoA {
            sym_ids: p_sym,
            dirs: p_dirs,
            open_dts: p_open_dts,
            close_dts: p_close_dts,
            open_prices: p_open_prices,
            close_prices: p_close_prices,
            hold_bars: p_hold_bars,
            event_seqs: p_event_seqs,
            profit_bps: p_profit_bps,
            counts: p_counts,
            time_unit,
            symbol_dict: symbol_dict.clone(),
        };

        Ok((reports, dailys_soa, pairs_soa, daily_totals, symbol_dict))
    }

    /// 核心游标流向量引擎
    #[inline(never)]
    fn process_symbol_chunk(
        dt_slice: &[i64],
        w_slice: &[f64],
        p_slice: &[f64],
        digits_pow10: f64,
        fee_rate: f64,
        tu: TimeUnit,
    ) -> (SymbolDailysSoA, SymbolPairsSoA) {
        let n = dt_slice.len();
        let mut d = SymbolDailysSoA::new(n / 2);
        let mut p = SymbolPairsSoA::new(n / 10);
        if n == 0 {
            return (d, p);
        }

        let mut lots = LotsSoA::default();

        let mut pending_dt_ticks = dt_slice[0];
        let mut pending_weight = (w_slice[0] * 10000.0).round() / 10000.0;
        let mut pending_long_weight = if pending_weight > 0.0 {
            pending_weight
        } else {
            0.0
        };
        let mut pending_short_weight = if pending_weight < 0.0 {
            pending_weight
        } else {
            0.0
        };
        let mut pending_price = p_slice[0];
        let mut pending_date_key = dt_to_day_ordinal(pending_dt_ticks, tu);

        let mut pending_turnover = 0.0;
        let mut pending_cost = 0.0;
        let mut pending_long_turnover = 0.0;
        let mut pending_long_cost = 0.0;
        let mut pending_short_turnover = 0.0;
        let mut pending_short_cost = 0.0;

        let mut prev_weight = pending_weight;
        let mut prev_long_weight = pending_long_weight;
        let mut prev_short_weight = pending_short_weight;

        let mut active_date_key = pending_date_key;
        let mut active_dt_ticks = pending_dt_ticks;

        let mut d_n1b = 0.0;
        let mut d_edge = 0.0;
        let mut d_ret = 0.0;
        let mut d_cost = 0.0;
        let mut d_turnover = 0.0;
        let mut d_long_edge = 0.0;
        let mut d_short_edge = 0.0;
        let mut d_long_cost = 0.0;
        let mut d_short_cost = 0.0;
        let mut d_long_turnover = 0.0;
        let mut d_short_turnover = 0.0;
        let mut d_long_return = 0.0;
        let mut d_short_return = 0.0;

        let mut last_volume = (pending_weight * digits_pow10).round() as i64;

        if last_volume > 0 {
            lots.push(
                true,
                dt_slice[0],
                1,
                p_slice[0],
                last_volume,
                TradeAction::OpenLong,
            );
        } else if last_volume < 0 {
            lots.push(
                false,
                dt_slice[0],
                1,
                p_slice[0],
                -last_volume,
                TradeAction::OpenShort,
            );
        }

        for i in 1..n {
            let dt = dt_slice[i];
            let price = p_slice[i];
            let raw_w = w_slice[i];
            let weight = (raw_w * 10000.0).round() / 10000.0;
            let long_weight = if weight > 0.0 { weight } else { 0.0 };
            let short_weight = if weight < 0.0 { weight } else { 0.0 };

            let curr_turnover = (prev_weight - weight).abs();
            let curr_cost = curr_turnover * fee_rate;
            let curr_long_turnover = (prev_long_weight - long_weight).abs();
            let curr_short_turnover = (prev_short_weight - short_weight).abs();
            let curr_long_cost = curr_long_turnover * fee_rate;
            let curr_short_cost = curr_short_turnover * fee_rate;

            let n1b = if pending_price == 0.0 {
                0.0
            } else {
                price / pending_price - 1.0
            };
            let edge = pending_weight * n1b;
            let ret = edge - pending_cost;
            let long_edge = pending_long_weight * n1b;
            let short_edge = pending_short_weight * n1b;
            let long_ret = long_edge - pending_long_cost;
            let short_ret = short_edge - pending_short_cost;

            let curr_date_key = dt_to_day_ordinal(dt, tu);

            if pending_date_key != active_date_key {
                d.date_ticks.push(active_dt_ticks);
                d.n1b.push(d_n1b);
                d.edge.push(d_edge);
                d.ret.push(d_ret);
                d.cost.push(d_cost);
                d.turnover.push(d_turnover);
                d.long_edge.push(d_long_edge);
                d.short_edge.push(d_short_edge);
                d.long_cost.push(d_long_cost);
                d.short_cost.push(d_short_cost);
                d.long_turnover.push(d_long_turnover);
                d.short_turnover.push(d_short_turnover);
                d.long_return.push(d_long_return);
                d.short_return.push(d_short_return);

                d_n1b = 0.0;
                d_edge = 0.0;
                d_ret = 0.0;
                d_cost = 0.0;
                d_turnover = 0.0;
                d_long_edge = 0.0;
                d_short_edge = 0.0;
                d_long_cost = 0.0;
                d_short_cost = 0.0;
                d_long_turnover = 0.0;
                d_short_turnover = 0.0;
                d_long_return = 0.0;
                d_short_return = 0.0;

                active_date_key = pending_date_key;
                active_dt_ticks = pending_dt_ticks;
            }

            d_n1b += n1b;
            d_edge += edge;
            d_ret += ret;
            d_cost += pending_cost;
            d_turnover += pending_turnover;
            d_long_edge += long_edge;
            d_short_edge += short_edge;
            d_long_cost += pending_long_cost;
            d_short_cost += pending_short_cost;
            d_long_turnover += pending_long_turnover;
            d_short_turnover += pending_short_turnover;
            d_long_return += long_ret;
            d_short_return += short_ret;

            pending_dt_ticks = dt;
            pending_price = price;
            pending_weight = weight;
            pending_long_weight = long_weight;
            pending_short_weight = short_weight;
            pending_date_key = curr_date_key;

            pending_turnover = curr_turnover;
            pending_cost = curr_cost;
            pending_long_turnover = curr_long_turnover;
            pending_long_cost = curr_long_cost;
            pending_short_turnover = curr_short_turnover;
            pending_short_cost = curr_short_cost;

            prev_weight = weight;
            prev_long_weight = long_weight;
            prev_short_weight = short_weight;

            let curr_volume = (weight * digits_pow10).round() as i64;
            if curr_volume != last_volume {
                let bar_id = (i + 1) as i64;
                Self::process_pairs_block_matching(
                    &mut lots,
                    &mut p,
                    last_volume,
                    curr_volume,
                    dt,
                    bar_id,
                    price,
                );
                last_volume = curr_volume;
            }
        }

        // Push final day piece
        d.date_ticks.push(active_dt_ticks);
        d.n1b.push(d_n1b);
        d.edge.push(d_edge);
        d.ret.push(d_ret);
        d.cost.push(d_cost);
        d.turnover.push(d_turnover);
        d.long_edge.push(d_long_edge);
        d.short_edge.push(d_short_edge);
        d.long_cost.push(d_long_cost);
        d.short_cost.push(d_short_cost);
        d.long_turnover.push(d_long_turnover);
        d.short_turnover.push(d_short_turnover);
        d.long_return.push(d_long_return);
        d.short_return.push(d_short_return);

        (d, p)
    }

    /// O(1) Block Matching Method
    #[inline(always)]
    fn process_pairs_block_matching(
        lots: &mut LotsSoA,
        p: &mut SymbolPairsSoA,
        last_vol: i64,
        curr_vol: i64,
        dt_ticks: i64,
        bar_id: i64,
        price: f64,
    ) {
        if last_vol >= 0 && curr_vol >= 0 {
            let diff = curr_vol - last_vol;
            if diff > 0 {
                Self::open(lots, true, dt_ticks, bar_id, price, diff);
            } else {
                Self::close(
                    lots,
                    p,
                    TradeAction::CloseLong,
                    dt_ticks,
                    bar_id,
                    price,
                    -diff,
                );
            }
            return;
        }

        if last_vol <= 0 && curr_vol <= 0 {
            let diff = curr_vol - last_vol;
            if diff > 0 {
                Self::close(
                    lots,
                    p,
                    TradeAction::CloseShort,
                    dt_ticks,
                    bar_id,
                    price,
                    diff,
                );
            } else {
                Self::open(lots, false, dt_ticks, bar_id, price, -diff);
            }
            return;
        }

        if last_vol >= 0 && curr_vol <= 0 {
            if last_vol > 0 {
                Self::close(
                    lots,
                    p,
                    TradeAction::CloseLong,
                    dt_ticks,
                    bar_id,
                    price,
                    last_vol,
                );
            }
            if curr_vol < 0 {
                Self::open(lots, false, dt_ticks, bar_id, price, -curr_vol);
            }
            return;
        }

        if last_vol < 0 {
            Self::close(
                lots,
                p,
                TradeAction::CloseShort,
                dt_ticks,
                bar_id,
                price,
                -last_vol,
            );
        }
        if curr_vol > 0 {
            Self::open(lots, true, dt_ticks, bar_id, price, curr_vol);
        }
    }

    #[inline(always)]
    fn open(lots: &mut LotsSoA, is_long: bool, dt_ticks: i64, bar_id: i64, price: f64, count: i64) {
        if count <= 0 {
            return;
        }
        let action = if is_long {
            TradeAction::OpenLong
        } else {
            TradeAction::OpenShort
        };
        lots.push(is_long, dt_ticks, bar_id, price, count, action);
    }

    #[inline(always)]
    fn close(
        lots: &mut LotsSoA,
        p: &mut SymbolPairsSoA,
        close_action: TradeAction,
        close_dt: i64,
        bar_id: i64,
        price: f64,
        mut count: i64,
    ) {
        while count > 0 && lots.head > 0 {
            let last_idx = lots.head - 1;
            let lot_count = lots.volume[last_idx];
            let matched = lot_count.min(count);
            count -= matched;

            let is_long = lots.is_long[last_idx];
            let lot_px = lots.price[last_idx];
            let lot_bid = lots.bar_id[last_idx];
            let lot_dt = lots.dt_ticks[last_idx];
            let lot_act = lots.action[last_idx];

            if matched < lot_count {
                lots.volume[last_idx] -= matched;
            } else {
                lots.head -= 1;
            }

            let profit_bp = if lot_px == 0.0 {
                0.0
            } else if is_long {
                ((price - lot_px) / lot_px * 10000.0 * 100.0).round() / 100.0
            } else {
                ((lot_px - price) / lot_px * 10000.0 * 100.0).round() / 100.0
            };

            let hold_bars = bar_id - lot_bid + 1;
            let event_seq = lot_act.get_event_seq(close_action);

            p.dirs.push(if is_long { "多头" } else { "空头" });
            p.dirs_enum.push(if is_long {
                TradeDir::Long
            } else {
                TradeDir::Short
            });
            p.open_dts.push(lot_dt);
            p.close_dts.push(close_dt);
            p.open_prices.push(lot_px);
            p.close_prices.push(price);
            p.hold_bars.push(hold_bars);
            p.event_seqs.push(event_seq);
            p.profit_bps.push(profit_bp);
            p.counts.push(matched);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_key_fast_matches_original_across_time_units() {
        let base_secs = 1_735_689_600_i64; // 2025-01-01 00:00:00 UTC
        let stamps = vec![
            base_secs,
            base_secs + 1,
            base_secs + 86_399,
            base_secs + 86_400,
            base_secs + 31 * 86_400,
            base_secs + 365 * 86_400,
        ];

        let tus = [
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ];

        for tu in tus {
            for sec in &stamps {
                let raw = match tu {
                    TimeUnit::Milliseconds => sec * 1_000,
                    TimeUnit::Microseconds => sec * 1_000_000,
                    TimeUnit::Nanoseconds => sec * 1_000_000_000,
                };
                let expected = {
                    let secs2 = match tu {
                        TimeUnit::Nanoseconds => raw / 1_000_000_000,
                        TimeUnit::Microseconds => raw / 1_000_000,
                        TimeUnit::Milliseconds => raw / 1_000,
                    };
                    let dt = chrono::DateTime::from_timestamp(secs2, 0)
                        .unwrap()
                        .naive_utc();
                    let d = dt.date();
                    d.year() * 10000 + d.month() as i32 * 100 + d.day() as i32
                };
                let actual = dt_to_date_key_fast(raw, tu);
                assert_eq!(actual, expected, "time_unit={tu:?}, raw={raw}");
            }
        }
    }
}
