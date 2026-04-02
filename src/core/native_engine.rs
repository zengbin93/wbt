use crate::core::errors::WbtError;
use crate::core::report::SymbolsReport;
use crate::core::trade_dir::{TradeAction, TradeDir};
use chrono::Datelike;
use polars::prelude::*;
use rayon::prelude::*;

/// Sparse day-index array capacity (~164 years from epoch)
const DAY_INDEX_CAPACITY: usize = 60_000;
/// Offset to shift days-since-epoch into non-negative index space
const DAY_INDEX_OFFSET: i64 = 10_000;

#[inline(always)]
fn dt_val_to_secs(dt_val: i64, tu: TimeUnit) -> i64 {
    match tu {
        TimeUnit::Nanoseconds => dt_val / 1_000_000_000,
        TimeUnit::Microseconds => dt_val / 1_000_000,
        TimeUnit::Milliseconds => dt_val / 1_000,
    }
}

/// 极速纳秒/微秒/毫秒时间戳 -> 整数 YYYYMMDD 的转换 (仅用于冷路径)
#[inline(always)]
pub fn dt_to_date_key_fast(dt_val: i64, tu: TimeUnit) -> i32 {
    let secs = dt_val_to_secs(dt_val, tu);
    let dt = chrono::DateTime::from_timestamp(secs, 0)
        .unwrap_or_default()
        .naive_utc();
    let d = dt.date();
    d.year() * 10000 + d.month() as i32 * 100 + d.day() as i32
}

/// 极速计算距离 1970-01-01 的绝对天数（用于日边界检测和数组全量映射 O(1)）
#[inline(always)]
pub fn dt_to_days_since_epoch(dt_val: i64, tu: TimeUnit) -> i32 {
    (dt_val_to_secs(dt_val, tu) / 86400) as i32
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
        let sym_strs: Vec<&str> = self
            .sym_ids
            .iter()
            .map(|&id| self.symbol_dict[id as usize].as_str())
            .collect();
        let sym_series = Series::new("symbol".into(), &sym_strs);

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
        let sym_strs: Vec<&str> = self
            .sym_ids
            .iter()
            .map(|&id| self.symbol_dict[id as usize].as_str())
            .collect();
        let sym_series = Series::new("symbol".into(), &sym_strs);

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

struct DailysBuilder {
    sym: Vec<u32>,
    date: Vec<i64>,
    n1b: Vec<f64>,
    edge: Vec<f64>,
    ret: Vec<f64>,
    cost: Vec<f64>,
    turnover: Vec<f64>,
    long_edge: Vec<f64>,
    short_edge: Vec<f64>,
    long_cost: Vec<f64>,
    short_cost: Vec<f64>,
    long_turnover: Vec<f64>,
    short_turnover: Vec<f64>,
    long_return: Vec<f64>,
    short_return: Vec<f64>,
}

impl DailysBuilder {
    fn with_capacity(cap: usize) -> Self {
        Self {
            sym: Vec::with_capacity(cap),
            date: Vec::with_capacity(cap),
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

    fn extend(&mut self, sym_id: u32, s: &SymbolDailysSoA) {
        let n = s.date_ticks.len();
        if n == 0 {
            return;
        }
        self.sym.extend(std::iter::repeat_n(sym_id, n));
        self.date.extend_from_slice(&s.date_ticks);
        self.n1b.extend_from_slice(&s.n1b);
        self.edge.extend_from_slice(&s.edge);
        self.ret.extend_from_slice(&s.ret);
        self.cost.extend_from_slice(&s.cost);
        self.turnover.extend_from_slice(&s.turnover);
        self.long_edge.extend_from_slice(&s.long_edge);
        self.short_edge.extend_from_slice(&s.short_edge);
        self.long_cost.extend_from_slice(&s.long_cost);
        self.short_cost.extend_from_slice(&s.short_cost);
        self.long_turnover.extend_from_slice(&s.long_turnover);
        self.short_turnover.extend_from_slice(&s.short_turnover);
        self.long_return.extend_from_slice(&s.long_return);
        self.short_return.extend_from_slice(&s.short_return);
    }
}

struct PairsBuilder {
    sym: Vec<u32>,
    dirs: Vec<&'static str>,
    open_dts: Vec<i64>,
    close_dts: Vec<i64>,
    open_prices: Vec<f64>,
    close_prices: Vec<f64>,
    hold_bars: Vec<i64>,
    event_seqs: Vec<&'static str>,
    profit_bps: Vec<f64>,
    counts: Vec<i64>,
}

impl PairsBuilder {
    fn with_capacity(cap: usize) -> Self {
        Self {
            sym: Vec::with_capacity(cap),
            dirs: Vec::with_capacity(cap),
            open_dts: Vec::with_capacity(cap),
            close_dts: Vec::with_capacity(cap),
            open_prices: Vec::with_capacity(cap),
            close_prices: Vec::with_capacity(cap),
            hold_bars: Vec::with_capacity(cap),
            event_seqs: Vec::with_capacity(cap),
            profit_bps: Vec::with_capacity(cap),
            counts: Vec::with_capacity(cap),
        }
    }

    fn extend(&mut self, sym_id: u32, p: &SymbolPairsSoA) {
        let n = p.dirs.len();
        if n == 0 {
            return;
        }
        self.sym.extend(std::iter::repeat_n(sym_id, n));
        self.dirs.extend_from_slice(&p.dirs);
        self.open_dts.extend_from_slice(&p.open_dts);
        self.close_dts.extend_from_slice(&p.close_dts);
        self.open_prices.extend_from_slice(&p.open_prices);
        self.close_prices.extend_from_slice(&p.close_prices);
        self.hold_bars.extend_from_slice(&p.hold_bars);
        self.event_seqs.extend_from_slice(&p.event_seqs);
        self.profit_bps.extend_from_slice(&p.profit_bps);
        self.counts.extend_from_slice(&p.counts);
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

        let sym_id_col = dfw
            .column("sym_id")?
            .as_materialized_series()
            .u32()?
            .rechunk();
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

        let mut dailys_builder = DailysBuilder::with_capacity(total_dailys);
        let mut pairs_builder = PairsBuilder::with_capacity(total_pairs);

        let mut global_sum_by_day = vec![0.0f64; DAY_INDEX_CAPACITY];
        let mut global_n1b_by_day = vec![0.0f64; DAY_INDEX_CAPACITY];
        let mut global_active_by_day = vec![0u32; DAY_INDEX_CAPACITY];
        let mut min_abs_day = 50000i32;
        let mut max_abs_day = -(DAY_INDEX_OFFSET as i32);
        let mut total_long_count = 0u64;
        let mut total_short_count = 0u64;
        let mut total_weight_rows = 0u64;

        let mut reports = Vec::with_capacity(symbol_dict.len());

        for (sym_id, d, p) in processed {
            let n_daily = d.date_ticks.len();
            dailys_builder.extend(sym_id, &d);

            for i in 0..n_daily {
                let ticks = d.date_ticks[i];
                let ret = d.ret[i];
                let n1b_val = d.n1b[i];

                let abs_day = dt_to_days_since_epoch(ticks, time_unit);
                let idx_signed = abs_day as i64 + DAY_INDEX_OFFSET;
                if idx_signed >= 0 && (idx_signed as usize) < DAY_INDEX_CAPACITY {
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

            pairs_builder.extend(sym_id, &p);

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
                let idx = ad as usize + DAY_INDEX_OFFSET as usize;
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
                    out_totals.push(v);
                    out_n1b.push(n);
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
            sym_ids: dailys_builder.sym,
            date_ticks: dailys_builder.date,
            n1b: dailys_builder.n1b,
            edge: dailys_builder.edge,
            ret: dailys_builder.ret,
            cost: dailys_builder.cost,
            turnover: dailys_builder.turnover,
            long_edge: dailys_builder.long_edge,
            short_edge: dailys_builder.short_edge,
            long_cost: dailys_builder.long_cost,
            short_cost: dailys_builder.short_cost,
            long_turnover: dailys_builder.long_turnover,
            short_turnover: dailys_builder.short_turnover,
            long_return: dailys_builder.long_return,
            short_return: dailys_builder.short_return,
            time_unit,
            symbol_dict: symbol_dict.clone(),
        };

        let pairs_soa = PairsSoA {
            sym_ids: pairs_builder.sym,
            dirs: pairs_builder.dirs,
            open_dts: pairs_builder.open_dts,
            close_dts: pairs_builder.close_dts,
            open_prices: pairs_builder.open_prices,
            close_prices: pairs_builder.close_prices,
            hold_bars: pairs_builder.hold_bars,
            event_seqs: pairs_builder.event_seqs,
            profit_bps: pairs_builder.profit_bps,
            counts: pairs_builder.counts,
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
        let mut pending_date_key = dt_to_days_since_epoch(pending_dt_ticks, tu);

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

            let curr_date_key = dt_to_days_since_epoch(dt, tu);

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

    // --- dt_to_days_since_epoch ---
    #[test]
    fn days_since_epoch_known_date() {
        let secs = 1_735_689_600_i64; // 2025-01-01
        let expected_days = (secs / 86400) as i32;
        assert_eq!(
            dt_to_days_since_epoch(secs * 1000, TimeUnit::Milliseconds),
            expected_days
        );
        assert_eq!(
            dt_to_days_since_epoch(secs * 1_000_000, TimeUnit::Microseconds),
            expected_days
        );
        assert_eq!(
            dt_to_days_since_epoch(secs * 1_000_000_000, TimeUnit::Nanoseconds),
            expected_days
        );
    }

    // --- LotsSoA ---
    #[test]
    fn lots_soa_push_and_state() {
        let mut lots = LotsSoA::new(4);
        assert_eq!(lots.head, 0);

        lots.push(true, 1000, 1, 100.0, 5, TradeAction::OpenLong);
        assert_eq!(lots.head, 1);
        assert!(lots.is_long[0]);
        assert_eq!(lots.volume[0], 5);
        assert_eq!(lots.action[0], TradeAction::OpenLong);

        lots.push(false, 2000, 2, 200.0, 3, TradeAction::OpenShort);
        assert_eq!(lots.head, 2);
        assert!(!lots.is_long[1]);
    }

    #[test]
    fn lots_soa_reuse_slots() {
        let mut lots = LotsSoA::new(2);
        lots.push(true, 1000, 1, 100.0, 5, TradeAction::OpenLong);
        lots.push(true, 2000, 2, 200.0, 3, TradeAction::OpenLong);
        lots.head = 1;
        lots.push(false, 3000, 3, 300.0, 7, TradeAction::OpenShort);
        assert_eq!(lots.head, 2);
        assert!(!lots.is_long[1]);
        assert_eq!(lots.volume[1], 7);
    }

    // --- DailysSoA::to_dataframe ---
    #[test]
    fn dailys_soa_to_dataframe() {
        let soa = DailysSoA {
            sym_ids: vec![0, 0, 1],
            date_ticks: vec![1_735_689_600_000, 1_735_776_000_000, 1_735_689_600_000],
            n1b: vec![0.01, -0.02, 0.03],
            edge: vec![0.005, -0.01, 0.015],
            ret: vec![0.004, -0.011, 0.014],
            cost: vec![0.001, 0.001, 0.001],
            turnover: vec![0.5, 0.3, 0.4],
            long_edge: vec![0.005, 0.0, 0.015],
            short_edge: vec![0.0, -0.01, 0.0],
            long_cost: vec![0.001, 0.0, 0.001],
            short_cost: vec![0.0, 0.001, 0.0],
            long_turnover: vec![0.5, 0.0, 0.4],
            short_turnover: vec![0.0, 0.3, 0.0],
            long_return: vec![0.004, 0.0, 0.014],
            short_return: vec![0.0, -0.011, 0.0],
            time_unit: TimeUnit::Milliseconds,
            symbol_dict: vec!["A".into(), "B".into()],
        };
        let df = soa.to_dataframe().unwrap();
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 15);
        let expected_cols = [
            "symbol",
            "date",
            "n1b",
            "edge",
            "return",
            "cost",
            "turnover",
            "long_edge",
            "short_edge",
            "long_cost",
            "short_cost",
            "long_turnover",
            "short_turnover",
            "long_return",
            "short_return",
        ];
        for col in expected_cols {
            assert!(df.column(col).is_ok(), "missing column: {col}");
        }
    }

    // ===================================================================
    // process_symbol_chunk — direct unit tests
    // ===================================================================

    const FEE_RATE: f64 = 0.0001;
    const DIGITS_POW10: f64 = 100.0; // digits=2

    // 2024-01-02 00:00:00 UTC in milliseconds
    const DAY1_BASE: i64 = 1_704_153_600_000;
    const HOUR_MS: i64 = 3_600_000;
    // 2024-01-03 00:00:00 UTC in milliseconds
    const DAY2_BASE: i64 = DAY1_BASE + 86_400_000;

    fn assert_f64_eq(actual: f64, expected: f64, label: &str) {
        assert!(
            (actual - expected).abs() < 1e-8,
            "{label}: expected {expected}, got {actual}"
        );
    }

    // Test A: Single day, single bar transition (simplest case)
    #[test]
    fn chunk_single_day_single_transition() {
        let dt = [DAY1_BASE + 9 * HOUR_MS, DAY1_BASE + 10 * HOUR_MS];
        let weight = [0.5, 0.0];
        let price = [100.0, 102.0];

        let (d, _p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        assert_eq!(d.date_ticks.len(), 1, "should produce 1 daily row");

        // n1b = 102/100 - 1 = 0.02
        let n1b = 0.02;
        assert_f64_eq(d.n1b[0], n1b, "n1b");

        // edge = 0.5 * 0.02 = 0.01
        let edge = 0.5 * n1b;
        assert_f64_eq(d.edge[0], edge, "edge");

        // pending_cost at i=1 is still 0 (from init), so ret = edge - 0
        assert_f64_eq(d.ret[0], edge, "ret");

        // d_cost accumulates pending_cost=0 (from init)
        assert_f64_eq(d.cost[0], 0.0, "cost");

        // long_edge = 0.5 * 0.02 = 0.01
        assert_f64_eq(d.long_edge[0], edge, "long_edge");
        assert_f64_eq(d.short_edge[0], 0.0, "short_edge");
        assert_f64_eq(d.long_return[0], edge, "long_return");
        assert_f64_eq(d.short_return[0], 0.0, "short_return");
    }

    // Test B: Multi-bar per day aggregation
    #[test]
    fn chunk_multi_bar_single_day() {
        let dt = [
            DAY1_BASE + 9 * HOUR_MS,
            DAY1_BASE + 10 * HOUR_MS,
            DAY1_BASE + 11 * HOUR_MS,
            DAY1_BASE + 12 * HOUR_MS,
        ];
        let weight = [0.5, 0.5, 0.3, 0.0];
        let price = [100.0, 102.0, 101.0, 103.0];

        let (d, _p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        assert_eq!(d.date_ticks.len(), 1);

        // bar1->2: n1b = 102/100 - 1 = 0.02, edge = 0.5*0.02 = 0.01
        // bar2->3: n1b = 101/102 - 1 = -1/102, edge = 0.5*(-1/102)
        // bar3->4: n1b = 103/101 - 1 = 2/101, edge = 0.3*(2/101)
        let n1b_1 = 102.0 / 100.0 - 1.0;
        let n1b_2 = 101.0 / 102.0 - 1.0;
        let n1b_3 = 103.0 / 101.0 - 1.0;

        let edge_1 = 0.5 * n1b_1;
        let edge_2 = 0.5 * n1b_2;
        let edge_3 = 0.3 * n1b_3;

        assert_f64_eq(d.n1b[0], n1b_1 + n1b_2 + n1b_3, "n1b");
        assert_f64_eq(d.edge[0], edge_1 + edge_2 + edge_3, "edge");

        // pending_cost at i=1: 0 (init)
        // pending_cost at i=2: |0.5-0.5|*FEE = 0
        // pending_cost at i=3: |0.5-0.3|*FEE = 0.2*FEE
        // d_cost = 0 + 0 + 0.2*FEE
        let cost_at_2 = 0.0; // |0.5-0.5|*FEE
        let cost_at_3 = 0.2 * FEE_RATE; // |0.5-0.3|*FEE
        assert_f64_eq(d.cost[0], 0.0 + cost_at_2 + cost_at_3, "cost");

        // ret: sum of (edge_i - pending_cost_i)
        // i=1: edge_1 - 0
        // i=2: edge_2 - 0  (pending_cost from i=1 was |0.5-0.5|*FEE=0... wait)
        // Actually pending_cost after i=1 = |0.5-0.5|*FEE = 0 (prev_weight at i=1 init is 0.5, weight[1]=0.5)
        // Wait: prev_weight is initialized to pending_weight = weight[0] = 0.5
        // At i=1: curr_cost = |prev_weight - weight| = |0.5 - 0.5| * FEE = 0
        //         ret = edge_1 - pending_cost(=0) = edge_1
        //         then pending_cost = 0
        // At i=2: curr_cost = |0.5 - 0.3| * FEE = 0.2*FEE
        //         ret = edge_2 - pending_cost(=0) = edge_2
        //         then pending_cost = 0.2*FEE
        // At i=3: curr_cost = |0.3 - 0.0| * FEE = 0.3*FEE
        //         ret = edge_3 - pending_cost(=0.2*FEE) = edge_3 - 0.2*FEE
        //         then pending_cost = 0.3*FEE
        let ret_total = edge_1 + edge_2 + (edge_3 - 0.2 * FEE_RATE);
        assert_f64_eq(d.ret[0], ret_total, "ret");
    }

    // Test C: Two days, day boundary flush
    #[test]
    fn chunk_two_days_boundary() {
        let dt = [
            DAY1_BASE + 9 * HOUR_MS,
            DAY1_BASE + 10 * HOUR_MS,
            DAY2_BASE + 9 * HOUR_MS,
            DAY2_BASE + 10 * HOUR_MS,
        ];
        let weight = [0.5, 0.3, 0.3, 0.0];
        let price = [100.0, 102.0, 104.0, 106.0];

        let (d, _p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        assert_eq!(d.date_ticks.len(), 2, "should produce 2 daily rows");

        // Day boundary flush is triggered when pending_date_key != active_date_key.
        // The cross-day bar transition (bar1->bar2) is accumulated BEFORE the flush
        // at bar i=2, so day1 contains transitions i=1 AND i=2.
        //
        // i=1: n1b = 102/100 - 1 = 0.02, edge = 0.5 * 0.02
        // i=2: n1b = 104/102 - 1, edge = 0.3 * (104/102 - 1)
        let n1b_i1 = 102.0 / 100.0 - 1.0;
        let n1b_i2 = 104.0 / 102.0 - 1.0;
        let edge_i1 = 0.5 * n1b_i1;
        let edge_i2 = 0.3 * n1b_i2;

        assert_f64_eq(d.n1b[0], n1b_i1 + n1b_i2, "day1 n1b");
        assert_f64_eq(d.edge[0], edge_i1 + edge_i2, "day1 edge");

        // Day 2: transition i=3 only
        // i=3: n1b = 106/104 - 1, edge = 0.3 * (106/104 - 1)
        let n1b_i3 = 106.0 / 104.0 - 1.0;
        let edge_d2 = 0.3 * n1b_i3;
        assert_f64_eq(d.n1b[1], n1b_i3, "day2 n1b");
        assert_f64_eq(d.edge[1], edge_d2, "day2 edge");

        // Verify date_ticks: day1 uses the first bar's dt
        assert_eq!(d.date_ticks[0], DAY1_BASE + 9 * HOUR_MS);
    }

    // Test D: Short position basic
    #[test]
    fn chunk_short_position_basic() {
        let dt = [DAY1_BASE + 9 * HOUR_MS, DAY1_BASE + 10 * HOUR_MS];
        let weight = [-0.3, 0.0];
        let price = [100.0, 98.0];

        let (d, _p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        assert_eq!(d.date_ticks.len(), 1);

        let n1b = 98.0 / 100.0 - 1.0; // -0.02
        assert_f64_eq(d.n1b[0], n1b, "n1b");

        // edge = pending_weight * n1b = -0.3 * (-0.02) = 0.006
        let edge = -0.3 * n1b;
        assert_f64_eq(d.edge[0], edge, "edge");

        // long_edge = pending_long_weight * n1b = 0 * n1b = 0
        assert_f64_eq(d.long_edge[0], 0.0, "long_edge");

        // short_edge = pending_short_weight * n1b = -0.3 * (-0.02) = 0.006
        let short_edge = -0.3 * n1b;
        assert_f64_eq(d.short_edge[0], short_edge, "short_edge");
    }

    // Test E: Long-to-short crossover
    #[test]
    fn chunk_long_to_short_crossover() {
        let dt = [
            DAY1_BASE + 9 * HOUR_MS,
            DAY1_BASE + 10 * HOUR_MS,
            DAY1_BASE + 11 * HOUR_MS,
        ];
        let weight = [0.5, -0.3, 0.0];
        let price = [100.0, 102.0, 100.0];

        let (d, _p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        assert_eq!(d.date_ticks.len(), 1);

        // i=1: n1b = 102/100 - 1 = 0.02
        //   edge = 0.5 * 0.02 = 0.01
        //   long_edge = 0.5 * 0.02 = 0.01
        //   short_edge = 0.0 * 0.02 = 0.0
        let n1b_1 = 102.0 / 100.0 - 1.0;

        // i=2: n1b = 100/102 - 1 = -2/102
        //   edge = -0.3 * (-2/102) = 0.3*2/102
        //   long_edge = 0.0 * n1b = 0.0  (pending_long_weight = 0 since weight=-0.3)
        //   short_edge = -0.3 * (-2/102) = positive
        let n1b_2 = 100.0 / 102.0 - 1.0;
        let edge_2 = -0.3 * n1b_2;
        let short_edge_2 = -0.3 * n1b_2;

        assert_f64_eq(d.long_edge[0], 0.5 * n1b_1, "long_edge");
        assert_f64_eq(d.short_edge[0], short_edge_2, "short_edge");
        assert_f64_eq(d.edge[0], 0.5 * n1b_1 + edge_2, "total edge");
    }

    // ===================================================================
    // process_pairs_block_matching — via process_symbol_chunk
    // ===================================================================

    // Test F: Simple long open/close
    #[test]
    fn pairs_simple_long_open_close() {
        let dt = [
            DAY1_BASE + 9 * HOUR_MS,
            DAY1_BASE + 10 * HOUR_MS,
            DAY1_BASE + 11 * HOUR_MS,
        ];
        let weight = [0.0, 0.5, 0.0];
        let price = [100.0, 102.0, 105.0];

        let (_d, p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        // weight goes 0 -> 0.5 -> 0
        // At init: last_volume = (0.0 * 100).round() = 0 -> no open
        // At i=1: curr_volume = (0.5*100).round() = 50, diff=50>0 -> open long 50 lots at price 102
        // At i=2: curr_volume = 0, last_volume=50 -> close long 50 lots at price 105
        assert_eq!(p.dirs.len(), 1, "should produce 1 pair");
        assert_eq!(p.dirs[0], "多头");
        assert_f64_eq(p.open_prices[0], 102.0, "open_price");
        assert_f64_eq(p.close_prices[0], 105.0, "close_price");

        // profit_bp = (105 - 102) / 102 * 10000 rounded to 2 decimals
        let expected_bp = ((105.0_f64 - 102.0) / 102.0 * 10000.0 * 100.0).round() / 100.0;
        assert_f64_eq(p.profit_bps[0], expected_bp, "profit_bp");

        // hold_bars = bar_id_close - bar_id_open + 1 = 3 - 2 + 1 = 2
        assert_eq!(p.hold_bars[0], 2);
        assert_eq!(p.counts[0], 50);
    }

    // Test G: Simple short open/close
    #[test]
    fn pairs_simple_short_open_close() {
        let dt = [
            DAY1_BASE + 9 * HOUR_MS,
            DAY1_BASE + 10 * HOUR_MS,
            DAY1_BASE + 11 * HOUR_MS,
        ];
        let weight = [0.0, -0.3, 0.0];
        let price = [100.0, 102.0, 98.0];

        let (_d, p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        // At init: last_volume = 0 -> no open
        // At i=1: curr_volume = (-0.3*100).round() = -30
        //   last_vol=0 >= 0 && curr_vol=-30 <= 0 -> close long (0, skip) + open short 30 at price 102
        // At i=2: curr_volume = 0
        //   last_vol=-30 <= 0 && curr_vol=0 >= 0... wait, 0 >= 0 && 0 <= 0?
        //   Actually: last_vol=-30 <= 0 && curr_vol=0 <= 0 -> yes! diff = 0 - (-30) = 30 > 0 -> close short 30 at price 98

        assert_eq!(p.dirs.len(), 1, "should produce 1 pair");
        assert_eq!(p.dirs[0], "空头");
        assert_f64_eq(p.open_prices[0], 102.0, "open_price");
        assert_f64_eq(p.close_prices[0], 98.0, "close_price");

        // profit_bp = (102 - 98) / 102 * 10000 rounded to 2 decimals
        let expected_bp = ((102.0_f64 - 98.0) / 102.0 * 10000.0 * 100.0).round() / 100.0;
        assert_f64_eq(p.profit_bps[0], expected_bp, "profit_bp");
        assert_eq!(p.hold_bars[0], 2); // bar 3 - bar 2 + 1
        assert_eq!(p.counts[0], 30);
    }

    // Test H: Long increase (same direction)
    #[test]
    fn pairs_long_increase_same_direction() {
        let dt = [
            DAY1_BASE + 9 * HOUR_MS,
            DAY1_BASE + 10 * HOUR_MS,
            DAY1_BASE + 11 * HOUR_MS,
            DAY1_BASE + 12 * HOUR_MS,
        ];
        let weight = [0.0, 0.3, 0.5, 0.0];
        let price = [100.0, 102.0, 104.0, 106.0];

        let (_d, p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        // At init: last_volume = 0
        // At i=1: curr_volume = 30. open long 30 at price 102. bar_id=2
        // At i=2: curr_volume = 50. diff=20>0. open long 20 more at price 104. bar_id=3
        // At i=3: curr_volume = 0. close long 50. bar_id=4
        //   Close uses LIFO (stack). Last in = 20 lots at bar 3. Then 30 lots at bar 2.
        //   So we get 2 pairs.

        assert_eq!(p.dirs.len(), 2, "should produce 2 pairs (LIFO close)");
        assert_eq!(p.dirs[0], "多头");
        assert_eq!(p.dirs[1], "多头");

        // First closed: 20 lots opened at bar 3 (price 104), closed at bar 4 (price 106)
        assert_f64_eq(p.open_prices[0], 104.0, "pair1 open_price");
        assert_f64_eq(p.close_prices[0], 106.0, "pair1 close_price");
        assert_eq!(p.counts[0], 20);
        assert_eq!(p.hold_bars[0], 4 - 3 + 1); // 2

        // Second closed: 30 lots opened at bar 2 (price 102), closed at bar 4 (price 106)
        assert_f64_eq(p.open_prices[1], 102.0, "pair2 open_price");
        assert_f64_eq(p.close_prices[1], 106.0, "pair2 close_price");
        assert_eq!(p.counts[1], 30);
        assert_eq!(p.hold_bars[1], 4 - 2 + 1); // 3
    }

    // Test I: Long-to-short crossover in pairs
    #[test]
    fn pairs_long_to_short_crossover() {
        let dt = [
            DAY1_BASE + 9 * HOUR_MS,
            DAY1_BASE + 10 * HOUR_MS,
            DAY1_BASE + 11 * HOUR_MS,
            DAY1_BASE + 12 * HOUR_MS,
        ];
        let weight = [0.0, 0.5, -0.3, 0.0];
        let price = [100.0, 102.0, 104.0, 100.0];

        let (_d, p) = NativeEngine::process_symbol_chunk(
            &dt,
            &weight,
            &price,
            DIGITS_POW10,
            FEE_RATE,
            TimeUnit::Milliseconds,
        );

        // At init: last_volume = 0
        // At i=1: curr_volume = 50. open long 50 at price 102, bar_id=2
        // At i=2: curr_volume = -30.
        //   last_vol=50 >= 0 && curr_vol=-30 <= 0
        //   -> close long 50 at price 104, bar_id=3
        //   -> open short 30 at price 104, bar_id=3
        // At i=3: curr_volume = 0.
        //   last_vol=-30 <= 0 && curr_vol=0 <= 0
        //   diff = 0 - (-30) = 30 > 0 -> close short 30 at price 100, bar_id=4

        assert_eq!(p.dirs.len(), 2, "should produce 2 pairs");

        // Pair 1: long close
        assert_eq!(p.dirs[0], "多头");
        assert_f64_eq(p.open_prices[0], 102.0, "pair1 open");
        assert_f64_eq(p.close_prices[0], 104.0, "pair1 close");
        assert_eq!(p.counts[0], 50);
        assert_eq!(p.hold_bars[0], 3 - 2 + 1); // 2

        // Pair 2: short close
        assert_eq!(p.dirs[1], "空头");
        assert_f64_eq(p.open_prices[1], 104.0, "pair2 open");
        assert_f64_eq(p.close_prices[1], 100.0, "pair2 close");
        assert_eq!(p.counts[1], 30);
        assert_eq!(p.hold_bars[1], 4 - 3 + 1); // 2

        // profit_bp for short: (104 - 100) / 104 * 10000
        let expected_short_bp = ((104.0_f64 - 100.0) / 104.0 * 10000.0 * 100.0).round() / 100.0;
        assert_f64_eq(p.profit_bps[1], expected_short_bp, "pair2 profit_bp");
    }

    // --- PairsSoA::to_dataframe ---
    #[test]
    fn pairs_soa_to_dataframe() {
        let soa = PairsSoA {
            sym_ids: vec![0],
            dirs: vec!["多头"],
            open_dts: vec![1_735_689_600_000],
            close_dts: vec![1_735_776_000_000],
            open_prices: vec![100.0],
            close_prices: vec![105.0],
            hold_bars: vec![10],
            event_seqs: vec!["开多 -> 平多"],
            profit_bps: vec![500.0],
            counts: vec![1],
            time_unit: TimeUnit::Milliseconds,
            symbol_dict: vec!["A".into()],
        };
        let df = soa.to_dataframe().unwrap();
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 11);
        let expected_cols = [
            "symbol",
            "交易方向",
            "开仓时间",
            "平仓时间",
            "开仓价格",
            "平仓价格",
            "持仓K线数",
            "事件序列",
            "盈亏比例",
            "持仓数量",
            "持仓天数",
        ];
        for col in expected_cols {
            assert!(df.column(col).is_ok(), "missing column: {col}");
        }
    }
}
