use crate::WeightBacktest;
use crate::errors::WbtError;
use crate::trade_dir::{TradeAction, TradeDir};
use anyhow::Context;
use polars::prelude::*;

#[derive(Debug, Clone, Copy)]
struct Operate {
    volume: i64,
    datetime: i64,
    bar_id: i64,
    price: f64,
    action: TradeAction,
}

impl WeightBacktest {
    pub(crate) fn calc_all_dailys(
        dfw: &DataFrame,
        fee_rate: f64,
    ) -> Result<DataFrame, WbtError> {
        let dfs = dfw
            .clone()
            .lazy()
            .with_column(
                (col("price").shift(lit(-1)).over(["symbol"]) / col("price") - lit(1)).alias("n1b"),
            )
            .with_column((col("weight") * col("n1b")).alias("edge"))
            .with_column(
                (col("weight").shift(lit(1)).over(["symbol"]) - col("weight"))
                    .abs()
                    .alias("turnover"),
            )
            .with_column((col("turnover") * lit(fee_rate)).alias("cost"))
            .with_column((col("edge") - col("cost")).alias("return"))
            .with_column(
                when(col("weight").gt(lit(0)))
                    .then(col("weight"))
                    .otherwise(lit(0))
                    .alias("long_weight"),
            )
            .with_column(
                when(col("weight").lt(lit(0)))
                    .then(col("weight"))
                    .otherwise(lit(0))
                    .alias("short_weight"),
            )
            .with_column((col("long_weight") * col("n1b")).alias("long_edge"))
            .with_column((col("short_weight") * col("n1b")).alias("short_edge"))
            .with_column(
                (col("long_weight").shift(lit(1)).over(["symbol"]) - col("long_weight"))
                    .abs()
                    .alias("long_turnover"),
            )
            .with_column(
                (col("short_weight").shift(lit(1)).over(["symbol"]) - col("short_weight"))
                    .abs()
                    .alias("short_turnover"),
            )
            .with_column((col("long_turnover") * lit(fee_rate)).alias("long_cost"))
            .with_column((col("short_turnover") * lit(fee_rate)).alias("short_cost"))
            .with_column((col("long_edge") - col("long_cost")).alias("long_return"))
            .with_column((col("short_edge") - col("short_cost")).alias("short_return"))
            .with_column(col("dt").cast(DataType::Date).alias("date"))
            .group_by([col("date"), col("symbol")])
            .agg([
                col("n1b").sum(),
                col("edge").sum(),
                col("return").sum(),
                col("cost").sum(),
                col("turnover").sum(),
                col("long_edge").sum(),
                col("short_edge").sum(),
                col("long_cost").sum(),
                col("short_cost").sum(),
                col("long_turnover").sum(),
                col("short_turnover").sum(),
                col("long_return").sum(),
                col("short_return").sum(),
            ])
            .collect()
            .context("Failed to calculate global dailys dataframe")?;
        Ok(dfs)
    }

    /// 计算某个合约的交易对
    pub(crate) fn calc_symbol_pairs(
        symbol_df: DataFrame,
        symbol: &str,
        digits: i64,
    ) -> Result<LazyFrame, WbtError> {
        let dfs = symbol_df
            .lazy()
            .with_column(
                (col("weight") * lit(digits))
                    .cast(DataType::Int64)
                    .alias("volume"),
            )
            .with_column((col("volume").shift(lit(-1)) - col("volume")).alias("operation"))
            .collect()
            .context("Failed to filter symbol")?;

        fn add_operate(operates: &mut Vec<Operate>, mut op: Operate) {
            op.volume = op.volume.abs();
            operates.push(op);
        }

        let mut operates: Vec<Operate> = vec![];

        // first
        let initial_volume = dfs
            .column("volume")?.as_materialized_series()
            .i64()?
            .get(0)
            .ok_or(WbtError::NoneValue("volume column".to_string()))?;

        let initial_datetime = dfs
            .column("dt")?.as_materialized_series()
            .datetime()?
            .get(0)
            .ok_or(WbtError::NoneValue("dt column".to_string()))?;

        let initial_price = dfs
            .column("price")?.as_materialized_series()
            .f64()?
            .get(0)
            .ok_or(WbtError::NoneValue("price column".to_string()))?;

        if let Some(action) = TradeAction::first_create(initial_volume) {
            add_operate(
                &mut operates,
                Operate {
                    volume: initial_volume,
                    datetime: initial_datetime,
                    bar_id: 1,
                    price: initial_price,
                    action,
                },
            );
        }

        // 3列同时按行遍历
        let volumes = dfs.column("volume")?.as_materialized_series().i64()?.into_iter();
        let datetimes = dfs.column("dt")?.as_materialized_series().datetime()?.into_iter();
        let prices = dfs.column("price")?.as_materialized_series().f64()?.into_iter();

        let combined_iter = volumes
            .zip(datetimes)
            .zip(prices)
            .map(|((volume, datetime), price)| (volume, datetime, price));

        let mut last_volume = initial_volume;

        let mut bar_id = 1;
        for (volume, datetime, price) in combined_iter.skip(1) {
            let volume =
                volume.ok_or(WbtError::NoneValue("volume column".to_string()))?;
            let datetime =
                datetime.ok_or(WbtError::NoneValue("dt column".to_string()))?;
            let price = price.ok_or(WbtError::NoneValue("price column".to_string()))?;

            bar_id += 1;

            match (last_volume, volume) {
                (v, v2) if v >= 0 && v2 >= 0 && v2 > v => {
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v2 - v,
                            action: TradeAction::OpenLong,
                        },
                    );
                }
                (v, v2) if v >= 0 && v2 >= 0 && v2 < v => {
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v - v2,
                            action: TradeAction::CloseLong,
                        },
                    );
                }
                (v, v2) if v <= 0 && v2 <= 0 && v2 > v => {
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v - v2,
                            action: TradeAction::CloseShort,
                        },
                    );
                }
                (v, v2) if v <= 0 && v2 <= 0 && v2 < v => {
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v - v2,
                            action: TradeAction::OpenShort,
                        },
                    );
                }
                (v, v2) if v >= 0 && v2 <= 0 => {
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v,
                            action: TradeAction::CloseLong,
                        },
                    );
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v2,
                            action: TradeAction::OpenShort,
                        },
                    );
                }
                (v, v2) if v <= 0 && v2 >= 0 => {
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v,
                            action: TradeAction::CloseShort,
                        },
                    );
                    add_operate(
                        &mut operates,
                        Operate {
                            datetime,
                            bar_id,
                            price,
                            volume: v2,
                            action: TradeAction::OpenLong,
                        },
                    );
                }
                _ => {}
            };
            last_volume = volume;
        }

        let mut opens: Vec<Operate> = vec![];

        let total_close_vol: i64 = operates
            .iter()
            .filter(|o| matches!(o.action, TradeAction::CloseShort | TradeAction::CloseLong))
            .map(|o| o.volume)
            .sum();

        let mut results: Vec<(TradeDir, i64, i64, f64, f64, i64, &str, f64)> =
            Vec::with_capacity(total_close_vol as usize);

        for op in operates {
            match op.action {
                TradeAction::OpenShort | TradeAction::OpenLong => {
                    opens.push(op);
                    continue;
                }
                _ => {}
            };

            if opens.is_empty() {
                continue;
            }

            let mut remaining_close_vol = op.volume;

            while remaining_close_vol > 0 {
                if opens.is_empty() {
                    break;
                }

                let mut open_op = opens.pop().unwrap();
                let match_vol = remaining_close_vol.min(open_op.volume);

                let (p_dir, p_ret) = match open_op.action {
                    TradeAction::OpenShort => (
                        TradeDir::Short,
                        ((open_op.price - op.price) / open_op.price * 10000_f64 * 100_f64).round()
                            / 100_f64,
                    ),
                    TradeAction::OpenLong => (
                        TradeDir::Long,
                        ((op.price - open_op.price) / open_op.price * 10000_f64 * 100_f64).round()
                            / 100_f64,
                    ),
                    _ => unreachable!(
                        "open_op.action 只可能是 OpenLong 或 OpenShort，其它值不应出现"
                    ),
                };

                let close_bar_id_diff = op.bar_id - open_op.bar_id + 1;
                let event_seq = open_op.action.get_event_seq(op.action);

                for _ in 0..match_vol {
                    results.push((
                        p_dir,
                        open_op.datetime,
                        op.datetime,
                        open_op.price,
                        op.price,
                        close_bar_id_diff,
                        event_seq,
                        p_ret,
                    ));
                }

                open_op.volume -= match_vol;
                remaining_close_vol -= match_vol;

                if open_op.volume > 0 {
                    opens.push(open_op);
                }
            }
        }

        let dfs = DataFrame::new(vec![
            Series::new(
                "交易方向".into(),
                results.iter().map(|x| x.0.as_ref()).collect::<Vec<_>>(),
            ).into_column(),
            Series::new("开仓时间".into(), results.iter().map(|x| x.1).collect::<Vec<_>>()).into_column(),
            Series::new("平仓时间".into(), results.iter().map(|x| x.2).collect::<Vec<_>>()).into_column(),
            Series::new("开仓价格".into(), results.iter().map(|x| x.3).collect::<Vec<_>>()).into_column(),
            Series::new("平仓价格".into(), results.iter().map(|x| x.4).collect::<Vec<_>>()).into_column(),
            Series::new("持仓K线数".into(), results.iter().map(|x| x.5).collect::<Vec<_>>()).into_column(),
            Series::new("事件序列".into(), results.iter().map(|x| x.6).collect::<Vec<_>>()).into_column(),
            Series::new("盈亏比例".into(), results.iter().map(|x| x.7).collect::<Vec<_>>()).into_column(),
        ])?
        .lazy()
        .with_columns([
            col("开仓时间").cast(DataType::Datetime(TimeUnit::Nanoseconds, None)),
            col("平仓时间").cast(DataType::Datetime(TimeUnit::Nanoseconds, None)),
        ])
        .with_column(lit(symbol).alias("symbol"));

        Ok(dfs)
    }

    pub(crate) fn get_symbol_str_from_a_symbol_df(
        symbol_df: &DataFrame,
    ) -> Result<&str, WbtError> {
        let symbol = symbol_df
            .column("symbol")
            .context("Column 'symbol' not found in DataFrame")?
            .as_materialized_series()
            .str()
            .context("Column 'symbol' is not of string type")?
            .get(0)
            .context("Failed to get the first element from 'symbol' column")?;
        Ok(symbol)
    }
}
