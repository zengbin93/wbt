use std::collections::HashMap;

use polars::prelude::*;

pub const RISK_COLUMNS: [&str; 7] = [
    "total_risk",
    "long_risk",
    "short_risk",
    "net_exposure",
    "max_single_risk",
    "herfindahl",
    "long_short_ratio",
];

#[derive(Clone, Copy, Default)]
struct Exposure {
    total: f64,
    long: f64,
    short: f64,
    net: f64,
    maximum: f64,
    squares: f64,
}

impl Exposure {
    fn weight(weight: f64) -> Self {
        Self {
            total: weight.abs(),
            long: weight.max(0.0),
            short: (-weight).max(0.0),
            net: weight,
            maximum: weight.abs(),
            squares: weight * weight,
        }
    }

    fn merge(left: Self, right: Self) -> Self {
        Self {
            total: left.total + right.total,
            long: left.long + right.long,
            short: left.short + right.short,
            net: left.net + right.net,
            maximum: left.maximum.max(right.maximum),
            squares: left.squares + right.squares,
        }
    }

    fn values(self) -> [f64; 7] {
        [
            self.total,
            self.long,
            self.short,
            self.net,
            self.maximum,
            self.squares,
            if self.short == 0.0 {
                f64::NAN
            } else {
                self.long / self.short
            },
        ]
    }
}

/// Chronological forward-filled exposure, without materializing a time-symbol grid.
///
/// Semantics: missing weights carry the previous position (initially zero);
/// duplicate `(dt, symbol)` rows **overwrite** in input order (the last
/// non-missing weight wins, they are not accumulated); positions persist
/// overnight and an explicit 0 closes a position; `herfindahl` is the sum of
/// squared weights, not normalized by total exposure.
pub fn calculate_position_risk(frame: &DataFrame) -> PolarsResult<DataFrame> {
    let dt = frame.column("dt")?;
    polars_ensure!(matches!(dt.dtype(), DataType::Datetime(_, _)), InvalidOperation: "dt must be Datetime");
    let timestamps = dt.cast(&DataType::Int64)?;
    let symbols = frame.column("symbol")?.str()?;
    let weights = frame.column("weight")?.strict_cast(&DataType::Float64)?;
    let mut symbol_ids = HashMap::new();
    let mut events = Vec::with_capacity(frame.height());
    for ((timestamp, symbol), weight) in timestamps
        .i64()?
        .into_iter()
        .zip(symbols)
        .zip(weights.f64()?)
    {
        let timestamp =
            timestamp.ok_or_else(|| polars_err!(ComputeError: "dt must not contain nulls"))?;
        let symbol =
            symbol.ok_or_else(|| polars_err!(ComputeError: "symbol must not contain nulls"))?;
        let next_id = symbol_ids.len();
        let symbol_id = *symbol_ids.entry(symbol).or_insert(next_id);
        let weight = weight.unwrap_or(f64::NAN);
        polars_ensure!(!weight.is_infinite(), ComputeError: "weight must not contain infinity");
        events.push((timestamp, symbol_id, weight));
    }
    events.sort_by_key(|event| event.0);
    let leaves = symbol_ids.len().max(1).next_power_of_two();
    let mut tree = vec![Exposure::default(); 2 * leaves];
    let mut output_times = Vec::new();
    let mut output_values = Vec::new();
    for (index, &(timestamp, symbol_id, weight)) in events.iter().enumerate() {
        if !weight.is_nan() {
            let mut node = leaves + symbol_id;
            tree[node] = Exposure::weight(weight);
            while node > 1 {
                node /= 2;
                tree[node] = Exposure::merge(tree[node * 2], tree[node * 2 + 1]);
            }
        }
        if index + 1 == events.len() || events[index + 1].0 != timestamp {
            output_times.push(timestamp);
            output_values.push(tree[1].values());
        }
    }
    let datetime = dt.datetime()?;
    let output_dt = Int64Chunked::from_vec("dt".into(), output_times)
        .into_datetime(datetime.time_unit(), datetime.time_zone().clone())
        .into_series();
    let mut columns = vec![output_dt.into()];
    for (index, name) in RISK_COLUMNS.iter().enumerate() {
        let values: Vec<f64> = output_values.iter().map(|row| row[index]).collect();
        columns.push(Series::new((*name).into(), values).into());
    }
    DataFrame::new(output_values.len(), columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(times: &[i64], symbols: &[&str], weights: &[Option<f64>]) -> DataFrame {
        let dt = Int64Chunked::from_slice("dt".into(), times)
            .into_datetime(TimeUnit::Nanoseconds, None)
            .into_series();
        DataFrame::new(
            times.len(),
            vec![
                dt.into(),
                Series::new("symbol".into(), symbols).into(),
                Series::new("weight".into(), weights).into(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn chronological_hand_calculation() {
        let input = frame(
            &[2, 1, 3],
            &["A", "B", "A"],
            &[Some(0.5), Some(-0.25), Some(0.0)],
        );
        let output = calculate_position_risk(&input).unwrap();
        let expected = [
            [0.25, 0.0, 0.25, -0.25, 0.25, 0.0625, 0.0],
            [0.75, 0.5, 0.25, 0.25, 0.5, 0.3125, 2.0],
            [0.25, 0.0, 0.25, -0.25, 0.25, 0.0625, 0.0],
        ];
        for (column, name) in RISK_COLUMNS.iter().enumerate() {
            let actual = output.column(name).unwrap().f64().unwrap();
            for (row, values) in expected.iter().enumerate() {
                assert_eq!(actual.get(row), Some(values[column]));
            }
        }
    }

    #[test]
    fn null_and_nan_updates_do_not_close_positions() {
        let input = frame(
            &[1, 1, 2, 3],
            &["A"; 4],
            &[Some(2.0), None, Some(f64::NAN), Some(0.0)],
        );
        let output = calculate_position_risk(&input).unwrap();
        assert_eq!(
            output
                .column("total_risk")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![2.0, 2.0, 0.0]
        );
        assert!(
            output
                .column("long_short_ratio")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .all(f64::is_nan)
        );
    }

    #[test]
    fn empty_output_has_all_columns() {
        let output = calculate_position_risk(&frame(&[], &[], &[])).unwrap();
        assert_eq!(output.shape(), (0, 8));
        assert_eq!(
            output.column("dt").unwrap().dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );
    }

    #[test]
    fn invalid_native_inputs_are_errors() {
        let mut input = frame(&[1], &["A"], &[Some(1.0)]);
        input
            .with_column(Series::new("weight".into(), &["invalid"]).into())
            .unwrap();
        assert!(calculate_position_risk(&input).is_err());
        for weight in [f64::INFINITY, f64::NEG_INFINITY] {
            assert!(calculate_position_risk(&frame(&[1], &["A"], &[Some(weight)])).is_err());
        }
        let mut input = frame(&[1], &["A"], &[Some(1.0)]);
        input
            .with_column(Series::new("symbol".into(), &[None::<&str>]).into())
            .unwrap();
        assert!(calculate_position_risk(&input).is_err());
        let mut input = frame(&[1], &["A"], &[Some(1.0)]);
        input
            .with_column(
                Series::new("dt".into(), &[None::<i64>])
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap()
                    .into(),
            )
            .unwrap();
        assert!(calculate_position_risk(&input).is_err());
    }
}
