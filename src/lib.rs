pub mod core;

use std::collections::HashMap;
use std::io::Cursor;
use std::str::FromStr;

use polars::prelude::*;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyDict};
use serde_json::Value;

use crate::core::{WeightBacktest, WeightType};

// ---------------------------------------------------------------------------
// Arrow IPC <-> Polars DataFrame helpers
// ---------------------------------------------------------------------------

fn pyarrow_to_df(data: &[u8]) -> PyResult<DataFrame> {
    let cursor = Cursor::new(data);
    IpcReader::new(cursor)
        .finish()
        .map_err(|e| PyException::new_err(e.to_string()))
}

fn df_to_pyarrow(dataframe: &mut DataFrame) -> PyResult<Vec<u8>> {
    let mut buffer = Cursor::new(Vec::new());
    IpcWriter::new(&mut buffer)
        .finish(dataframe)
        .map_err(|e| PyException::new_err(e.to_string()))?;
    Ok(buffer.into_inner())
}

// ---------------------------------------------------------------------------
// HashMap<String, Value> -> PyDict helper
// ---------------------------------------------------------------------------

fn hashmap_to_pydict<'py>(
    py: Python<'py>,
    map: &HashMap<String, Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        match v {
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    dict.set_item(k, i)?;
                } else if let Some(u) = n.as_u64() {
                    dict.set_item(k, u)?;
                } else if let Some(f) = n.as_f64() {
                    dict.set_item(k, f)?;
                }
            }
            Value::String(s) => {
                dict.set_item(k, s)?;
            }
            _ => {}
        }
    }
    Ok(dict)
}

// ---------------------------------------------------------------------------
// PyWeightBacktest
// ---------------------------------------------------------------------------

#[pyclass(module = "wbt._wbt")]
#[repr(transparent)]
pub struct PyWeightBacktest {
    inner: WeightBacktest,
}

#[pymethods]
impl PyWeightBacktest {
    #[staticmethod]
    #[pyo3(signature = (data, digits=2, fee_rate=Some(0.0002), n_jobs=Some(4), weight_type="ts", yearly_days=252))]
    fn from_arrow<'py>(
        py: Python<'py>,
        data: Bound<'py, PyBytes>,
        digits: i64,
        fee_rate: Option<f64>,
        n_jobs: Option<usize>,
        weight_type: &str,
        yearly_days: usize,
    ) -> PyResult<Self> {
        let data = data.as_bytes();
        let df = pyarrow_to_df(data)?;
        let weight_type = WeightType::from_str(weight_type).unwrap_or(WeightType::TS);

        let mut inner = WeightBacktest::new(df, digits, fee_rate)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        py.allow_threads(|| {
            inner
                .backtest(n_jobs, weight_type, yearly_days)
                .map_err(|e| PyException::new_err(e.to_string()))
        })?;
        Ok(Self { inner })
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let py_dict = PyDict::new(py);

        if let Some(ref report) = self.inner.report {
            let stats = &report.stats;

            let dp = &stats.daily_performance;
            let ep = &stats.evaluate_pairs;
            let pwr = &stats.period_win_rates;

            // 收益
            py_dict.set_item("绝对收益", dp.absolute_return)?;
            py_dict.set_item("年化收益", dp.annual_returns)?;
            py_dict.set_item("夏普比率", dp.sharpe_ratio)?;
            py_dict.set_item("卡玛比率", dp.calmar_ratio)?;
            py_dict.set_item("新高占比", dp.new_high_ratio)?;
            py_dict.set_item("单笔盈亏比", ep.single_profit_loss_ratio)?;
            py_dict.set_item("单笔收益", ep.single_trade_profit)?;
            py_dict.set_item("日胜率", dp.daily_win_rate)?;
            py_dict.set_item("周胜率", pwr.week)?;
            py_dict.set_item("月胜率", pwr.month)?;
            py_dict.set_item("季胜率", pwr.quarter)?;
            py_dict.set_item("年胜率", pwr.year)?;

            // 风险
            py_dict.set_item("最大回撤", dp.max_drawdown)?;
            py_dict.set_item("年化波动率", dp.annual_volatility)?;
            py_dict.set_item("下行波动率", dp.downside_volatility)?;
            py_dict.set_item("新高间隔", dp.new_high_interval)?;

            // 特质
            py_dict.set_item("交易次数", stats.trade_count)?;
            py_dict.set_item("年化交易次数", stats.annual_trade_count)?;
            py_dict.set_item("持仓K线数", ep.position_k_days)?;
            py_dict.set_item("交易胜率", ep.win_rate)?;
            py_dict.set_item("多头占比", stats.long_rate)?;
            py_dict.set_item("空头占比", stats.short_rate)?;
            py_dict.set_item("品种数量", stats.symbols_count)?;

            // 元数据
            py_dict.set_item("开始日期", stats.start_date.to_string())?;
            py_dict.set_item("结束日期", stats.end_date.to_string())?;
        }

        Ok(py_dict)
    }

    #[pyo3(text_signature = "($self)")]
    fn daily_return<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let df = self
            .inner
            .daily_return_df()
            .map_err(|e| PyException::new_err(e.to_string()))?;
        let df_bytes = df_to_pyarrow(df)?;
        Ok(PyBytes::new(py, &df_bytes))
    }

    fn dailys<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let df = self
            .inner
            .dailys_df()
            .map_err(|e| PyException::new_err(e.to_string()))?;
        let df_bytes = df_to_pyarrow(df)?;
        Ok(PyBytes::new(py, &df_bytes))
    }

    fn alpha<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut df = self
            .inner
            .alpha_df()
            .map_err(|e| PyException::new_err(e.to_string()))?;
        let df_bytes = df_to_pyarrow(&mut df)?;
        Ok(PyBytes::new(py, &df_bytes))
    }

    #[pyo3(text_signature = "($self)")]
    fn pairs<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match self
            .inner
            .pairs_df()
            .map_err(|e| PyException::new_err(e.to_string()))?
        {
            Some(df) => {
                let df_bytes = df_to_pyarrow(df)?;
                Ok(PyBytes::new(py, &df_bytes))
            }
            None => Ok(PyBytes::new(py, b"".as_slice())),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, digits=2, fee_rate=Some(0.0002), n_jobs=Some(4), weight_type="ts", yearly_days=252))]
    fn from_file<'py>(
        py: Python<'py>,
        path: &str,
        digits: i64,
        fee_rate: Option<f64>,
        n_jobs: Option<usize>,
        weight_type: &str,
        yearly_days: usize,
    ) -> PyResult<Self> {
        let weight_type_enum = WeightType::from_str(weight_type).unwrap_or(WeightType::TS);
        let mut inner = WeightBacktest::from_file(path, digits, fee_rate)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        py.allow_threads(|| {
            inner
                .backtest(n_jobs, weight_type_enum, yearly_days)
                .map_err(|e| PyException::new_err(e.to_string()))
        })?;
        Ok(Self { inner })
    }

    #[pyo3(text_signature = "($self)")]
    fn symbol_dict(&self) -> PyResult<Vec<String>> {
        if let Some(ref report) = self.inner.report {
            Ok(report.symbol_dict.clone())
        } else {
            Err(PyException::new_err("Report not found"))
        }
    }

    fn long_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(ref report) = self.inner.report {
            hashmap_to_pydict(py, &report.long_stats)
        } else {
            Err(PyException::new_err("Report not found"))
        }
    }

    fn short_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(ref report) = self.inner.report {
            hashmap_to_pydict(py, &report.short_stats)
        } else {
            Err(PyException::new_err("Report not found"))
        }
    }

    #[pyo3(signature = (sdt=None, edt=None, kind="多空"))]
    fn segment_stats<'py>(
        &self,
        py: Python<'py>,
        sdt: Option<i32>,
        edt: Option<i32>,
        kind: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let map = self
            .inner
            .segment_stats(sdt, edt, kind)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        hashmap_to_pydict(py, &map)
    }

    fn long_alpha_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let map = self
            .inner
            .long_alpha_stats()
            .map_err(|e| PyException::new_err(e.to_string()))?;
        hashmap_to_pydict(py, &map)
    }
}

// ---------------------------------------------------------------------------
// daily_performance standalone function
// ---------------------------------------------------------------------------

/// 采用单利计算日收益数据的各项指标
#[pyfunction]
#[pyo3(signature = (daily_returns, yearly_days=None))]
pub fn daily_performance<'py>(
    py: Python<'py>,
    daily_returns: numpy::PyReadonlyArray1<'py, f64>,
    yearly_days: Option<usize>,
) -> PyResult<PyObject> {
    let daily_returns = daily_returns
        .as_slice()
        .map_err(|e| PyException::new_err(e.to_string()))?;
    let dp = crate::core::daily_performance::daily_performance(daily_returns, yearly_days)
        .map_err(|e| PyException::new_err(e.to_string()))?;

    let py_dict = PyDict::new(py);

    py_dict.set_item("绝对收益", dp.absolute_return)?;
    py_dict.set_item("年化", dp.annual_returns)?;
    py_dict.set_item("夏普", dp.sharpe_ratio)?;
    py_dict.set_item("最大回撤", dp.max_drawdown)?;
    py_dict.set_item("卡玛", dp.calmar_ratio)?;
    py_dict.set_item("日胜率", dp.daily_win_rate)?;
    py_dict.set_item("日盈亏比", dp.daily_profit_loss_ratio)?;
    py_dict.set_item("日赢面", dp.daily_win_probability)?;
    py_dict.set_item("年化波动率", dp.annual_volatility)?;
    py_dict.set_item("下行波动率", dp.downside_volatility)?;
    py_dict.set_item("非零覆盖", dp.non_zero_coverage)?;
    py_dict.set_item("盈亏平衡点", dp.break_even_point)?;
    py_dict.set_item("新高间隔", dp.new_high_interval)?;
    py_dict.set_item("新高占比", dp.new_high_ratio)?;
    py_dict.set_item("回撤风险", dp.drawdown_risk)?;
    py_dict.set_item("回归年度回报率", dp.annual_lin_reg_cumsum_return)?;
    py_dict.set_item(
        "长度调整平均最大回撤",
        dp.length_adjusted_average_max_drawdown,
    )?;

    Ok(py_dict.into())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _wbt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWeightBacktest>()?;
    m.add_function(wrap_pyfunction!(daily_performance, m)?)?;
    Ok(())
}
