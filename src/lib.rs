use std::io::Cursor;
use std::str::FromStr;

use polars::prelude::*;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyDict};

use wbt_core::{WeightBacktest, WeightType};

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
            py_dict.set_item("开始日期", stats.start_date.to_string())?;
            py_dict.set_item("结束日期", stats.end_date.to_string())?;
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
            py_dict.set_item("交易胜率", ep.win_rate)?;
            py_dict.set_item("单笔收益", ep.single_trade_profit)?;
            py_dict.set_item("持仓K线数", ep.position_k_days)?;
            py_dict.set_item("多头占比", stats.long_rate)?;
            py_dict.set_item("空头占比", stats.short_rate)?;
            py_dict.set_item("与基准相关性", stats.relevance)?;
            py_dict.set_item("与基准空头相关性", stats.relevance_short)?;
            py_dict.set_item("波动比", stats.volatility_ratio)?;
            py_dict.set_item("与基准波动相关性", stats.relevance_volatility)?;
            py_dict.set_item("品种数量", stats.symbols_count)?;
        }

        Ok(py_dict)
    }

    #[pyo3(text_signature = "($self)")]
    fn daily_return<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if let Some(ref mut report) = self.inner.report {
            let df_bytes = df_to_pyarrow(&mut report.daily_return)?;
            Ok(PyBytes::new(py, &df_bytes))
        } else {
            Err(PyException::new_err("Report not found"))
        }
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

    #[pyo3(text_signature = "($self)")]
    fn symbol_dict(&self) -> PyResult<Vec<String>> {
        if let Some(ref report) = self.inner.report {
            Ok(report.symbol_dict.clone())
        } else {
            Err(PyException::new_err("Report not found"))
        }
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
    let dp = wbt_core::daily_performance::daily_performance(daily_returns, yearly_days)
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
