use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WbtError {
    #[error("expected value for {0}, got None")]
    NoneValue(String),

    #[error("polars: {0}")]
    Polars(#[from] PolarsError),

    #[error("returns should not be empty")]
    ReturnsEmpty,

    #[error("{0:#}")]
    Unexpected(#[from] anyhow::Error),
}
