use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WbtError {
    #[error("expected value for {0}, got None")]
    NoneValue(String),

    #[error("io: {0}")]
    Io(String),

    #[error("polars: {0}")]
    Polars(#[from] PolarsError),

    #[error("returns should not be empty")]
    ReturnsEmpty,

    #[error("{0:#}")]
    Unexpected(#[from] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_value_display() {
        let err = WbtError::NoneValue("test_field".into());
        assert_eq!(err.to_string(), "expected value for test_field, got None");
    }

    #[test]
    fn returns_empty_display() {
        let err = WbtError::ReturnsEmpty;
        assert_eq!(err.to_string(), "returns should not be empty");
    }

    #[test]
    fn from_anyhow() {
        let anyhow_err = anyhow::anyhow!("something went wrong");
        let wbt_err: WbtError = anyhow_err.into();
        assert!(wbt_err.to_string().contains("something went wrong"));
    }
}
