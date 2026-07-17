//! BacktestResult MessagePack 交换格式的读取（Rust 侧）。
//!
//! 与 Python `wbt.serialization` 对应：解析封装头（`format` / `format_version`），
//! 校验通过后返回 `payload`（完整嵌套结果对象）。第一版用 `serde_json::Value` 兼容整个
//! payload，不维护巨大的强类型 struct；后续若需直接消费 `curves` / `rolling` 等热点字段，
//! 再逐步强类型化。
//!
//! 定位：完整嵌套结果对象的交换格式，不替代 Arrow IPC / Parquet 处理列式表格热数据。

use crate::core::errors::WbtError;
use serde_json::Value;
use std::path::Path;

/// 封装格式标识，须与 Python 端 `wbt.serialization.FORMAT` 一致。
pub const FORMAT: &str = "wbt.backtest_result";
/// 当前支持的封装版本，须与 Python 端 `wbt.serialization.FORMAT_VERSION` 一致。
pub const FORMAT_VERSION: u64 = 1;

/// 解码 MessagePack 字节，校验封装头后返回 `payload`。
///
/// `format` 不匹配或 `format_version` 未知均返回 [`WbtError::InvalidInput`]。
pub fn decode_wire(bytes: &[u8]) -> Result<Value, WbtError> {
    let envelope: Value = rmp_serde::from_slice(bytes)
        .map_err(|e| WbtError::InvalidInput(format!("failed to decode msgpack: {e}")))?;

    let format = envelope.get("format").and_then(Value::as_str);
    if format != Some(FORMAT) {
        return Err(WbtError::InvalidInput(format!(
            "unexpected format {format:?}, expected {FORMAT:?}"
        )));
    }

    let version = envelope.get("format_version").and_then(Value::as_u64);
    if version != Some(FORMAT_VERSION) {
        return Err(WbtError::InvalidInput(format!(
            "unsupported format_version {version:?}, expected {FORMAT_VERSION}"
        )));
    }

    match envelope.get("payload") {
        Some(Value::Object(_)) => Ok(envelope["payload"].clone()),
        _ => Err(WbtError::InvalidInput(
            "invalid msgpack envelope: missing or malformed payload".into(),
        )),
    }
}

/// 从文件读取并解码 `.msgpack`，返回 `payload`。
pub fn load_wire(path: impl AsRef<Path>) -> Result<Value, WbtError> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| WbtError::Io(format!("read {}: {e}", path.as_ref().display())))?;
    decode_wire(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用 rmp-serde 编一个合法 envelope，方便单测复用。
    fn pack(value: &Value) -> Vec<u8> {
        rmp_serde::to_vec_named(value).unwrap()
    }

    fn valid_envelope() -> Value {
        serde_json::json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "payload": {
                "symbol_count": 2,
                "dates": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
                "curves": {"多空": {"daily": [0.0, 0.1]}},
            },
        })
    }

    #[test]
    fn decode_valid_returns_payload() {
        let payload = decode_wire(&pack(&valid_envelope())).unwrap();
        assert_eq!(payload["symbol_count"].as_u64(), Some(2));
        assert_eq!(payload["dates"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn decode_rejects_wrong_format() {
        let mut env = valid_envelope();
        env["format"] = Value::String("something.else".into());
        let err = decode_wire(&pack(&env)).unwrap_err();
        assert!(err.to_string().contains("unexpected format"));
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let mut env = valid_envelope();
        env["format_version"] = Value::from(999u64);
        let err = decode_wire(&pack(&env)).unwrap_err();
        assert!(err.to_string().contains("unsupported format_version"));
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(decode_wire(&[0xc1, 0x00, 0xff]).is_err());
    }

    /// 读取 Python 生成的 fixture，校验关键字段：symbol_count、dates 长度、curves keys。
    #[test]
    fn reads_python_fixture() {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backtest_result.msgpack"
        ));
        let payload = decode_wire(bytes).unwrap();

        let symbol_count = payload["symbol_count"].as_u64().unwrap();
        assert!(symbol_count >= 1, "symbol_count should be positive");

        let dates = payload["dates"].as_array().unwrap();
        assert!(!dates.is_empty(), "dates should not be empty");

        let curves = payload["curves"].as_object().unwrap();
        for key in ["多空", "多头", "空头", "基准", "超额"] {
            assert!(curves.contains_key(key), "curves missing key {key}");
        }
    }
}
