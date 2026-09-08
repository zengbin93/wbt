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
pub const FORMAT_VERSION: u64 = 2;

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
    if version != Some(1) && version != Some(FORMAT_VERSION) {
        return Err(WbtError::InvalidInput(format!(
            "unsupported format_version {version:?}, expected 1 or {FORMAT_VERSION}"
        )));
    }

    match envelope.get("payload") {
        Some(Value::Object(_)) => {
            if version == Some(FORMAT_VERSION) {
                validate_payload(&envelope["payload"], &envelope["full"])?;
            }
            Ok(envelope["payload"].clone())
        }
        _ => Err(WbtError::InvalidInput(
            "invalid msgpack envelope: missing or malformed payload".into(),
        )),
    }
}

/// v2 validates the structural minimum; v1 payloads retain their legacy shape.
fn validate_payload(payload: &Value, full: &Value) -> Result<(), WbtError> {
    let invalid = |field: &str| WbtError::InvalidInput(format!("invalid payload.{field}"));
    let full = full
        .as_bool()
        .ok_or_else(|| invalid("full: expected boolean"))?;
    for key in ["start_date", "end_date", "weight_type"] {
        if !payload[key].is_string() {
            return Err(invalid(key));
        }
    }
    for key in ["symbol_count", "yearly_days"] {
        if !payload[key].is_i64() && !payload[key].is_u64() {
            return Err(invalid(key));
        }
    }
    for key in ["dates", "year_starts"] {
        if !payload[key].is_array() {
            return Err(invalid(key));
        }
    }
    for key in [
        "curves",
        "return_dist",
        "monthly",
        "symbol_returns",
        "pairs_dist",
        "stats",
        "stats_by_side",
    ] {
        if !payload[key].is_object() {
            return Err(invalid(key));
        }
    }
    for key in [
        "curves_voladj",
        "drawdowns",
        "key_trades",
        "verdict",
        "verdict_recent",
        "yearly_returns",
        "rolling",
        "segment_comparison",
    ] {
        if full {
            let valid = if key == "drawdowns" {
                payload[key].is_array()
            } else {
                payload[key].is_object()
            };
            if !valid {
                return Err(invalid(key));
            }
        } else if payload.get(key).is_some() {
            return Err(invalid("full=False must omit full-only fields"));
        }
    }
    let n = payload["dates"].as_array().unwrap().len();
    for name in ["多空", "多头", "空头", "基准", "超额"] {
        for field in ["daily", "cum", "drawdown"] {
            if payload["curves"][name][field].as_array().map(Vec::len) != Some(n) {
                return Err(invalid(&format!(
                    "curves.{name}.{field}: must align with dates"
                )));
            }
        }
    }
    Ok(())
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
            "format_version": 1,
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

    fn v2_envelope() -> Value {
        rmp_serde::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backtest_result_v2.msgpack"
        )))
        .unwrap()
    }

    #[test]
    fn reads_python_v2_and_validates_schema() {
        let env = v2_envelope();
        assert_eq!(env["format_version"], FORMAT_VERSION);
        assert_eq!(env["full"], true);
        let payload = decode_wire(&pack(&env)).unwrap();
        assert_eq!(payload["symbol_count"], 1);
        assert!(payload.get("rolling").is_some());
        for key in ["stats", "dates", "curves", "rolling"] {
            let mut bad = env.clone();
            bad["payload"].as_object_mut().unwrap().remove(key);
            assert!(decode_wire(&pack(&bad)).is_err(), "missing {key}");
        }
        let mut bad = env.clone();
        bad["payload"]["curves"]["多空"]["daily"] = serde_json::json!([]);
        assert!(decode_wire(&pack(&bad)).is_err());
        for full in [Value::Null, Value::from(1), Value::from(false)] {
            let mut bad = env.clone();
            bad["full"] = full;
            assert!(decode_wire(&pack(&bad)).is_err());
        }
        let mut future = env;
        future["payload"]["future"] = serde_json::json!([1, null]);
        assert_eq!(
            decode_wire(&pack(&future)).unwrap()["future"],
            serde_json::json!([1, null])
        );
    }

    #[test]
    fn reads_compact_v2_payload() {
        let mut env = v2_envelope();
        env["full"] = Value::Bool(false);
        for key in [
            "curves_voladj",
            "drawdowns",
            "key_trades",
            "verdict",
            "verdict_recent",
            "yearly_returns",
            "rolling",
            "segment_comparison",
        ] {
            env["payload"].as_object_mut().unwrap().remove(key);
        }
        assert!(decode_wire(&pack(&env)).is_ok());
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
