use serde::Serialize;
use strum_macros::{AsRefStr, Display, EnumString};

pub static OP_OPEN_LONG_CLOSE_SHORT: &str = "开多 -> 平空";
pub static OP_OPEN_LONG_CLOSE_LONG: &str = "开多 -> 平多";
pub static OP_OPEN_SHORT_CLOSE_SHORT: &str = "开空 -> 平空";
pub static OP_OPEN_SHORT_CLOSE_LONG: &str = "开空 -> 平多";
pub static OP_UNKNOWN: &str = "未知操作";

#[derive(Debug, Clone, Copy, PartialEq, EnumString, AsRefStr, Display, Serialize)]
pub enum TradeDir {
    #[strum(serialize = "多头")]
    Long,
    #[strum(serialize = "空头")]
    Short,
    #[strum(serialize = "多空")]
    LongShort,
}

#[derive(Debug, Clone, Copy, PartialEq, EnumString, AsRefStr, Display)]
pub enum TradeAction {
    #[strum(serialize = "开空")]
    OpenShort,
    #[strum(serialize = "开多")]
    OpenLong,
    #[strum(serialize = "平空")]
    CloseShort,
    #[strum(serialize = "平多")]
    CloseLong,
}

impl TradeAction {
    pub fn first_create(vol: i64) -> Option<Self> {
        match vol {
            vol if vol > 0 => Some(Self::OpenLong),
            vol if vol < 0 => Some(Self::OpenShort),
            _ => None,
        }
    }

    pub fn get_event_seq(&self, op: Self) -> &'static str {
        match (*self, op) {
            (TradeAction::OpenLong, TradeAction::CloseShort) => OP_OPEN_LONG_CLOSE_SHORT,
            (TradeAction::OpenLong, TradeAction::CloseLong) => OP_OPEN_LONG_CLOSE_LONG,
            (TradeAction::OpenShort, TradeAction::CloseShort) => OP_OPEN_SHORT_CLOSE_SHORT,
            (TradeAction::OpenShort, TradeAction::CloseLong) => OP_OPEN_SHORT_CLOSE_LONG,
            _ => OP_UNKNOWN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_create_positive_volume() {
        let action = TradeAction::first_create(5);
        assert_eq!(action, Some(TradeAction::OpenLong));
    }

    #[test]
    fn test_first_create_negative_volume() {
        let action = TradeAction::first_create(-3);
        assert_eq!(action, Some(TradeAction::OpenShort));
    }

    #[test]
    fn test_first_create_zero_volume() {
        let action = TradeAction::first_create(0);
        assert_eq!(action, None);
    }
}
