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

    // --- TradeDir Display/FromStr ---
    #[test]
    fn trade_dir_display_from_str() {
        assert_eq!(TradeDir::Long.to_string(), "多头");
        assert_eq!(TradeDir::Short.to_string(), "空头");
        assert_eq!(TradeDir::LongShort.to_string(), "多空");
        assert_eq!("多头".parse::<TradeDir>().unwrap(), TradeDir::Long);
        assert_eq!("空头".parse::<TradeDir>().unwrap(), TradeDir::Short);
        assert_eq!("多空".parse::<TradeDir>().unwrap(), TradeDir::LongShort);
    }

    // --- TradeAction Display/FromStr ---
    #[test]
    fn trade_action_display_from_str() {
        assert_eq!(TradeAction::OpenLong.to_string(), "开多");
        assert_eq!(TradeAction::OpenShort.to_string(), "开空");
        assert_eq!(TradeAction::CloseLong.to_string(), "平多");
        assert_eq!(TradeAction::CloseShort.to_string(), "平空");
        assert_eq!(
            "开多".parse::<TradeAction>().unwrap(),
            TradeAction::OpenLong
        );
        assert_eq!(
            "开空".parse::<TradeAction>().unwrap(),
            TradeAction::OpenShort
        );
        assert_eq!(
            "平多".parse::<TradeAction>().unwrap(),
            TradeAction::CloseLong
        );
        assert_eq!(
            "平空".parse::<TradeAction>().unwrap(),
            TradeAction::CloseShort
        );
    }

    // --- first_create edge cases ---
    #[test]
    fn first_create_large_volumes() {
        assert_eq!(
            TradeAction::first_create(i64::MAX),
            Some(TradeAction::OpenLong)
        );
        assert_eq!(
            TradeAction::first_create(i64::MIN),
            Some(TradeAction::OpenShort)
        );
    }

    // --- get_event_seq ---
    #[test]
    fn get_event_seq_valid_combos() {
        assert_eq!(
            TradeAction::OpenLong.get_event_seq(TradeAction::CloseShort),
            OP_OPEN_LONG_CLOSE_SHORT
        );
        assert_eq!(
            TradeAction::OpenLong.get_event_seq(TradeAction::CloseLong),
            OP_OPEN_LONG_CLOSE_LONG
        );
        assert_eq!(
            TradeAction::OpenShort.get_event_seq(TradeAction::CloseShort),
            OP_OPEN_SHORT_CLOSE_SHORT
        );
        assert_eq!(
            TradeAction::OpenShort.get_event_seq(TradeAction::CloseLong),
            OP_OPEN_SHORT_CLOSE_LONG
        );
    }

    #[test]
    fn get_event_seq_invalid_combos() {
        assert_eq!(
            TradeAction::CloseLong.get_event_seq(TradeAction::OpenLong),
            OP_UNKNOWN
        );
        assert_eq!(
            TradeAction::OpenLong.get_event_seq(TradeAction::OpenLong),
            OP_UNKNOWN
        );
    }
}
