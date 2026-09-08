import json

import msgpack
import pytest

from wbt import load_json, load_msgpack, to_json, to_msgpack
from wbt.metrics import METRIC_LABELS, lookup_metric, to_machine_metrics
from wbt.serialization import FORMAT, FULL_FIELDS, _unwrap


def test_export_defaults_are_compact_and_symmetric(wb, tmp_path):
    result = wb.to_result()
    expected = result.to_dict()
    for encode in (result.to_json, result.to_msgpack):
        raw = encode()
        env = json.loads(raw) if raw.startswith(b"{") else msgpack.unpackb(raw, raw=False)
        assert env["full"] is False
        assert env["payload"] == expected
    assert json.loads(to_json(result))["payload"] == expected
    assert msgpack.unpackb(to_msgpack(result), raw=False)["payload"] == expected
    for dump, load, suffix in ((result.dump_json, load_json, "json"), (result.dump_msgpack, load_msgpack, "msgpack")):
        path = tmp_path / f"result.{suffix}"
        dump(path)
        assert load(path) == expected
    assert not FULL_FIELDS.keys() & expected.keys()


@pytest.mark.parametrize("full", [False, True])
def test_v2_payload_validation_and_unknown_fields(wb, full):
    env = json.loads(wb.to_result().to_json(full=full))
    env["payload"]["future_field"] = [1, None]
    assert _unwrap(env)["future_field"] == [1, None]
    for key in ("stats", "dates", "curves"):
        bad = json.loads(json.dumps(env))
        del bad["payload"][key]
        with pytest.raises(ValueError, match=f"payload.{key}"):
            _unwrap(bad)
    env["payload"]["curves"]["多空"]["daily"].append(0)
    with pytest.raises(ValueError, match="align with dates"):
        _unwrap(env)


def test_v1_migration_preserves_legacy_payload():
    legacy = {"custom": True}
    assert _unwrap({"format": FORMAT, "format_version": 1, "payload": legacy}) == legacy
    for version in (True, 2.0, 999):
        with pytest.raises(ValueError, match="format_version"):
            _unwrap({"format": FORMAT, "format_version": version, "payload": legacy})


def test_full_metadata_must_match_payload(wb):
    env = json.loads(wb.to_result().to_json(full=True))
    for full in (None, 1, "true"):
        with pytest.raises(ValueError, match="full"):
            _unwrap({**env, "full": full})
    with pytest.raises(ValueError, match="full=False"):
        _unwrap({**env, "full": False})
    del env["payload"]["rolling"]
    with pytest.raises(ValueError, match="rolling"):
        _unwrap(env)


def test_machine_names_and_legacy_labels_share_one_mapping():
    from wbt.plotting.tables import _lookup_metric
    from wbt.report._html_tables import _mget

    for values in (
        {"年化": 0.1, "夏普": 2},
        {"年化收益": 0.1, "夏普比率": 2},
        {"annual_returns": 0.1, "sharpe_ratio": 2},
    ):
        assert to_machine_metrics(values) == {"annual_returns": 0.1, "sharpe_ratio": 2}
        for lookup in (lookup_metric, _lookup_metric, _mget):
            assert lookup(values, "年化收益") == 0.1
            assert lookup(values, "夏普比率") == 2
    assert METRIC_LABELS["annual_returns"] == "年化收益"
    assert to_machine_metrics({"annual_returns": None, "年化": 0.2, "custom": 1}) == {
        "annual_returns": None,
        "custom": 1,
    }
