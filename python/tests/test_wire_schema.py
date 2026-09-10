import json

import msgpack
import pytest

import wbt
from wbt.metrics import METRIC_LABELS, lookup_metric, to_machine_metrics
from wbt.serialization import FORMAT, FULL_FIELDS, _unwrap


@pytest.mark.parametrize("name", ["to_json", "to_msgpack", "dump_json", "dump_msgpack"])
@pytest.mark.parametrize("module_helper", [False, True])
@pytest.mark.parametrize("full", [None, True, False], ids=["default", "full", "compact"])
def test_export_defaults_are_full_and_compact_is_explicit(wb, tmp_path, name, module_helper, full):
    result = wb.to_result()
    export = getattr(wbt, name) if module_helper else getattr(result, name)
    args = [result] if module_helper else []
    kwargs = {} if full is None else {"full": full}
    if name.startswith("dump_"):
        path = tmp_path / "result"
        export(*args, path, **kwargs)
        raw = path.read_bytes()
    else:
        raw = export(*args, **kwargs)
    env = json.loads(raw) if name.endswith("json") else msgpack.unpackb(raw, raw=False)
    expected_full = full is not False
    assert env["full"] is expected_full
    assert env["format_version"] == 2
    assert env["payload"] == result.to_dict(full=expected_full)
    if expected_full:
        assert FULL_FIELDS.keys() <= env["payload"].keys()
    else:
        assert not FULL_FIELDS.keys() & env["payload"].keys()
        assert not FULL_FIELDS.keys() & result.__dict__.keys()


def test_to_dict_default_remains_compact(wb):
    result = wb.to_result()
    assert result.to_dict() == result.to_dict(full=False)
    assert not FULL_FIELDS.keys() & result.to_dict().keys()
    assert not FULL_FIELDS.keys() & result.__dict__.keys()


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
