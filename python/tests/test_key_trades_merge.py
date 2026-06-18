"""验证 key_trades 的「真实持仓段合并去重」（任务 t101322）。

main 原本按 (symbol, 开仓时间, 平仓时间) 精确聚合选榜，无法合并 LIFO 撮合对同一段
持仓的拆分（同开分批平 / 分批开同平）。本次改为选榜前先用并查集把 open 或 close 任一
相同的聚合记录传递合并成真实持仓段。

这里用一份完全独立的纯 Python 实现复刻 aggregate→merge→select 链，对随机多场景与
Rust 的 wb.key_trades 输出逐行比对，校验数值正确性；并验证去重的关键不变量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wbt import WeightBacktest


def _uf_find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[ra] = rb


def _ref_merge(agg: pd.DataFrame) -> list[tuple]:
    """独立实现：把 aggregated_pairs 按真实持仓段合并。

    返回段集合 (symbol, dir, open, close, profit, hold, count)。
    """
    segs: list[tuple] = []
    for (sym, direction), grp in agg.groupby(["symbol", "交易方向"]):
        g = grp.reset_index(drop=True)
        n = len(g)
        parent = list(range(n))
        o_seen: dict = {}
        c_seen: dict = {}
        for i in range(n):
            o, c = g.loc[i, "开仓时间"], g.loc[i, "平仓时间"]
            if o in o_seen:
                _uf_union(parent, i, o_seen[o])
            else:
                o_seen[o] = i
            if c in c_seen:
                _uf_union(parent, i, c_seen[c])
            else:
                c_seen[c] = i
        comp: dict[int, list[int]] = {}
        for i in range(n):
            comp.setdefault(_uf_find(parent, i), []).append(i)
        for members in comp.values():
            rep = min(members, key=lambda i: (-int(g.loc[i, "持仓K线数"]), g.loc[i, "开仓时间"]))
            cnt = sum(int(g.loc[i, "count"]) for i in members)
            segs.append(
                (
                    sym,
                    direction,
                    g.loc[rep, "开仓时间"],
                    g.loc[rep, "平仓时间"],
                    round(float(g.loc[rep, "盈亏比例"]), 4),
                    int(g.loc[rep, "持仓K线数"]),
                    cnt,
                )
            )
    return sorted(segs)


def _kt_segs(kt: pd.DataFrame) -> list[tuple]:
    """key_trades（去 year/kind 去重）→ 段集合。"""
    segs = {
        (
            r["symbol"],
            r["交易方向"],
            r["开仓时间"],
            r["平仓时间"],
            round(float(r["盈亏比例"]), 4),
            int(r["持仓K线数"]),
            int(r["count"]),
        )
        for _, r in kt.iterrows()
    }
    return sorted(segs)


def _ref_board(segs: list[tuple], top: int) -> dict:
    """复刻 select_key_trades：按平仓年份 best 降序 / worst 升序，worst 剔除已入 best。"""
    by_year: dict[int, list[tuple]] = {}
    for s in segs:
        by_year.setdefault(pd.Timestamp(s[3]).year, []).append(s)
    out: dict = {}
    for y, items in by_year.items():
        best = sorted(items, key=lambda s: (-s[4], s[2]))[:top]
        best_set = set(best)
        worst = [s for s in sorted(items, key=lambda s: (s[4], s[2])) if s not in best_set][:top]
        # Rust 端空的 best/worst 不产生行，参照也应跳过空列表，保证 dict 键集合一致。
        if best:
            out[(y, "best")] = sorted(round(s[4], 4) for s in best)
        if worst:
            out[(y, "worst")] = sorted(round(s[4], 4) for s in worst)
    return out


def _kt_board(kt: pd.DataFrame) -> dict:
    out: dict = {}
    for (y, kind), g in kt.groupby(["year", "kind"]):
        out[(int(y), kind)] = sorted(round(float(x), 4) for x in g["盈亏比例"])
    return out


def _make_bt(rng: np.random.Generator) -> WeightBacktest:
    rows = []
    for sym in ["AAA", "BBB", "CCC"][: rng.integers(1, 4)]:
        year = int(rng.choice([2022, 2023, 2024]))
        for dd in range(int(rng.integers(6, 40))):
            w = round(float(rng.choice([-0.5, -0.25, 0.0, 0.0, 0.25, 0.5])), 2)
            rows.append(
                {
                    "dt": f"{year}-{1 + dd // 28:02d}-{1 + dd % 28:02d} 09:30:00",
                    "symbol": sym,
                    "weight": w,
                    "price": round(100.0 + float(rng.normal(0, 3)), 4),
                }
            )
    return WeightBacktest(pd.DataFrame(rows), digits=2, n_jobs=1)


def test_key_trades_merge_matches_reference() -> None:
    """随机多场景：合并段集合与年度选榜均须与独立参照逐行一致。"""
    rng = np.random.default_rng(20240618)
    checked = 0
    for _ in range(60):
        bt = _make_bt(rng)
        agg = bt.aggregated_pairs
        if agg.empty:
            continue
        checked += 1
        top = int(rng.integers(1, 5))
        ref_segs = _ref_merge(agg)
        assert ref_segs == _kt_segs(bt.key_trades(top=10**9)), "合并段集合与参照不一致"
        assert _ref_board(ref_segs, top) == _kt_board(bt.key_trades(top=top)), "年度选榜与参照不一致"
    assert checked >= 15, f"有效随机场景过少（{checked}）"


def test_key_trades_dedups_lifo_split() -> None:
    """同 open 分批平仓（LIFO 拆成多条聚合记录）在 key_trades 里只上榜一次。"""
    # 单 symbol，建仓后分两批平仓 → aggregated_pairs 多条同 open、不同 close。
    rows = []
    for i, w in enumerate([0.5, 0.5, 0.25, 0.0]):
        rows.append({"dt": f"2024-03-{i + 1:02d} 09:30:00", "symbol": "AAA", "weight": w, "price": 100.0 + i})
    bt = WeightBacktest(pd.DataFrame(rows), digits=2, n_jobs=1)
    agg = bt.aggregated_pairs
    kt = bt.key_trades(top=10)
    seg = kt.drop(columns=["year", "kind"]).drop_duplicates()
    # 合并只会减少或持平：真实持仓段数 <= 聚合记录数。
    assert len(seg) <= len(agg)
    # 不变量：同 (symbol, 交易方向) 的开仓/平仓时间各自唯一。
    for _, gp in seg.groupby(["symbol", "交易方向"]):
        assert gp["开仓时间"].is_unique
        assert gp["平仓时间"].is_unique
