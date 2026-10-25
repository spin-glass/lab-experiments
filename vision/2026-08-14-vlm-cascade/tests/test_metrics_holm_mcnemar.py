"""cascade.metrics.holm / mcnemar: step-down の標準結果と McNemar 正確検定の既知値（plan §8.2、§8.4）。"""

from __future__ import annotations

import pytest

from cascade.metrics.holm import holm
from cascade.metrics.mcnemar import mcnemar_exact


def test_holm_standard_step_down() -> None:
    res = holm({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert set(res) == {"a", "b", "c"}
    # 昇順 a(0.01) → c(0.03) → b(0.04)。水準 0.05/3, 0.05/2, 0.05/1
    assert res["a"]["level"] == pytest.approx(0.05 / 3)
    assert res["c"]["level"] == pytest.approx(0.05 / 2)
    assert res["b"]["level"] == pytest.approx(0.05)
    # a は棄却、c は 0.03 > 0.025 で停止、b は以降なので非棄却（0.04 ≤ 0.05 でも棄却しない）
    assert res["a"]["reject"] is True
    assert res["c"]["reject"] is False
    assert res["b"]["reject"] is False
    assert res["a"]["p_adj"] == pytest.approx(0.03)
    assert res["c"]["p_adj"] == pytest.approx(0.06)
    assert res["b"]["p_adj"] == pytest.approx(0.06)
    for r in res.values():
        assert r["reject"] == (r["p_adj"] <= 0.05)


def test_holm_all_reject_and_none_reject() -> None:
    res = holm({"x": 0.001, "y": 0.002, "z": 0.003})
    assert all(r["reject"] for r in res.values())
    res = holm({"x": 0.5, "y": 0.6})
    assert not any(r["reject"] for r in res.values())


def test_holm_seven_configs_matches_hand_computation() -> None:
    # §8.2: 残り 7 構成を m = 7 で調整
    p = {f"c{i}": v for i, v in enumerate([0.001, 0.005, 0.009, 0.02, 0.03, 0.2, 0.9])}
    res = holm(p)
    levels = [0.05 / (7 - i) for i in range(7)]
    for i, name in enumerate(p):
        assert res[name]["level"] == pytest.approx(levels[i])
    assert [res[n]["reject"] for n in p] == [True, True, True, False, False, False, False]
    # p_adj_6 = max(…, (7-6)·0.9) = 0.9（単調化しても 1 で打ち切られない）
    assert res["c6"]["p_adj"] == pytest.approx(0.9)


def test_holm_preserves_input_order_and_handles_empty() -> None:
    res = holm({"z": 0.3, "a": 0.1})
    assert list(res) == ["z", "a"]
    assert holm({}) == {}


def test_holm_invalid() -> None:
    with pytest.raises(ValueError):
        holm({"a": 1.5})
    with pytest.raises(ValueError):
        holm({"a": 0.1}, alpha=0.0)


def test_mcnemar_known_value() -> None:
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)


def test_mcnemar_symmetry_and_bounds() -> None:
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 3) == mcnemar_exact(3, 10)
    assert 0.0 < mcnemar_exact(2, 12) < 0.05


def test_mcnemar_one_sided() -> None:
    # b = 5, c = 0: b の側が多い → greater は小さく、less は 1
    assert mcnemar_exact(5, 0, "greater") == pytest.approx(1 / 32)
    assert mcnemar_exact(5, 0, "less") == 1.0
    assert mcnemar_exact(0, 5, "less") == pytest.approx(1 / 32)
    assert mcnemar_exact(0, 5, "greater") == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(1, 1, "both")
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 1)
