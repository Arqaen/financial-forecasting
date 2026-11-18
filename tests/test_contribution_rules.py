"""Funding, causality and scale invariance of signal-based purchase sizing."""

import numpy as np
import pandas as pd
import pytest

from scripts.compare_contribution_rules import RULES, requested_amount, simulate_amounts


def market():
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    frame = pd.DataFrame({"p0": [0.8, 0.2] * 6, "alert": [True, False] * 6}, index=dates)
    close = pd.Series(100.0, index=pd.bdate_range("2000-01-01", "2002-01-31"))
    return frame, close


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("funding", ["equal", "variable"])
def test_flat_market_conserves_all_external_cash(rule, funding):
    frame, close = market()
    result, ledger = simulate_amounts(
        requested_amount(frame, rule), close, close.index[-1], funding
    )
    assert result["wealth"] == pytest.approx(result["contributed"])
    assert result["roi_pct"] == pytest.approx(0)
    assert result["max_drawdown_pct"] == pytest.approx(0)
    assert (ledger.cash_after >= 0).all()
    assert result["deployed"] + result["terminal_cash"] == pytest.approx(result["contributed"])
    if funding == "equal":
        assert result["contributed"] == 1200


def test_double_class_purchase_scales_wealth_not_roi():
    frame, close = market()
    close[:] = np.linspace(50, 200, len(close))
    one, _ = simulate_amounts(
        requested_amount(frame, "class_1_0"), close, close.index[-1], "variable"
    )
    two, _ = simulate_amounts(
        requested_amount(frame, "class_2_0"), close, close.index[-1], "variable"
    )
    assert two["contributed"] == 2 * one["contributed"]
    assert two["wealth"] == pytest.approx(2 * one["wealth"])
    assert two["roi_pct"] == pytest.approx(one["roi_pct"])
    assert two["xirr_pct"] == pytest.approx(one["xirr_pct"])


def test_extra_purchase_requires_previously_accumulated_cash():
    frame, close = market()
    _, ledger = simulate_amounts(
        requested_amount(frame, "class_2_0"), close, close.index[-1], "equal"
    )
    assert ledger.purchase.tolist() == [0, 200] * 6
    assert ledger.cash_after.tolist() == [100, 0] * 6
    assert (ledger.execution_date > ledger.signal_date).all()


def test_200_100_with_only_100_funding_collapses_to_dca():
    frame, close = market()
    close[:] = np.linspace(50, 200, len(close))
    a, _ = simulate_amounts(requested_amount(frame, "class_2_1"), close, close.index[-1], "equal")
    b, _ = simulate_amounts(requested_amount(frame, "dca"), close, close.index[-1], "equal")
    assert a["wealth"] == pytest.approx(b["wealth"])


def test_future_changes_cannot_rescale_prior_purchases():
    frame, close = market()
    request = requested_amount(frame, "probability_2")
    altered = request.copy()
    altered.iloc[6:] = 300
    for funding in ("equal", "variable"):
        _, a = simulate_amounts(request, close, close.index[-1], funding)
        _, b = simulate_amounts(altered, close, close.index[-1], funding)
        pd.testing.assert_frame_equal(a.iloc[:6], b.iloc[:6])


def test_no_variable_contributions_has_no_defined_roi():
    frame, close = market()
    request = pd.Series(0.0, index=frame.index)
    result, _ = simulate_amounts(request, close, close.index[-1], "variable")
    assert result["wealth"] == result["contributed"] == 0
    assert np.isnan(result["roi_pct"])
