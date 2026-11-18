"""Economic accounting, causal execution and selection checks for signal policies."""

import numpy as np
import pandas as pd
import pytest

from scripts.compare_signal_strategies import (
    POLICIES,
    next_session,
    purchase_schedule,
    sample_regression,
    select_development,
    simulate,
)


def fixture_market():
    close = pd.Series(100.0, index=pd.bdate_range("2000-01-01", "2005-01-31"))
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    signals = pd.DataFrame({"alert": [True, False] * 6, "exposure": 0.4}, index=dates)
    return close, signals


@pytest.mark.parametrize("policy", ("dca", *POLICIES))
def test_flat_market_conserves_contributions_and_never_borrows(policy):
    close, signals = fixture_market()
    metrics, history = simulate(signals, close, policy, close.index[-1])
    assert metrics["contributed"] == 1200
    assert metrics["wealth"] == pytest.approx(1200)
    assert metrics["roi_pct"] == pytest.approx(0)
    assert metrics["max_drawdown_pct"] == pytest.approx(0)
    assert (history.cash >= 0).all()
    assert (history.wealth >= history.cash - 1e-8).all()


def test_dca_and_fixed_delay_match_independent_lot_valuation():
    close, signals = fixture_market()
    close[:] = np.linspace(50, 200, len(close))
    terminal = close.index[-1]
    for policy in ("dca", "delay_36"):
        expected = 0
        for date, row in signals.iterrows():
            delay = 36 if policy == "delay_36" and row.alert else 0
            buy = next_session(close.index, date + pd.offsets.MonthEnd(delay))
            expected += 100 * close.loc[terminal] / close.loc[buy]
        result, _ = simulate(signals, close, policy, terminal)
        assert result["wealth"] == pytest.approx(expected)


def test_wait_good_releases_only_on_later_signal_and_never_same_close():
    close, signals = fixture_market()
    contributions, purchases = purchase_schedule(signals, close, "wait_good_36", close.index[-1])
    first = next_session(close.index, signals.index[0])
    second = next_session(close.index, signals.index[1])
    assert first > signals.index[0] and second > signals.index[1]
    assert contributions[first] == 100
    assert purchases[first] == 0
    assert purchases[second] == 200


def test_future_signal_changes_cannot_change_earlier_wealth():
    close, signals = fixture_market()
    changed = signals.copy()
    boundary = signals.index[6]
    changed.loc[boundary:, "alert"] = ~changed.loc[boundary:, "alert"]
    for policy in ("wait_good_36", "rebalance_binary"):
        _, h1 = simulate(signals, close, policy, close.index[-1])
        _, h2 = simulate(changed, close, policy, close.index[-1])
        pd.testing.assert_frame_equal(h1.loc[:boundary], h2.loc[:boundary])


def test_pending_cash_is_counted_at_terminal_without_forced_purchase():
    close, signals = fixture_market()
    signals["alert"] = True
    terminal = pd.Timestamp("2001-01-31")
    result, history = simulate(signals, close, "delay_36", terminal)
    assert result["wealth"] == result["terminal_cash"] == 1200
    assert history.nav.iloc[-1] == pytest.approx(1)


def test_fee_accounting_and_risk_are_not_distorted_by_contributions():
    close, signals = fixture_market()
    result, history = simulate(signals, close, "rebalance_binary", close.index[-1], fee_bps=25)
    assert result["fees"] > 0
    assert result["wealth"] + result["fees"] == pytest.approx(1200)
    assert result["roi_pct"] < 0
    # Deposit after an index loss must not mask the flow-adjusted drawdown.
    drop_date = next_session(close.index, signals.index[1])
    close.loc[drop_date:] = 50
    _, history = simulate(signals, close, "dca", close.index[-1])
    assert history.loc[drop_date, "drawdown_pct"] == pytest.approx(-50)


def test_negative_oversampling_preserves_continuous_labels():
    x = pd.DataFrame({"x": range(6)})
    y = pd.Series([-0.4, -0.1, 0.1, 0.2, 0.3, 0.4])
    sampled_x, sampled_y = sample_regression(x, y)
    assert len(sampled_y) == 8
    assert (sampled_y <= 0).sum() == (sampled_y > 0).sum() == 4
    assert sampled_y.tolist()[:6] == y.tolist()
    assert sampled_y.iloc[6:].isin(y.iloc[:2]).all()
    np.testing.assert_array_equal(sampled_y, y.iloc[sampled_x.x])


def test_model_strategy_selection_cannot_use_test_roi():
    rows = pd.DataFrame(
        {
            "combination": ["a", "b"],
            "dev_roi_pct": [30, 20],
            "dev_turnover_multiple": [1, 1],
            "test_roi_pct": [-20, 2000],
        }
    )
    assert select_development(rows).combination == "a"
    rows["test_roi_pct"] = [1e9, -1e9]
    assert select_development(rows).combination == "a"
