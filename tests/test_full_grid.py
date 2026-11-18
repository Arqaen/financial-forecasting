"""Temporal, selection and accounting checks for the factorial experiment."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from scripts.compare_full_grid import HORIZONS, TEST_DATES, make_folds, metrics, select_features


@pytest.mark.parametrize("horizon", HORIZONS)
def test_all_fold_labels_mature_before_use(horizon):
    dates = pd.date_range("1962-01-31", "2020-12-31", freq="ME")
    y = pd.Series(np.arange(len(dates)) % 2, index=dates)
    x = pd.DataFrame({"credit_spread": np.sin(np.arange(len(dates)))}, index=dates)
    folds, audit = make_folds(x, y, horizon, "previous")
    assert len(folds) == 34
    assert pd.concat([f[-1] for f in folds[18:]]).index.equals(TEST_DATES)
    for row in audit:
        assert row["fit_label_end"] < row["early_stop_start"]
        assert row["threshold_label_end"] < row["origin"]
    dev_end = folds[17][-1].index.max()
    assert dev_end + pd.offsets.MonthEnd(horizon) < TEST_DATES.min()


def test_supervised_feature_selection_does_not_see_future(monkeypatch):
    dates = pd.date_range("1962-01-31", "2020-12-31", freq="ME")
    rng = np.random.RandomState(42)
    y = pd.Series(rng.randint(0, 2, len(dates)), index=dates)
    x = pd.DataFrame({"a": rng.randn(len(dates)), "b": rng.randn(len(dates))}, index=dates)
    folds, audit = make_folds(x, y, 36, "selected8")
    changed = x.copy()
    changed.loc["2005":] = 1e9
    y_changed = y.copy()
    y_changed.loc["2005":] = 1 - y_changed.loc["2005":]
    folds2, audit2 = make_folds(changed, y_changed, 36, "selected8")
    assert audit[:19] == audit2[:19]
    pd.testing.assert_frame_equal(folds[18][0], folds2[18][0])
    pd.testing.assert_series_equal(folds[18][1], folds2[18][1])


def test_selection_excludes_insufficient_history_and_redundancy():
    y = pd.Series([0, 1] * 50)
    x = pd.DataFrame(
        {"a": y, "duplicate": y, "too_new": [np.nan] * 50 + list(y[:50]), "constant": 1}
    )
    assert select_features(x, y, "selected8") == ["a"]
    assert select_features(x, y, "broad") == ["a", "duplicate"]


def test_metrics_score_zero_class_and_account_for_uninvested_contributions():
    dates = pd.date_range("2005-01-31", periods=24, freq="ME")
    actual = [0, 1] * 12
    frame = pd.DataFrame(
        {"actual": actual, "p0": [0.9, 0.1] * 12, "alert": [True, False] * 12}, index=dates
    )
    market = pd.DataFrame({"immediate": 200.0, "delayed": 150.0}, index=dates)
    result = metrics(frame, market)
    assert result["AP0"] == average_precision_score(np.array(actual) == 0, frame.p0) == 1
    assert result["F1_0"] == 1
    assert result["wealth"] == 12 * 150 + 12 * 200
    assert result["roi_pct"] == 75
    assert result["vs_DCA_pp"] == -25
