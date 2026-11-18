import numpy as np
import pandas as pd

from scripts.compare_ensembles import MODELS, build_models, select_members, summarize


def test_ensemble_uses_only_predictions_mature_before_origin():
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    history = pd.DataFrame(index=dates)
    history["y"] = [0, 1] * 6
    for name in MODELS:
        history[name] = np.linspace(0.1, 0.8, 12)
    history.loc[dates[:4], MODELS[0]] = [0.85, 0.75, 0.1, 0.01]
    history.loc[dates[:4], MODELS[1]] = [0.1, 0.01, 0.85, 0.75]
    # Neither base ranking is perfect, but their mean separates all four months.
    origin = pd.Timestamp("2003-05-31")
    expected = select_members(history.iloc[:4], origin)
    assert expected[0] == MODELS[:2]
    assert expected[2] == 4
    history.loc[dates[4] :, "y"] = 0
    history.loc[dates[4] :, list(MODELS)] = 1
    assert select_members(history, origin) == expected


def test_ensemble_has_predefined_fallback_until_both_classes_mature():
    dates = pd.date_range("2000-01-31", periods=4, freq="ME")
    history = pd.DataFrame(0.5, index=dates, columns=MODELS).assign(y=1)
    assert select_members(history, dates[0]) == (MODELS, (MODELS[0],), 0)
    assert select_members(history, pd.Timestamp("2004-01-31")) == (MODELS, (MODELS[0],), 4)


def test_paired_comparison_scores_each_sampling_policy_separately():
    assert tuple(build_models()) == MODELS
    dates = pd.to_datetime(["2000-01-31", "2005-01-31", "2010-01-31"])
    perfect = pd.DataFrame(
        {
            "date": dates,
            "panel": "hybrid",
            "model": "random_forest",
            "oversampling": False,
            "actual": [1, 0, 1],
            "p0": [0.2, 0.8, 0.1],
            "alert": [False, True, False],
        }
    )
    other = perfect.assign(oversampling=True, p0=1 - perfect.p0, alert=~perfect.alert)
    metrics = summarize(pd.concat([perfect, other]))
    recent = metrics[metrics.period == "2005-2022"].set_index("oversampling")
    assert recent.months.tolist() == [2, 2]
    assert recent.AP_0.tolist() == [1.0, 0.5]
    assert recent.F1_0.tolist() == [1.0, 0.0]
