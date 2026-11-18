"""Compare directional targets on shared monthly forecast origins.

Run from the repository root with python -m scripts.compare_target_horizons.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from models.src.config import DEFAULT_FEATURES, METRICS_DIR, MIN_TRAIN_SIZE, TEST_SIZE
from models.src.data_loader import load_raw_dataset
from models.src.features import prepare_modeling_dataset
from models.src.metrics import class_zero_metrics
from models.src.walk_forward import generate_folds, run_walk_forward_evaluation
from scripts.compare_negative_policies import SHALLOW

HORIZONS = (6, 12, 24, 36, 48, 60)
FIRST_ORIGIN = pd.Timestamp("1982-01-31")
SHORT_END = pd.Timestamp("2022-12-31")
ALL_END = pd.Timestamp("2020-12-31")
RECENT_START = pd.Timestamp("2005-01-31")


def predict_horizon(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame, features = prepare_modeling_dataset(raw, horizon=horizon)
    last_origin = SHORT_END if horizon <= 36 else ALL_END
    # generate_folds requires a real extra month after the final test block.
    frame = frame.loc[: last_origin + pd.offsets.MonthEnd(1)]
    assert features == DEFAULT_FEATURES
    assert frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], freq="ME"))
    assert frame.index[MIN_TRAIN_SIZE] == FIRST_ORIGIN
    folds = generate_folds(len(frame), MIN_TRAIN_SIZE, TEST_SIZE, purge=horizon)

    # Reconstruct the engine's historical, fold-specific probability baseline.
    baselines = []
    for train, test in folds:
        history = frame.iloc[train]
        dates = frame.index[test]
        assert history.index[-1] + pd.offsets.MonthEnd(horizon) < dates[0]
        baselines.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "fold_start": dates[0],
                    "baseline_proba_up": np.clip(history.target.mean(), 1e-6, 1 - 1e-6),
                }
            )
        )

    with TemporaryDirectory() as diagnostics:
        predictions, _ = run_walk_forward_evaluation(
            frame,
            features,
            model_name="xgboost",
            oversampling=False,
            horizon=horizon,
            min_train_size=MIN_TRAIN_SIZE,
            test_size=TEST_SIZE,
            manual_params=SHALLOW,
            do_random_search=False,
            class_zero_weight=1.0,
            threshold_class=1,
            out_dir=Path(diagnostics),
        )
    predictions = predictions.merge(pd.concat(baselines), on="date", validate="one_to_one")
    expected = pd.date_range(FIRST_ORIGIN, last_origin, freq="ME")
    assert pd.DatetimeIndex(predictions.date).equals(expected)
    maturity = predictions.date + pd.offsets.MonthEnd(horizon)
    np.testing.assert_array_equal(predictions.close_t_plus_h, raw.loc[maturity, "Close"])
    np.testing.assert_array_equal(
        predictions.actual, predictions.close_t_plus_h > predictions.close_t
    )
    predictions.insert(0, "horizon", horizon)
    return predictions


def summarize(predictions: pd.DataFrame) -> dict:
    actual = predictions.actual
    probabilities = predictions.proba_up
    baseline = predictions.baseline_proba_up
    brier = brier_score_loss(actual, probabilities)
    baseline_brier = brier_score_loss(actual, baseline)
    model_loss = log_loss(actual, probabilities, labels=[0, 1])
    base_loss = log_loss(actual, baseline, labels=[0, 1])
    return {
        "first_date": predictions.date.min().date(),
        "last_date": predictions.date.max().date(),
        "months": len(actual),
        "actual_0": int((actual == 0).sum()),
        "rate_0": (actual == 0).mean(),
        "accuracy": (actual == predictions.pred).mean(),
        "balanced_accuracy": balanced_accuracy_score(actual, predictions.pred),
        "roc_auc": roc_auc_score(actual, probabilities),
        "logloss": model_loss,
        "baseline_logloss": base_loss,
        "logloss_skill": 1 - model_loss / base_loss,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": 1 - brier / baseline_brier,
        **class_zero_metrics(actual, probabilities, predictions.pred),
    }


def main() -> None:
    raw = load_raw_dataset()
    pieces = []
    for horizon in HORIZONS:
        print(f"Evaluating {horizon} months", flush=True)
        pieces.append(predict_horizon(raw, horizon))
    predictions = pd.concat(pieces, ignore_index=True)
    panels = (
        ("short", FIRST_ORIGIN, SHORT_END, HORIZONS[:4]),
        ("short_recent", RECENT_START, SHORT_END, HORIZONS[:4]),
        ("all", FIRST_ORIGIN, ALL_END, HORIZONS),
        ("all_recent", RECENT_START, ALL_END, HORIZONS),
    )
    rows = []
    for panel, start, end, horizons in panels:
        period = predictions[predictions.date.between(start, end)]
        for horizon in horizons:
            part = period[period.horizon == horizon]
            assert pd.DatetimeIndex(part.date).equals(pd.date_range(start, end, freq="ME"))
            rows.append({"panel": panel, "horizon": horizon, **summarize(part)})
    result = pd.DataFrame(rows)
    output = METRICS_DIR / "target_horizons"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    result.to_csv(output / "summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
