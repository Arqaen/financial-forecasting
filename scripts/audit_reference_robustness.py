"""Audit the fixed 36-month LightGBM study without selecting a new model or strategy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from models.src.config import DATA_DIR, DEFAULT_FEATURES, DEFAULT_HORIZON, DEFAULT_MODEL
from models.src.data_loader import load_raw_dataset
from models.src.estimators import build_classifier, model_parameters
from models.src.features import create_targets
from scripts.compare_full_grid import (
    TEST_DATES,
    make_folds,
    market_inputs,
    predict_fold,
    prepare_grid_values,
)
from scripts.compare_signal_strategies import simulate
from scripts.train_selected_model import ARCHIVED_FEATURES, DEFAULT_OUTPUT

HORIZON = DEFAULT_HORIZON
DELAY_POLICY = f"delay_{HORIZON}"
EPISODE_END = "2008-12-31"


def classification(frame):
    negative = frame.actual.eq(0)
    n0 = int(negative.sum())
    has_both_classes = 0 < n0 < len(frame)
    return {
        "months": len(frame),
        "negative_months": n0,
        "prevalence_0": float(negative.mean()),
        "roc_auc": roc_auc_score(negative, frame.p0) if has_both_classes else None,
        "ap_0": average_precision_score(negative, frame.p0) if has_both_classes else None,
        "brier": brier_score_loss(negative, frame.p0),
        "hits": int((negative & frame.alert).sum()),
        "false_alarms": int((~negative & frame.alert).sum()),
    }


def block_sensitivity(frame, payoff_pp):
    """Resample paired untuned outcomes in contiguous circular blocks."""
    rng = np.random.default_rng(42)
    draws = 5000
    n = len(frame)
    negative = frame.actual.eq(0).to_numpy()
    results = []
    for block in (HORIZON, 48, 60):
        starts = rng.integers(n, size=(draws, (n + block - 1) // block))
        indices = (starts[:, :, None] + np.arange(block)) % n
        indices = indices.reshape(draws, -1)[:, :n]
        delta = payoff_pp.to_numpy()[indices].mean(axis=1)
        negative_counts = negative[indices].sum(axis=1)
        results.append(
            {
                "block_months": block,
                "draws": draws,
                "roi_delta_pp_percentiles_2_5_50_97_5": np.percentile(
                    delta, [2.5, 50, 97.5]
                ).tolist(),
                "share_roi_delta_at_or_below_zero": float(np.mean(delta <= 0)),
                "share_without_negative_labels": float(np.mean(negative_counts == 0)),
            }
        )
    return results


def compare_portfolios(frame, close, terminal, fee_bps=0, cash_rate=0):
    dca, _ = simulate(frame, close, "dca", terminal, fee_bps, cash_rate)
    signal, _ = simulate(frame, close, DELAY_POLICY, terminal, fee_bps, cash_rate)
    np.testing.assert_allclose([dca["contributed"], signal["contributed"]], 100 * len(frame))
    return {
        "fee_bps": fee_bps,
        "cash_rate": cash_rate,
        "dca": dca,
        "signal": signal,
        "roi_delta_pp": signal["roi_pct"] - dca["roi_pct"],
        "xirr_delta_pp": signal["xirr_pct"] - dca["xirr_pct"],
    }


def panel_sensitivity(archive, reference, payoff):
    raw = load_raw_dataset()
    target = create_targets(raw, HORIZON).target.loc["1962":"2020"].astype(int)
    columns = [name for name in DEFAULT_FEATURES if name in archive]
    hybrid = prepare_grid_values(raw, archive, HORIZON)
    archived_only = archive.reindex(target.index)[columns]
    template = build_classifier(DEFAULT_MODEL, **model_parameters())
    results = {}
    for name, values in (
        ("hybrid_five", hybrid),
        ("hybrid_without_erp", hybrid[columns]),
        ("archive_only", archived_only),
    ):
        folds, audit = make_folds(values, target, HORIZON, "previous")
        predicted = pd.concat(
            [predict_fold(template, {}, DEFAULT_MODEL, fold, True)[0] for fold in folds]
        )
        test = predicted.loc[TEST_DATES]
        pd.testing.assert_series_equal(test.actual, reference.actual, check_index_type=False)
        if name == "hybrid_five":
            np.testing.assert_allclose(test.p0, reference.p0, atol=1e-12)
            np.testing.assert_array_equal(test.alert, reference.alert)
        results[name] = {
            "development": classification(predicted.loc["1982":"1999"]),
            "evaluation": classification(test),
            "roi_delta_pp": float(payoff.where(test.alert, 0.0).mean()),
            "features_in_first_test_fold": json.loads(
                next(row["columns"] for row in audit if row["origin"] == TEST_DATES[0])
            ),
        }
    return results


def audit(spy_csv=None):
    prediction_path = DEFAULT_OUTPUT / "walk_forward_predictions.csv"
    price_path = DATA_DIR / "sp500.csv"
    predictions = pd.read_csv(prediction_path, index_col="date", parse_dates=True)
    test = predictions.loc[TEST_DATES]
    market = market_inputs(HORIZON)
    payoff = market.delayed - market.immediate  # 100 contributed per month
    realized_payoff = payoff.where(test.alert, 0.0)
    terminal = market.terminal.iloc[0]
    close = pd.read_csv(price_path, skiprows=[1], index_col="Date", parse_dates=True).Close
    archive = pd.read_csv(ARCHIVED_FEATURES, index_col="fecha", parse_dates=True)
    cohorts = {"2005-2008": test.loc[:EPISODE_END], "2009-2020": test.loc["2009":]}

    checks = (
        (0, 0.0),
        (10, 0.0),
        (25, 0.0),
        (10, 0.02),
    )
    scenarios = [compare_portfolios(test, close, terminal, fee, cash) for fee, cash in checks]
    np.testing.assert_allclose(scenarios[0]["roi_delta_pp"], realized_payoff.mean())
    phases = [test.iloc[offset::HORIZON] for offset in range(HORIZON)]
    negative_counts = [int(phase.actual.eq(0).sum()) for phase in phases]
    nonoverlap = {
        "offsets": len(phases),
        "windows_per_offset_min_max": [min(map(len, phases)), max(map(len, phases))],
        "negative_windows_per_offset_min_max": [min(negative_counts), max(negative_counts)],
        "offsets_with_both_classes": sum(
            0 < n0 < len(phase) for n0, phase in zip(negative_counts, phases)
        ),
    }
    result = {
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (prediction_path, price_path, ARCHIVED_FEATURES)
        },
        "valuation_date": terminal.strftime("%Y-%m-%d"),
        "classification": classification(test),
        "cohorts": {
            name: {
                **classification(frame),
                "roi_delta_pp_within_cohort": float(realized_payoff.loc[frame.index].mean()),
                "roi_delta_pp_contributed_to_full_result": float(
                    realized_payoff.loc[frame.index].sum() / len(test)
                ),
            }
            for name, frame in cohorts.items()
        },
        "nonoverlapping_label_windows": nonoverlap,
        "historical_block_sensitivity": block_sensitivity(test, realized_payoff),
        "price_index_scenarios": scenarios,
        "archived_months": {
            period: archive.reindex(columns=DEFAULT_FEATURES).loc[start:end].count().to_dict()
            for period, start, end in (
                ("1962-1981", "1962", "1981"),
                ("1982-1999", "1982", "1999"),
                ("2005-2020", "2005", "2020"),
            )
        },
        "panel_sensitivity": panel_sensitivity(archive, test, payoff),
    }
    if spy_csv is not None:
        prices = pd.read_csv(spy_csv, index_col="Date", parse_dates=True)["Adj Close"]
        required = close.loc[market.buy_date.min() : terminal].index
        prices = prices.reindex(required)
        if not np.isfinite(prices).all() or not prices.gt(0).all():
            raise ValueError("SPY adjusted closes must cover every valuation and execution session")
        result["input_sha256"]["spy_adjusted"] = hashlib.sha256(spy_csv.read_bytes()).hexdigest()
        result["spy_adjusted_scenarios"] = [
            compare_portfolios(test, prices, terminal, fee, cash) for fee, cash in checks
        ]
        result["spy_adjusted_cohorts"] = {
            name: compare_portfolios(frame, prices, terminal)["roi_delta_pp"]
            for name, frame in cohorts.items()
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spy-csv", type=Path, help="Optional daily SPY CSV with Date and Adj Close"
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.spy_csv), indent=2, allow_nan=False))
