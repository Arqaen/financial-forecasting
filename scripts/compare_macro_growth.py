"""Measure the Random Forest impact of replacing lagged macro levels with growth rates."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.compare_full_grid import (
    HORIZONS,
    TEST_DATES,
    apply_macro_release_lags,
    create_targets,
    load_raw_dataset,
    make_folds,
    market_inputs,
    metrics,
    prepare_grid_values,
    run_task,
)

POLICIES = ("broad", "selected8")
LEVEL_NAMES = {"houst_yoy": "HOUST", "totalsa_yoy": "TOTALSA"}
GROUP = ["horizon", "features", "oversampling"]
MODEL = "random_forest"
DEVELOPMENT_DATES = pd.date_range("1982-01-31", "1999-12-31", freq="ME")
AUDIT_DATES = [
    "origin",
    "fit_end",
    "fit_label_end",
    "early_stop_start",
    "threshold_end",
    "threshold_label_end",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--vintages", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    raw = load_raw_dataset()
    lagged = apply_macro_release_lags(raw)
    archived = pd.read_csv(args.vintages, index_col="fecha", parse_dates=True)
    old_audit = pd.read_csv(args.baseline / "feature_audit.csv", parse_dates=AUDIT_DATES)
    old_audit = old_audit.loc[old_audit.features.isin(POLICIES)].reset_index(drop=True)
    targets, markets, fold_groups = {}, {}, {}
    growth_audit, level_audit = [], []
    for horizon in HORIZONS:
        target = create_targets(raw, horizon).target.loc["1962":"2020"].astype(int)
        targets[horizon] = target
        markets[horizon] = market_inputs(horizon)
        old_market = pd.read_csv(
            args.baseline / f"market_h{horizon}.csv",
            index_col="date",
            parse_dates=["date", "buy_date", "delayed_date", "terminal"],
        )
        pd.testing.assert_frame_equal(
            markets[horizon].rename_axis("date"), old_market, check_freq=False
        )

        growth = prepare_grid_values(raw, archived, horizon, historical_erp=True)
        levels = growth.rename(columns=LEVEL_NAMES).assign(**lagged[list(LEVEL_NAMES.values())])
        for policy in POLICIES:
            _, audit = make_folds(levels, target, horizon, policy)
            level_audit.extend(audit)
            fold_groups[horizon, policy], audit = make_folds(growth, target, horizon, policy)
            growth_audit.extend(audit)

    pd.testing.assert_frame_equal(pd.DataFrame(level_audit), old_audit)
    growth_audit = pd.DataFrame(growth_audit)
    dates = [column for column in old_audit if column not in ("columns", "n_features")]
    pd.testing.assert_frame_equal(growth_audit[dates], old_audit[dates])

    groups = [
        (horizon, policy, sampling) for horizon, policy in fold_groups for sampling in (False, True)
    ]
    old_results = pd.read_csv(args.baseline / "selected_models.csv")
    old_results = old_results.loc[old_results.model.eq(MODEL) & old_results.features.isin(POLICIES)]
    old_predictions = pd.read_csv(
        args.baseline / "selected_predictions.csv.gz",
        parse_dates=["date"],
        float_precision="round_trip",
    )
    old_predictions = old_predictions.loc[
        old_predictions.model.eq(MODEL) & old_predictions.features.isin(POLICIES)
    ]
    old_results = old_results.set_index(GROUP).sort_index()
    assert old_results.index.is_unique and set(old_results.index) == set(groups)
    assert set(old_predictions.groupby(GROUP).groups) == set(groups)
    print(
        f"Verified original feature selections, prices and dates for {len(groups)} groups",
        flush=True,
    )

    with TemporaryDirectory() as temporary_dir:
        work = Path(temporary_dir)
        checkpoints = work / "checkpoints"
        checkpoints.mkdir()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_task, h, p, s, MODEL, fold_groups[h, p], work) for h, p, s in groups
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                print(f"{completed}/{len(groups)} {future.result()[0]}", flush=True)

        rows, predictions = [], []
        for h, policy, sampling in groups:
            stem = f"h{h}_{policy}_{int(sampling)}_{MODEL}"
            candidates = json.loads((checkpoints / f"{stem}.json").read_text())
            frames = pd.read_csv(
                checkpoints / f"{stem}_pred.csv.gz",
                parse_dates=["date"],
                float_precision="round_trip",
            )
            for row in candidates:
                frame = frames.loc[frames.candidate.eq(row["candidate"])].set_index("date")
                row.update(verify_prediction(frame, row, targets[h], markets[h], markets[36]))
                if row["selected_by_dev"]:
                    predictions.append(
                        frame.reset_index().assign(
                            horizon=h, features=policy, oversampling=sampling
                        )
                    )
            rows.extend(candidates)

    for key, frame in old_predictions.groupby(GROUP):
        row = old_results.loc[key]
        fixed_wait = verify_prediction(
            frame.set_index("date"), row, targets[key[0]], markets[key[0]], markets[36]
        )
        for metric, value in fixed_wait.items():
            assert np.isclose(row[metric], value, rtol=0, atol=1e-9)

    results = pd.DataFrame(rows)
    winners = results.loc[results.selected_by_dev]
    metrics_to_compare = ["dev_AP0", "AP0", "F1_0", "brier", "roi_pct", "roi_fixed36_pct"]
    keys_and_metrics = [*GROUP, "params", *metrics_to_compare]
    comparison = old_results.reset_index()[keys_and_metrics].merge(
        winners[keys_and_metrics], on=GROUP, suffixes=("_level", "_yoy"), validate="one_to_one"
    )
    assert len(comparison) == len(groups)
    for metric in metrics_to_compare:
        comparison[f"delta_{metric}"] = comparison[f"{metric}_yoy"] - comparison[f"{metric}_level"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.sort_values(GROUP).to_csv(args.out_dir / "comparison.csv", index=False)
    results.to_csv(args.out_dir / "all_yoy_candidates.csv", index=False)
    growth_audit.to_csv(args.out_dir / "feature_audit.csv", index=False)
    pd.concat(predictions).to_csv(args.out_dir / "selected_yoy_predictions.csv.gz", index=False)


def verify_prediction(frame, row, target, market, fixed_market):
    assert frame.index.equals(DEVELOPMENT_DATES.append(TEST_DATES))
    assert np.array_equal(frame.actual, target.loc[frame.index])
    assert frame.p0.between(0, 1).all()
    assert (frame.alert == frame.p0.gt(frame.threshold_p0)).all()
    test = frame.loc["2005":]
    for name, expected in metrics(frame.loc[:"1999"]).items():
        assert np.isclose(row[f"dev_{name}"], expected, rtol=0, atol=1e-9)
    for name, expected in metrics(test, market).items():
        assert np.isclose(row[name], expected, rtol=0, atol=1e-9)
    fixed = metrics(test, fixed_market)
    return {"roi_fixed36_pct": fixed["roi_pct"], "vs_DCA_fixed36_pp": fixed["vs_DCA_pp"]}


if __name__ == "__main__":
    main()
