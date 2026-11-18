"""Resumable temporal grid across horizons, feature policies and class sampling."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from models.src.config import ALL_CANDIDATE_FEATURES, DATA_DIR, DEFAULT_FEATURES, ERP_FORMULA
from scripts.compare_ensembles import (
    MODELS,
    add_technical_indicators,
    apply_macro_release_lags,
    best_threshold_by_f1,
    build_models,
    compute_macro_and_valuation_features,
    create_targets,
    fit_temporal_classifier,
    load_raw_dataset,
    oversample_class_zero,
    split_train_val_internal,
)

HORIZONS = (6, 12, 24, 36, 48, 60)
FEATURE_SETS = ("previous", "broad", "selected8")
GRIDS = {
    "xgboost": {"max_depth": [1, 2, 4], "min_child_weight": [5, 10]},
    "logistic_l2": {"logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "extra_trees": {
        "extratreesclassifier__max_depth": [2, 4, 8],
        "extratreesclassifier__min_samples_leaf": [4, 8, 16],
    },
    "random_forest": {
        "randomforestclassifier__max_depth": [2, 4, 8],
        "randomforestclassifier__min_samples_leaf": [4, 8, 16],
    },
    "hist_gb": {"max_leaf_nodes": [3, 7, 15], "learning_rate": [0.025, 0.05]},
    "lightgbm": {"num_leaves": [3, 7, 15], "learning_rate": [0.025, 0.05]},
}
TEST_DATES = pd.date_range("2005-01-31", "2020-12-31", freq="ME")


def prepare_grid_values(raw, archived, horizon, *, historical_erp=False):
    values = compute_macro_and_valuation_features(
        apply_macro_release_lags(add_technical_indicators(raw, horizon)), horizon
    )
    if historical_erp:
        # Reconstruct the input of the published models for historical comparisons.
        values["equity_risk_premium"] = values["sp500_earnings_yield"] - values["DGS10"]
    technical = [
        f"ema_{max(3, horizon // 2)}_dist",
        f"ema_{horizon}_dist",
        f"ema_{2 * horizon}_dist",
        "ema_spread",
        "rsi_14",
        f"roc_{horizon}",
        "drawdown_12m",
        "momentum_change",
    ]
    candidates = list(dict.fromkeys([*DEFAULT_FEATURES, *ALL_CANDIDATE_FEATURES, *technical]))
    values = values[[c for c in candidates if c in values]].replace([np.inf, -np.inf], np.nan)
    archived_columns = archived.columns.intersection(DEFAULT_FEATURES).drop(
        "equity_risk_premium", errors="ignore"
    )
    values.update(archived[archived_columns])
    return values


def select_features(x, y, policy):
    """All eligibility and supervised ranking use the unsampled fitting window only."""
    candidates = DEFAULT_FEATURES if policy == "previous" else list(x)
    eligible = [c for c in candidates if c in x and x[c].count() >= 60 and x[c].nunique() > 1]
    if policy != "selected8":
        return eligible
    ranks = x[eligible].corrwith(y, method="spearman").abs().fillna(0)
    ranked = sorted(eligible, key=lambda c: (-ranks[c], c))
    correlation = x[eligible].corr(method="spearman").abs()
    selected = []
    for c in ranked:
        if all(pd.isna(correlation.loc[c, s]) or correlation.loc[c, s] < 0.9 for s in selected):
            selected.append(c)
        if len(selected) == 8:
            break
    return selected


def make_folds(values, target, horizon, policy):
    folds, audit = [], []
    gap = pd.offsets.MonthEnd(horizon)
    for year in [*range(1982, 2000), *range(2005, 2021)]:
        dates = target.loc[str(year)].index
        origin = dates.min()
        train = target.index[target.index + gap < origin]
        fit, es, val = [train[s] for s in split_train_val_internal(len(train), horizon)]
        assert len(fit) and len(es) and len(val) and len(dates) == 12
        assert fit.max() + gap < es.min() and val.max() + gap < origin
        if year < 2005:
            assert dates.max() + gap < TEST_DATES.min()
        cols = select_features(values.loc[fit], target.loc[fit], policy)
        assert cols
        x = values[cols]
        folds.append(
            (
                x.loc[fit],
                target.loc[fit],
                x.loc[es],
                target.loc[es],
                x.loc[val],
                target.loc[val],
                x.loc[dates],
                target.loc[dates],
            )
        )
        audit.append(
            dict(
                horizon=horizon,
                features=policy,
                origin=origin,
                fit_end=fit.max(),
                fit_label_end=fit.max() + gap,
                early_stop_start=es.min(),
                threshold_end=val.max(),
                threshold_label_end=val.max() + gap,
                n_features=len(cols),
                columns=json.dumps(cols),
            )
        )
    return folds, audit


def market_inputs(horizon):
    close = (
        pd.read_csv(DATA_DIR / "sp500.csv", skiprows=[1], index_col="Date", parse_dates=True)[
            "Close"
        ]
        .dropna()
        .sort_index()
    )
    assert close.index.is_unique and (close > 0).all()

    def next_session(dates):
        return close.index[close.index.searchsorted(dates, side="right")]

    buys = next_session(TEST_DATES)
    maturities = TEST_DATES + pd.offsets.MonthEnd(horizon)
    delayed = next_session(maturities)
    terminal = next_session(pd.DatetimeIndex(["2025-12-31"]))[0]
    assert (buys > TEST_DATES).all() and (delayed > maturities).all()
    assert (delayed <= terminal).all()
    return pd.DataFrame(
        dict(
            buy_date=buys,
            delayed_date=delayed,
            immediate=100 * close.loc[terminal] / close.loc[buys].to_numpy(),
            delayed=100 * close.loc[terminal] / close.loc[delayed].to_numpy(),
            terminal=terminal,
        ),
        index=TEST_DATES,
    )


def metrics(frame, market=None):
    positive = frame.actual.eq(0)
    alert = frame.alert.astype(bool)
    tp, fp = int((positive & alert).sum()), int((~positive & alert).sum())
    n0 = int(positive.sum())
    hits = int(positive.loc[frame.p0.nlargest(24).index].sum())
    result = dict(
        n=len(frame),
        n0=n0,
        prevalence0=n0 / len(frame),
        AP0=average_precision_score(positive, frame.p0),
        ROC_AUC=roc_auc_score(positive, frame.p0) if positive.nunique() == 2 else np.nan,
        brier=brier_score_loss(positive, frame.p0),
        logloss=log_loss(positive, frame.p0, labels=[False, True]),
        precision0=tp / (tp + fp) if tp + fp else 0.0,
        recall0=tp / n0 if n0 else 0.0,
        F1_0=2 * tp / (n0 + tp + fp) if n0 + tp + fp else 0.0,
        detected=tp,
        false_alarms=fp,
        alerts=tp + fp,
        precision24=hits / 24,
        recall24=hits / n0 if n0 else 0.0,
    )
    result["AP0_lift"] = result["AP0"] / result["prevalence0"] if n0 else np.nan
    if market is not None:
        assert frame.index.equals(market.index)
        wealth = market.immediate.where(~alert, market.delayed).sum()
        total = len(frame) * 100
        baseline = 100 * (market.immediate.sum() / total - 1)
        roi = 100 * (wealth / total - 1)
        result.update(
            wealth=wealth,
            contributed=total,
            roi_pct=roi,
            dca_roi_pct=baseline,
            vs_DCA_pp=roi - baseline,
        )
    return result


def predict_fold(template, params, name, fold, sampling):
    x, y, xes, yes, xv, yv, xt, yt = fold
    if sampling:
        x, y = oversample_class_zero(x, y)
    model = clone(template).set_params(**params)
    inputs = pd.concat([xv, xt])
    if y.nunique() == 1:
        p0 = pd.Series(float(y.iloc[0] == 0), index=inputs.index)
    else:
        if name == "xgboost":
            fit_temporal_classifier(model, x, y, xes, yes)
        else:
            model.fit(x, y)
        zero = list(model.classes_).index(0)
        p0 = pd.Series(model.predict_proba(inputs)[:, zero], index=inputs.index)
    threshold, _ = best_threshold_by_f1(yv, 1 - p0.loc[xv.index], pos_label=1)
    return (
        pd.DataFrame(
            dict(
                actual=yt,
                p0=p0.loc[xt.index],
                threshold_p0=1 - threshold,
                alert=p0.loc[xt.index] > 1 - threshold,
            )
        ),
        p0.loc[xv.index],
    )


def run_task(horizon, policy, sampling, name, folds, out):
    stem = f"h{horizon}_{policy}_{int(sampling)}_{name}"
    checkpoint = out / "checkpoints" / f"{stem}.json"
    if checkpoint.exists():
        return stem, "cached"
    template = build_models()[name]
    template.set_params(**{k: 1 for k in template.get_params() if k.split("__")[-1] == "n_jobs"})
    market = market_inputs(horizon)
    records, predictions = [], []
    best_key, best_val = None, None
    with threadpool_limits(limits=1):
        for candidate, params in enumerate(ParameterGrid(GRIDS[name])):
            frames, validations = zip(
                *[predict_fold(template, params, name, f, sampling) for f in folds]
            )
            frame = pd.concat(frames)
            dev = metrics(frame.loc[:"1999"])
            evaluation = metrics(frame.loc["2005":], market)
            key = (dev["AP0"], -dev["brier"], -candidate)
            if best_key is None or key > best_key:
                best_key, best_candidate, best_val = key, candidate, validations
            records.append(
                dict(
                    horizon=horizon,
                    features=policy,
                    oversampling=sampling,
                    model=name,
                    candidate=candidate,
                    params=json.dumps(params, sort_keys=True),
                    **{f"dev_{k}": v for k, v in dev.items()},
                    **evaluation,
                )
            )
            predictions.append(frame.assign(candidate=candidate).rename_axis("date").reset_index())
    for row in records:
        row["selected_by_dev"] = row["candidate"] == best_candidate
    pd.concat(predictions).to_csv(out / "checkpoints" / f"{stem}_pred.csv.gz", index=False)
    # Keep only selected models' internal validation probabilities for ensemble thresholds.
    pd.concat(
        [v.rename("p0").to_frame().assign(fold=i) for i, v in enumerate(best_val)]
    ).rename_axis("date").to_csv(out / "checkpoints" / f"{stem}_val.csv.gz")
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(checkpoint)
    return stem, "completed"


def aggregate(out, horizons, targets, fold_groups):
    rows, selected_preds, blends = [], [], []
    for h in horizons:
        for policy in FEATURE_SETS:
            for sampling in (False, True):
                winners, risks, vals = {}, {}, {}
                for name in MODELS:
                    stem = f"h{h}_{policy}_{int(sampling)}_{name}"
                    records = json.loads((out / "checkpoints" / f"{stem}.json").read_text())
                    rows.extend(records)
                    winner = next(r for r in records if r["selected_by_dev"])
                    winners[name] = winner
                    pred = pd.read_csv(
                        out / "checkpoints" / f"{stem}_pred.csv.gz", parse_dates=["date"]
                    )
                    risks[name] = pred.loc[pred.candidate.eq(winner["candidate"])].set_index("date")
                    vals[name] = pd.read_csv(
                        out / "checkpoints" / f"{stem}_val.csv.gz", parse_dates=["date"]
                    )
                    selected_preds.append(
                        risks[name].assign(
                            horizon=h, features=policy, oversampling=sampling, model=name
                        )
                    )
                dev_risk = pd.DataFrame({m: r.p0.loc[:"1999"] for m, r in risks.items()})
                ydev = targets[h].loc[dev_risk.index].eq(0)

                def quality(members):
                    p = dev_risk[list(members)].mean(axis=1)
                    return average_precision_score(ydev, p), -brier_score_loss(ydev, p)

                subsets = [g for n in range(2, 7) for g in combinations(MODELS, n)]
                best_blend = max(subsets, key=quality)
                best_single = max(((m,) for m in MODELS), key=quality)
                for members in subsets:
                    ap, minus_brier = quality(members)
                    blends.append(
                        dict(
                            horizon=h,
                            features=policy,
                            oversampling=sampling,
                            members="+".join(members),
                            dev_AP0=ap,
                            dev_brier=-minus_brier,
                            selected=members == best_blend,
                        )
                    )
                for name, members in (("ensemble", best_blend), ("selected_single", best_single)):
                    frames = []
                    for i, f in enumerate(fold_groups[h, policy]):
                        val = pd.concat(
                            [vals[m].loc[vals[m].fold.eq(i)].set_index("date").p0 for m in members],
                            axis=1,
                        ).mean(axis=1)
                        threshold, _ = best_threshold_by_f1(f[5], 1 - val, pos_label=1)
                        dates = f[-1].index
                        p0 = pd.concat([risks[m].loc[dates].p0 for m in members], axis=1).mean(
                            axis=1
                        )
                        frames.append(
                            pd.DataFrame(
                                dict(
                                    actual=f[-1],
                                    p0=p0,
                                    threshold_p0=1 - threshold,
                                    alert=p0 > 1 - threshold,
                                )
                            )
                        )
                    frame = pd.concat(frames).rename_axis("date")
                    rows.append(
                        dict(
                            horizon=h,
                            features=policy,
                            oversampling=sampling,
                            model=name,
                            candidate=0,
                            params=json.dumps({"members": members}),
                            selected_by_dev=True,
                            **{f"dev_{k}": v for k, v in metrics(frame.loc[:"1999"]).items()},
                            **metrics(frame.loc["2005":], market_inputs(h)),
                        )
                    )
                    selected_preds.append(
                        frame.assign(horizon=h, features=policy, oversampling=sampling, model=name)
                    )
    result = pd.DataFrame(rows).sort_values(
        ["horizon", "features", "oversampling", "model", "candidate"]
    )
    result.to_csv(out / "all_combinations.csv", index=False)
    result.loc[result.selected_by_dev].to_csv(out / "selected_models.csv", index=False)
    pd.concat(selected_preds).rename_axis("date").reset_index().to_csv(
        out / "selected_predictions.csv.gz", index=False
    )
    pd.DataFrame(blends).to_csv(out / "ensemble_validation.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vintages", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    # Refuse to silently reuse checkpoints after code, data or grid changes.
    files = [
        Path(__file__),
        Path(__file__).with_name("compare_ensembles.py"),
        args.vintages,
        *sorted(DATA_DIR.glob("*.csv")),
        *sorted((Path(__file__).parents[1] / "models/src").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    signature = dict(
        sha256=digest.hexdigest(),
        horizons=args.horizons,
        grids=GRIDS,
        equity_risk_premium_formula=ERP_FORMULA,
        development="1982-1999",
        evaluation="2005-2020",
    )
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Inputs changed: choose a new output directory.")
    manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    raw = load_raw_dataset()
    archived = pd.read_csv(args.vintages, index_col="fecha", parse_dates=True)
    targets, fold_groups, audit = {}, {}, []
    for h in args.horizons:
        values = prepare_grid_values(raw, archived, h)
        targets[h] = create_targets(raw, h).target.loc["1962":"2020"].astype(int)
        assert targets[h].index.equals(pd.date_range("1962-01-31", "2020-12-31", freq="ME"))
        for policy in FEATURE_SETS:
            fold_groups[h, policy], entries = make_folds(values, targets[h], h, policy)
            audit.extend(entries)
        market_inputs(h).to_csv(out / f"market_h{h}.csv", index_label="date")
    pd.DataFrame(audit).to_csv(out / "feature_audit.csv", index=False)
    tasks = [
        (h, p, s, m)
        for h in args.horizons
        for p in FEATURE_SETS
        for s in (False, True)
        for m in MODELS
    ]
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_task, h, p, s, m, fold_groups[h, p], out) for h, p, s, m in tasks
        ]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"{i}/{len(tasks)} {future.result()}", flush=True)
    result = aggregate(out, args.horizons, targets, fold_groups)
    print(
        f"Saved {len(result)} rows; {int(result.selected_by_dev.sum())} selected configurations",
        flush=True,
    )


if __name__ == "__main__":
    main()
