"""Compare probability ensembles on purged, expanding monthly time series."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from models.src.config import (
    DEFAULT_FEATURES,
    FIXED_PARAMS_BASE,
    MANUAL_PARAMS_BASE,
    METRICS_DIR,
    MIN_TRAIN_SIZE,
    OVERSAMPLE_CLASS_ZERO,
    OVERSAMPLING_SEED,
    TEST_SIZE,
)
from models.src.data_loader import load_raw_dataset
from models.src.features import (
    add_technical_indicators,
    apply_macro_release_lags,
    compute_macro_and_valuation_features,
    create_targets,
    filter_features_by_history,
)
from models.src.metrics import best_threshold_by_f1
from models.src.tuning import fit_temporal_classifier, oversample_class_zero
from models.src.walk_forward import generate_folds, split_train_val_internal

HORIZON = 36
MODELS = ("xgboost", "logistic_l2", "extra_trees", "hist_gb", "random_forest", "lightgbm")


def build_models():
    return {
        "xgboost": XGBClassifier(**FIXED_PARAMS_BASE, **MANUAL_PARAMS_BASE, n_jobs=2),
        "logistic_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, penalty="l2", max_iter=1000),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=8,
                n_jobs=2,
                random_state=OVERSAMPLING_SEED,
            ),
        ),
        "hist_gb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=OVERSAMPLING_SEED,
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=2,
                random_state=OVERSAMPLING_SEED,
            ),
        ),
        "lightgbm": LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            n_estimators=100,
            num_leaves=7,
            min_child_samples=15,
            reg_lambda=1.0,
            random_state=OVERSAMPLING_SEED,
            n_jobs=2,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ),
    }


def select_members(history, origin):
    """Select among equal-weight blends using only mature earlier walk-forward scores."""
    mature = history.loc[history.index + pd.offsets.MonthEnd(HORIZON) < origin]
    if mature.empty or mature["y"].nunique() < 2:
        return MODELS, (MODELS[0],), len(mature)
    positive = mature["y"].eq(0)

    def quality(members):
        p0 = mature[list(members)].mean(axis=1)
        return average_precision_score(positive, p0), -brier_score_loss(positive, p0)

    blends = (group for size in range(2, len(MODELS) + 1) for group in combinations(MODELS, size))
    singles = ((model,) for model in MODELS)
    return max(blends, key=quality), max(singles, key=quality), len(mature)


def compare_panel(panel, features, target, oversampling):
    history = pd.DataFrame(index=pd.DatetimeIndex([]), columns=[*MODELS, "y"], dtype=float)
    predictions = []
    for train, test in generate_folds(len(target), MIN_TRAIN_SIZE, TEST_SIZE, purge=HORIZON):
        dates = target.index[test]
        fit, es, score = [
            target.index[train][s]
            for s in split_train_val_internal(len(target.iloc[train]), HORIZON)
        ]
        assert fit.max() + pd.offsets.MonthEnd(HORIZON) < es.min()
        assert score.max() + pd.offsets.MonthEnd(HORIZON) < dates.min()
        if panel == "archived_only":
            cols = [c for c in features if features.loc[fit, c].count() >= 60]
        else:
            cols, _ = filter_features_by_history(
                features.loc[features.index < dates[0]], DEFAULT_FEATURES
            )
        if not cols:
            raise ValueError(f"No eligible features for {panel} at {dates[0].date()}")
        X = features[cols]
        X_fit, y_fit = X.loc[fit], target.loc[fit]
        if oversampling:
            X_fit, y_fit = oversample_class_zero(X_fit, y_fit)

        blend, single, n_mature = select_members(history, dates[0])
        groups = {name: (name,) for name in MODELS}
        groups.update(ensemble=blend, selected_single=single)
        risk = pd.DataFrame(index=score.append(dates))
        for name, model in build_models().items():
            if y_fit.nunique() == 1:
                risk[name] = float(y_fit.iloc[0] == 0)
                continue
            if name == "xgboost":
                fit_temporal_classifier(model, X_fit, y_fit, X.loc[es], target.loc[es])
            else:
                model.fit(X_fit, y_fit)
            assert list(model.classes_) == [0, 1]
            risk[name] = model.predict_proba(X.loc[risk.index])[:, 0].astype(float)

        for name, members in groups.items():
            p0 = risk[list(members)].mean(axis=1)
            threshold, _ = best_threshold_by_f1(target.loc[score], 1.0 - p0.loc[score], pos_label=1)
            predictions.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "panel": panel,
                        "model": name,
                        "members": "+".join(members),
                        "selection_rows": n_mature,
                        "features": "|".join(cols),
                        "oversampling": oversampling,
                        "actual": target.loc[dates].to_numpy(),
                        "p0": p0.loc[dates].to_numpy(),
                        "threshold_p0": 1 - threshold,
                        "alert": (p0.loc[dates] > 1 - threshold).to_numpy(),
                    }
                )
            )
        history = pd.concat([history, risk.loc[dates].assign(y=target.loc[dates])])
    return pd.concat(predictions, ignore_index=True)


def summarize(predictions):
    rows = []
    periods = {
        "1982-2004": (1982, 2004),
        "2005-2022": (2005, 2022),
        "2005-2008": (2005, 2008),
        "2009-2022": (2009, 2022),
    }
    for (panel, model, oversampling), group in predictions.groupby(
        ["panel", "model", "oversampling"]
    ):
        for period, (start, end) in periods.items():
            g = group[group.date.dt.year.between(start, end)]
            pos = g.actual.eq(0)
            both = pos.nunique() == 2
            top = g.sort_values(["p0", "date"], ascending=[False, True]).head(24)
            hits = int(top.actual.eq(0).sum())
            detected = int((pos & g.alert).sum())
            alerts = int(g.alert.sum())
            rows.append(
                dict(
                    panel=panel,
                    model=model,
                    oversampling=oversampling,
                    period=period,
                    months=len(g),
                    actual_0=int(pos.sum()),
                    AP_0=average_precision_score(pos, g.p0) if both else np.nan,
                    ROC_AUC=roc_auc_score(pos, g.p0) if both else np.nan,
                    brier=brier_score_loss(pos, g.p0),
                    logloss=log_loss(pos, g.p0, labels=[False, True]),
                    mean_p0=g.p0.mean(),
                    hits24=hits,
                    precision24=hits / len(top),
                    recall24=hits / pos.sum() if pos.any() else np.nan,
                    detected=detected,
                    false_alarms=alerts - detected,
                    F1_0=2 * detected / (pos.sum() + alerts) if pos.sum() + alerts else np.nan,
                )
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vintages",
        type=Path,
        help="Archived monthly feature CSV indexed by fecha. Enables hybrid and archived-only panels.",
    )
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR / "ensemble")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--oversampling", action=argparse.BooleanOptionalAction, default=OVERSAMPLE_CLASS_ZERO
    )
    sampling.add_argument(
        "--both", action="store_true", help="Compare with and without 1:1 oversampling."
    )
    args = parser.parse_args()
    raw = load_raw_dataset()
    features = compute_macro_and_valuation_features(
        apply_macro_release_lags(add_technical_indicators(raw))
    )[DEFAULT_FEATURES]
    target = create_targets(raw)["target"].loc["1962-01":"2023-03"].astype(int)
    assert target.index.equals(pd.date_range("1962-01-31", "2023-03-31", freq="ME"))
    panels = {"revised": features}
    if args.vintages:
        vintage = pd.read_csv(args.vintages, index_col="fecha", parse_dates=True)
        if not vintage.index.is_unique or not vintage.columns.intersection(DEFAULT_FEATURES).size:
            raise ValueError("Vintage file requires unique dates and matching feature columns.")
        archived = vintage.reindex(
            index=features.index, columns=vintage.columns.intersection(DEFAULT_FEATURES)
        )
        hybrid = features.copy()
        hybrid.update(archived)
        panels = {"hybrid": hybrid, "archived_only": archived}
    with threadpool_limits(limits=2):
        parts = []
        for panel, values in panels.items():
            for oversampling in ((False, True) if args.both else (args.oversampling,)):
                parts.append(compare_panel(panel, values, target, oversampling))
                print(f"Completed {panel}, oversampling={oversampling}", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    assert not predictions.duplicated(["panel", "model", "oversampling", "date"]).any()
    summary = summarize(predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    summary.to_csv(args.out_dir / "metrics.csv", index=False)
    print(summary.loc[summary.period.eq("2005-2022")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
