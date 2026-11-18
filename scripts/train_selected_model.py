"""Evaluate and refit the selected LightGBM configuration with consistent ERP units.

Run from the repository root: python -m scripts.train_selected_model
No grid search or test-set parameter selection is performed here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import Booster

from models.src.config import DEFAULT_FEATURES, DEFAULT_HORIZON, DEFAULT_MODEL, MODELS_DIR
from models.src.data_loader import load_raw_dataset
from models.src.estimators import build_classifier, model_parameters, save_fitted_model
from models.src.features import create_targets
from models.src.tuning import oversample_class_zero
from scripts.compare_full_grid import (
    make_folds,
    market_inputs,
    metrics,
    predict_fold,
    prepare_grid_values,
)

ROOT = MODELS_DIR.parent
ARCHIVED_FEATURES = MODELS_DIR / "reference" / "archived_features.csv"
HISTORICAL_OUTPUT = MODELS_DIR / "artifacts" / "selected_lightgbm"
DEFAULT_OUTPUT = MODELS_DIR / "artifacts" / "selected_lightgbm_erp_corrected"


def train_reference(output_dir: Path | None = None, *, historical_erp: bool = False) -> dict:
    reference_dir = HISTORICAL_OUTPUT if historical_erp else DEFAULT_OUTPUT
    output_dir = output_dir or reference_dir
    other_reference = DEFAULT_OUTPUT if historical_erp else HISTORICAL_OUTPUT
    if output_dir.resolve() == other_reference.resolve():
        raise ValueError("Choose an output directory for the requested ERP formula.")

    horizon = DEFAULT_HORIZON
    raw = load_raw_dataset()
    archived = pd.read_csv(ARCHIVED_FEATURES, index_col="fecha", parse_dates=True)
    values = prepare_grid_values(raw, archived, horizon, historical_erp=historical_erp)[
        DEFAULT_FEATURES
    ]
    target = create_targets(raw, horizon).target.loc["1962":"2020"].astype(int)
    folds, audit = make_folds(values, target, horizon, "previous")
    template = build_classifier(DEFAULT_MODEL, **model_parameters())
    predictions = pd.concat(
        [predict_fold(template, {}, DEFAULT_MODEL, fold, True)[0] for fold in folds]
    )
    test = predictions.loc["2005":"2020"]
    results = metrics(test, market_inputs(horizon))
    if historical_erp:
        selected = pd.read_csv(ROOT / "docs/experiments/multihorizon-grid/selected_models.csv")
        expected = selected.loc[
            (selected.horizon == horizon)
            & (selected.features == "previous")
            & (selected.model == DEFAULT_MODEL)
            & selected.oversampling
        ].squeeze()
        for key in ("AP0", "ROC_AUC", "F1_0", "detected", "false_alarms", "roi_pct", "vs_DCA_pp"):
            if not np.isclose(results[key], expected[key], rtol=1e-9, atol=1e-10):
                raise ValueError(
                    f"Reference result changed for {key}: {results[key]} != {expected[key]}"
                )

    # Freeze the reference training calendar. Later datasets require a new evaluation.
    X = values.loc[target.index]
    X_fit, y_fit = oversample_class_zero(X, target)
    model = build_classifier(DEFAULT_MODEL, **model_parameters())
    model.fit(X_fit, y_fit)
    save_fitted_model(
        model,
        X,
        target,
        horizon=horizon,
        oversampling=True,
        class_zero_weight=1.0,
        output_dir=output_dir,
        historical_erp=historical_erp,
    )
    reloaded = Booster(model_file=str(output_dir / "model/model.txt"))
    np.testing.assert_allclose(reloaded.predict(X), model.predict_proba(X)[:, 1], rtol=1e-12)
    predictions.to_csv(output_dir / "walk_forward_predictions.csv", index_label="date")
    pd.DataFrame(audit).to_csv(output_dir / "fold_audit.csv", index=False)
    report = {
        "model": DEFAULT_MODEL,
        "feature_panel": "hybrid: archived values where available, otherwise lagged revised data",
        "archived_features_sha256": hashlib.sha256(ARCHIVED_FEATURES.read_bytes()).hexdigest(),
        "development_period": "1982-1999",
        "test_period": "2005-2020",
        "development": metrics(predictions.loc[:"1999"]),
        "test": results,
        "native_reload_verified": True,
        "note": "The exported model is refitted on 1962-2020, including the former test rows. Only the separate annual walk-forward predictions support the reported test metrics. This is a research reference fit, not a live forecast or an independently confirmed investment strategy.",
    }
    (output_dir / "verification.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"Verified reference model saved to {output_dir.resolve()}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--historical-erp", action="store_true", help="Reproduce the published fit."
    )
    args = parser.parse_args()
    train_reference(args.out_dir, historical_erp=args.historical_erp)
