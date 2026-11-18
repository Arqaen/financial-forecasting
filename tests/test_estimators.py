"""Regression tests for the selected configuration and portable native exports."""

import json

import numpy as np
import pandas as pd
import pytest
from lightgbm import Booster

from models.src.config import DEFAULT_HORIZON, DEFAULT_MODEL, OVERSAMPLE_CLASS_ZERO
from models.src.estimators import build_classifier, model_parameters, save_fitted_model
from models.src.tuning import fit_temporal_classifier, oversample_class_zero


def test_selected_candidate_matches_stored_development_winner():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    winners = pd.read_csv(root / "docs/experiments/multihorizon-grid/development_winners.csv")
    selected = winners.loc[winners.horizon == DEFAULT_HORIZON].squeeze()
    assert selected.model == DEFAULT_MODEL == "lightgbm"
    assert selected.features == "previous"
    assert bool(selected.oversampling) is OVERSAMPLE_CLASS_ZERO is True
    for key, value in json.loads(selected.params).items():
        assert model_parameters()[key] == value


def test_selected_fit_ignores_early_stopping_data_and_exports_native_model(tmp_path):
    X = pd.DataFrame(
        {"f1": np.linspace(-1, 1, 100)}, index=pd.date_range("1980", periods=100, freq="ME")
    )
    y = (X.f1 > -0.6).astype(int)
    xs, ys = oversample_class_zero(X, y)
    first = build_classifier(**model_parameters())
    second = build_classifier(**model_parameters())
    fit_temporal_classifier(first, xs, ys, X, y)
    fit_temporal_classifier(second, xs, ys, X * 10, 1 - y)
    assert first.n_estimators_ == 100
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))
    save_fitted_model(
        first, X, y, horizon=36, oversampling=True, class_zero_weight=1, output_dir=tmp_path
    )
    loaded = Booster(model_file=str(tmp_path / "model/model.txt"))
    np.testing.assert_allclose(loaded.predict(X), first.predict_proba(X)[:, 1])
    metadata = json.loads((tmp_path / "model/metadata.json").read_text())
    assert metadata["features"] == ["f1"]
    assert metadata["classes"] == [0, 1]
    assert metadata["training_rows_before_sampling"] == len(X)
    assert metadata["threshold"] is None
    assert len(metadata["training_sha256"]) == 64


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        model_parameters("typo")


def test_default_pipeline_rejects_legacy_search_before_loading_data(monkeypatch):
    from models import run_pipeline

    monkeypatch.setattr(run_pipeline, "load_raw_dataset", lambda: pytest.fail("Loaded data"))
    with pytest.raises(ValueError, match="--model xgboost"):
        run_pipeline.run_pipeline(do_random_search=True)
