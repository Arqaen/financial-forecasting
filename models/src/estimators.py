"""Shared estimator configuration for evaluation, final fitting and model export."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import (
    DEFAULT_MODEL,
    FINAL_FIXED_PARAMS,
    FIXED_PARAMS_BASE,
    LIGHTGBM_PARAMS,
    MANUAL_PARAMS_BASE,
)


def model_parameters(
    model_name: str = DEFAULT_MODEL, *, final: bool = False, overrides: Optional[Dict] = None
) -> Dict:
    if model_name == "lightgbm":
        params = dict(LIGHTGBM_PARAMS)
    elif model_name == "xgboost":
        params = {**(FINAL_FIXED_PARAMS if final else FIXED_PARAMS_BASE), **MANUAL_PARAMS_BASE}
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return {**params, **(overrides or {})}


def build_classifier(model_name: str = DEFAULT_MODEL, **params: Any) -> Any:
    if model_name == "lightgbm":
        return LGBMClassifier(**params)
    if model_name == "xgboost":
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def save_fitted_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: int,
    oversampling: bool,
    class_zero_weight: float,
    output_dir: Path,
) -> None:
    """Export a native model and its input contract, independent of MLflow."""
    directory = output_dir / "model"
    directory.mkdir(parents=True, exist_ok=True)
    family = "lightgbm" if isinstance(model, LGBMClassifier) else "xgboost"
    path = directory / ("model.txt" if family == "lightgbm" else "model.ubj")
    if family == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(path)
    metadata = {
        "model": family,
        "library_version": version(family),
        "horizon_months": horizon,
        "features": list(X.columns),
        "classes": model.classes_.tolist(),
        "probability": "P(target=1); P(target=0)=1-P(target=1)",
        "target": "1 if Close(t+h) > Close(t), otherwise 0",
        "parameters": model.get_params(),
        "oversampling": oversampling,
        "oversampling_seed": 42,
        "class_zero_weight": class_zero_weight,
        "training_start": str(X.index.min()),
        "training_end": str(X.index.max()),
        "training_rows_before_sampling": len(X),
        "training_class_counts_before_sampling": y.value_counts().to_dict(),
        "training_sha256": hashlib.sha256(
            pd.concat([X, y.rename("target")], axis=1).to_csv(lineterminator="\n").encode()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "threshold": None,
        "usage": "Full-history fit. Probabilities are uncalibrated. Select an operating threshold on past-only validation; final-fit diagnostics are in-sample.",
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
