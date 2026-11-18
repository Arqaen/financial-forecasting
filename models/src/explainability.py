from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False

from .config import (
    CLASS_ZERO_WEIGHT,
    DEFAULT_HORIZON,
    DEFAULT_MODEL,
    METRICS_DIR,
    OVERSAMPLE_CLASS_ZERO,
    TOP_SHAP_FEATURES,
)
from .estimators import build_classifier, model_parameters, save_fitted_model
from .metrics import binary_logloss, class_zero_metrics, precision_at_k
from .tuning import oversample_class_zero, training_sample_weights


def train_final_model_and_explain(
    df: pd.DataFrame,
    features: List[str],
    *,
    horizon: int = DEFAULT_HORIZON,
    class_zero_weight: float = CLASS_ZERO_WEIGHT,
    oversampling: bool = OVERSAMPLE_CLASS_ZERO,
    model_name: str = DEFAULT_MODEL,
    top_shap_features: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[Any, float, Dict]:
    """Train the selected final model on the complete historical dataset and generate SHAP explainability plots.

    Args:
        df: Prepared modeling DataFrame.
        features: Predictive feature columns.
        horizon: Forecast horizon in months.
        class_zero_weight: Relative weight of negative training examples.
        top_shap_features: Subset of features for which to produce SHAP dependence interaction plots.
        out_dir: Output directory for saving SHAP plots.

    Returns:
        Tuple containing (trained_model, last_training_row_probability, diagnostic_metrics_dict).
    """
    output_dir = out_dir or METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    features_to_plot = top_shap_features or TOP_SHAP_FEATURES

    X = df[features]
    y = df["target"].astype(int)

    final_model = build_classifier(model_name, **model_parameters(model_name, final=True))
    X_fit, y_fit = oversample_class_zero(X, y) if oversampling else (X, y)
    final_model.fit(X_fit, y_fit, sample_weight=training_sample_weights(y_fit, class_zero_weight))

    save_fitted_model(
        final_model,
        X,
        y,
        horizon=horizon,
        oversampling=oversampling,
        class_zero_weight=class_zero_weight,
        output_dir=output_dir,
    )

    last_X = X.iloc[[-1]]
    final_proba = float(final_model.predict_proba(last_X)[:, 1][0])
    last_date = X.index[-1]

    # In-sample Diagnostics
    final_proba_all = final_model.predict_proba(X)[:, 1]
    final_logloss = float(binary_logloss(y, final_proba_all))
    final_brier = float(brier_score_loss(y, final_proba_all))
    final_precision_top20 = precision_at_k(y, final_proba_all, top_frac=0.2)
    final_auc = float(roc_auc_score(y, final_proba_all)) if len(np.unique(y)) > 1 else float("nan")
    zero_metrics = class_zero_metrics(y, final_proba_all, final_model.predict(X))

    hist_mean = float(np.nanmean(final_proba_all))
    hist_std = float(np.nanstd(final_proba_all, ddof=0))
    final_proba_percentile = float(np.mean(final_proba_all <= final_proba) * 100.0)
    final_proba_z = float((final_proba - hist_mean) / hist_std) if hist_std > 0 else float("nan")

    diagnostics = {
        "last_date": last_date,
        "final_proba": final_proba,
        "in_sample_auc": final_auc,
        "in_sample_AP_0": zero_metrics["AP_0"],
        "in_sample_recall_0": zero_metrics["recall_0"],
        "in_sample_logloss": final_logloss,
        "in_sample_brier": final_brier,
        "precision_top20": final_precision_top20,
        "historical_percentile": final_proba_percentile,
        "historical_z_score": final_proba_z,
    }

    print("\n" + "=" * 60)
    print("FINAL MODEL & IN-SAMPLE DIAGNOSTICS")
    print("=" * 60)
    print(f"Última fecha: {last_date.date() if hasattr(last_date, 'date') else last_date}")
    print(f"P(sube) en {horizon} meses: {final_proba:.4f}")
    print(
        f"Percentil histórico de P(sube): {final_proba_percentile:.1f}% (z-score: {final_proba_z:.2f})"
    )
    print(
        f"In-sample ROC-AUC: {final_auc:.4f} | LogLoss: {final_logloss:.4f} | Brier: {final_brier:.4f}"
    )

    # ==========================================
    # SHAP Tree Explanations
    # ==========================================
    if not _HAS_SHAP:
        print(
            "[SHAP] Warning: shap library is not installed in the active environment. Skipping SHAP plots."
        )
        return final_model, final_proba, diagnostics

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_pos = shap_values[1]
        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        shap_values_pos = shap_values
        expected_value = explainer.expected_value

    # 1. SHAP Beeswarm Summary Plot
    plt.figure()
    shap.summary_plot(shap_values_pos, X, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_cls.png", dpi=160)
    plt.close()

    # 2. SHAP Global Feature Importance Bar Chart
    plt.figure()
    shap.summary_plot(shap_values_pos, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_bar_cls.png", dpi=160)
    plt.close()

    # 3. SHAP Dependence Interaction Plots for Top Features
    for feature in features_to_plot:
        if feature not in X.columns:
            continue
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(
            feature,
            shap_values_pos,
            X,
            interaction_index="auto",
            show=False,
        )
        fname = f"shap_dependence_cls_{feature}.png"
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=120)
        plt.close()

    # 4. SHAP Waterfall Decomposition for Latest Observation
    shap_values_last = explainer.shap_values(last_X)
    if isinstance(shap_values_last, list) and len(shap_values_last) == 2:
        shap_values_last_pos = shap_values_last[1][0]
    else:
        shap_values_last_pos = shap_values_last[0]

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_last_pos,
            base_values=expected_value,
            data=last_X.iloc[0],
            feature_names=X.columns,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_last_prediction_cls.png", dpi=160)
    plt.close()

    print(f"SHAP explanation figures successfully saved in: {output_dir}\n")

    return final_model, final_proba, diagnostics
