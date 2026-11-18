"""Regenerate the README figures from the saved LightGBM reference artifacts.

Run from the repository root: python -m scripts.export_lightgbm_figures
Evaluation uses annual out-of-sample predictions; SHAP explains the saved full fit.
"""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import Booster
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from models.src.config import DATA_DIR, DEFAULT_FEATURES
from models.src.data_loader import load_raw_dataset
from models.src.features import (
    add_technical_indicators,
    apply_macro_release_lags,
    compute_macro_and_valuation_features,
    create_targets,
)
from models.src.plots import plot_classification_timeline
from scripts.compare_full_grid import market_inputs, metrics
from scripts.train_selected_model import ARCHIVED_FEATURES, DEFAULT_OUTPUT, ROOT

OUT = ROOT / "docs/images/selected_lightgbm"
BLUE, ORANGE, RED = "#1976a5", "#d28424", "#bf4444"


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name):
    plt.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


def portfolio_history(predictions, market):
    """Mark shares and deferred cash daily using the study's equal-budget purchases."""
    close = (
        pd.read_csv(DATA_DIR / "sp500.csv", skiprows=[1], index_col="Date", parse_dates=True)
        .Close.dropna()
        .sort_index()
    )
    close = close.loc[market.buy_date.min() : market.terminal.iloc[0]]
    contributions = (
        pd.Series(100.0, index=market.buy_date)
        .groupby(level=0)
        .sum()
        .reindex(close.index, fill_value=0)
        .cumsum()
    )
    history = pd.DataFrame({"close": close, "contributed": contributions})
    for name, dates in {
        "lightgbm": market.buy_date.where(~predictions.alert, market.delayed_date),
        "dca": market.buy_date,
    }.items():
        purchases = (
            pd.Series(100.0, index=pd.DatetimeIndex(dates))
            .groupby(level=0)
            .sum()
            .reindex(close.index, fill_value=0)
        )
        shares = (purchases / close).cumsum()
        cash = contributions - purchases.cumsum()
        assert cash.min() >= -1e-8
        history[f"{name}_cash"] = cash
        history[f"{name}_wealth"] = shares * close + cash
        history[f"{name}_roi"] = 100 * (history[f"{name}_wealth"] / contributions - 1)
    return history


def main():
    plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    predictions_path = DEFAULT_OUTPUT / "walk_forward_predictions.csv"
    frame = pd.read_csv(predictions_path, index_col="date", parse_dates=True).loc["2005":"2020"]
    assert frame.index.equals(pd.date_range("2005-01-31", "2020-12-31", freq="ME"))
    assert frame.alert.equals(frame.p0 > frame.threshold_p0)
    report = json.loads((DEFAULT_OUTPUT / "verification.json").read_text())
    market = market_inputs(36)
    scores = metrics(frame, market)
    for key, value in scores.items():
        np.testing.assert_allclose(value, report["test"][key], rtol=1e-10)
    negative = frame.actual.eq(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    fpr, tpr, _ = roc_curve(negative, frame.p0)
    precision, recall, _ = precision_recall_curve(negative, frame.p0)
    axes[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC-AUC = {scores['ROC_AUC']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", label="Random ranking")
    axes[0].set(
        xlabel="False-positive rate (class 0)", ylabel="Recall (class 0)", title="ROC curve"
    )
    axes[1].step(
        recall, precision, where="post", color=BLUE, lw=2, label=f"AP = {scores['AP0']:.3f}"
    )
    axes[1].axhline(
        negative.mean(), ls="--", color=ORANGE, label=f"Prevalence = {negative.mean():.3f}"
    )
    axes[1].set(
        xlabel="Recall (class 0)", ylabel="Precision (class 0)", title="Precision–recall curve"
    )
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(alpha=0.15)
    fig.suptitle(
        "LightGBM · 36 months · 1:1 oversampling\nOut-of-sample evaluation: 2005–2020 (192 months)",
        fontsize=14,
    )
    save("roc_pr_class0.png")

    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    for date in frame.index[negative]:
        ax.axvspan(date - pd.offsets.MonthEnd(), date, color=RED, alpha=0.12, lw=0)
    ax.plot(frame.index, frame.p0, color=BLUE, label="P(class 0), uncalibrated", lw=1.8)
    ax.step(
        frame.index,
        frame.threshold_p0,
        where="post",
        color=ORANGE,
        ls="--",
        label="Past-validation alert threshold",
    )
    for mask, color, label in [
        (frame.alert & negative, RED, "Correct alert"),
        (frame.alert & ~negative, ORANGE, "False alarm"),
    ]:
        ax.scatter(frame.index[mask], frame.p0[mask], color=color, s=36, zorder=3, label=label)
    ax.set(
        ylim=(0, 1.03),
        ylabel="Class-0 probability / threshold",
        xlabel="Signal date",
        title="LightGBM · out-of-sample risk signals, 2005–2020\nShaded months have a non-positive 36-month forward price return",
    )
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    save("classification.png")

    raw = load_raw_dataset()
    labeled = create_targets(raw, 36).loc[frame.index]
    np.testing.assert_array_equal(labeled.target.astype(int), frame.actual)
    timeline = pd.DataFrame(
        {
            "date": frame.index,
            "actual": frame.actual.to_numpy(),
            "pred": (~frame.alert).astype(int).to_numpy(),
            "proba_up": (1 - frame.p0).to_numpy(),
            "close_t": labeled.Close.to_numpy(),
        }
    )
    plot_classification_timeline(
        timeline,
        out_path=OUT / "walk_forward_classification.png",
        title="LightGBM · walk-forward 2005–2020 · 36-month target · 1:1 oversampling",
    )

    matrix = confusion_matrix(frame.actual, (~frame.alert).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0)
    labels = [["Detected negatives", "Missed negatives"], ["False alarms", "Correct positives"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i,j]}\n{labels[i][j]}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if matrix[i, j] > matrix.max() / 2 else "#183044",
            )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["0 · Alert", "1 · No alert"],
        yticklabels=["0 · Non-positive", "1 · Positive"],
        xlabel="Predicted class",
        ylabel="Actual 36-month return",
        title=f"LightGBM · confusion matrix, 2005–2020\nClass-0 F1 = {scores['F1_0']:.3f} · 192 OOS months",
    )
    save("confusion_matrix.png")

    history = portfolio_history(frame, market)
    np.testing.assert_allclose(history.lightgbm_roi.iloc[-1], scores["roi_pct"], rtol=1e-10)
    np.testing.assert_allclose(history.dca_roi.iloc[-1], scores["dca_roi_pct"], rtol=1e-10)
    np.testing.assert_allclose(history.lightgbm_wealth.iloc[-1], scores["wealth"], rtol=1e-10)
    history.to_csv(OUT / "roi_history.csv", index_label="date")
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(
        history.index,
        history.lightgbm_roi,
        color=BLUE,
        lw=1.8,
        label=f"LightGBM: {scores['roi_pct']:.2f}%",
    )
    ax.plot(
        history.index,
        history.dca_roi,
        color=ORANGE,
        ls="--",
        lw=1.8,
        label=f"DCA: {scores['dca_roi_pct']:.2f}%",
    )
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(
        ylabel="Cumulative ROI on contributions (%)",
        xlabel="Portfolio valuation date",
        title="LightGBM vs DCA · equal monthly contributions\nSignals: 2005–2020 · valuation through 2 January 2026",
    )
    ax.legend(loc="upper left", title="Cash earns 0%; flagged contributions delayed 36 months")
    ax.grid(alpha=0.15)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 1))
    fig.text(
        0.5,
        0.015,
        "Next-session purchases · 19,200 contributed per strategy · excludes dividends, fees and taxes",
        ha="center",
        fontsize=10,
    )
    save("investment_strategies.png")

    metadata = json.loads((DEFAULT_OUTPUT / "model/metadata.json").read_text())
    model_path = DEFAULT_OUTPUT / "model/model.txt"
    assert fingerprint(model_path) == metadata["model_sha256"]
    assert fingerprint(ARCHIVED_FEATURES) == report["archived_features_sha256"]
    values = compute_macro_and_valuation_features(
        apply_macro_release_lags(add_technical_indicators(raw, 36)), 36
    )[DEFAULT_FEATURES].replace([np.inf, -np.inf], np.nan)
    archived = pd.read_csv(ARCHIVED_FEATURES, index_col="fecha", parse_dates=True)
    values.update(archived[archived.columns.intersection(DEFAULT_FEATURES)])
    y = create_targets(raw, 36).target.loc["1962":"2020"].astype(int)
    X = values.loc[y.index, metadata["features"]]
    training_hash = hashlib.sha256(
        pd.concat([X, y.rename("target")], axis=1).to_csv(lineterminator="\n").encode()
    ).hexdigest()
    assert training_hash == metadata["training_sha256"]
    model = Booster(model_file=str(model_path))
    contributions = model.predict(X, pred_contrib=True)
    np.testing.assert_allclose(
        contributions.sum(axis=1), model.predict(X, raw_score=True), atol=1e-10
    )
    np.random.seed(42)
    shap.summary_plot(contributions[:, :-1], X, show=False, plot_size=(10, 5.5))
    plt.title(
        "LightGBM · full reference fit, 1962–2020\nIn-sample feature attribution; output = log-odds of class 1",
        fontsize=12,
        pad=22,
    )
    plt.xlabel("SHAP contribution to class-1 log-odds")
    save("shap_summary_cls.png")
    shap.plots.waterfall(
        shap.Explanation(
            values=contributions[-1, :-1],
            base_values=contributions[-1, -1],
            data=X.iloc[-1].to_numpy(),
            feature_names=list(X),
        ),
        show=False,
    )
    plt.gcf().suptitle(
        "LightGBM · 31 December 2020\nLast labeled training row; in-sample explanation, not a live forecast",
        fontsize=11,
        y=1.08,
    )
    save("shap_last_prediction_cls.png")

    images = []
    for path in sorted(OUT.glob("*.png")):
        with Image.open(path) as img:
            images.append(
                {
                    "file": f"selected_lightgbm/{path.name}",
                    "width": img.width,
                    "height": img.height,
                    "sha256": fingerprint(path),
                }
            )
    ledger_path = ROOT / "docs/images/sources.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["lightgbm_runs"] = [
        {
            "model": "lightgbm_h36_previous_ros",
            "generator": "scripts/export_lightgbm_figures.py",
            "test_period": "2005-2020",
            "shap_training_period": "1962-2020",
            "predictions": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
            "predictions_sha256": fingerprint(predictions_path),
            "model_sha256": fingerprint(model_path),
            "training_sha256": training_hash,
            "test_metrics": scores,
            "roi_history_sha256": fingerprint(OUT / "roi_history.csv"),
            "figures": images,
        }
    ]
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Exported {len(images)} verified LightGBM figures to {OUT}")


if __name__ == "__main__":
    main()
