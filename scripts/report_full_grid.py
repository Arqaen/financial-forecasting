"""Audit stored grid predictions and publish reproducible tables and detailed methodology."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from scripts.compare_ensembles import create_targets, load_raw_dataset
from scripts.compare_full_grid import GRIDS, HORIZONS, MODELS, TEST_DATES, market_inputs, metrics

ROOT = Path(__file__).resolve().parents[1]


def table(frame, columns):
    rows = [
        "| " + " | ".join(columns.values()) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            value = row[c]
            if c == "oversampling":
                cells.append("1:1" if value else "Off")
            elif isinstance(value, (float, np.floating)):
                cells.append(
                    f"{value:.2f}" if "roi" in c.lower() or c == "vs_DCA_pp" else f"{value:.3f}"
                )
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    result = pd.read_csv(source / "all_combinations.csv")
    expected = len(HORIZONS) * 3 * 2 * sum(len(list(ParameterGrid(g))) for g in GRIDS.values())
    assert expected == 1476 and len(result) == expected + 72
    assert int(result.selected_by_dev.sum()) == 288
    assert not result.duplicated(
        ["horizon", "features", "oversampling", "model", "candidate"]
    ).any()
    selected_predictions = pd.read_csv(
        source / "selected_predictions.csv.gz", parse_dates=["date"], float_precision="round_trip"
    )
    raw = load_raw_dataset()
    targets = {h: create_targets(raw, h).target for h in HORIZONS}
    fixed_market = market_inputs(36)
    # Recompute every reported test metric from stored probabilities, including rejected candidates.
    for (h, policy, sampling, name), group in result.groupby(
        ["horizon", "features", "oversampling", "model"]
    ):
        market = market_inputs(h)
        if name in MODELS:
            stem = f"h{h}_{policy}_{int(sampling)}_{name}"
            predictions = pd.read_csv(
                source / "checkpoints" / f"{stem}_pred.csv.gz",
                parse_dates=["date"],
                float_precision="round_trip",
            )
            best = group.sort_values(
                ["dev_AP0", "dev_brier", "candidate"], ascending=[False, True, True]
            ).iloc[0]
            assert best.selected_by_dev
        else:
            predictions = selected_predictions.loc[
                selected_predictions.horizon.eq(h)
                & selected_predictions.features.eq(policy)
                & selected_predictions.oversampling.eq(sampling)
                & selected_predictions.model.eq(name)
            ].copy()
            predictions["candidate"] = 0
        for i, row in group.iterrows():
            frame = (
                predictions.loc[predictions.candidate.eq(row.candidate)]
                .set_index("date")
                .sort_index()
            )
            assert np.array_equal(frame.actual, targets[h].loc[frame.index])
            evaluation = frame.loc["2005":]
            assert evaluation.index.equals(TEST_DATES)
            assert np.isfinite(evaluation.p0).all() and evaluation.p0.between(0, 1).all()
            assert (evaluation.alert == (evaluation.p0 > evaluation.threshold_p0)).all()
            for key, value in metrics(evaluation, market).items():
                assert np.isclose(value, row[key], atol=1e-9), (
                    h,
                    policy,
                    sampling,
                    name,
                    row.candidate,
                    key,
                )
            for key, value in metrics(frame.loc[:"1999"]).items():
                assert np.isclose(value, row[f"dev_{key}"], atol=1e-9), (
                    h,
                    policy,
                    sampling,
                    name,
                    key,
                )
            fixed = metrics(evaluation, fixed_market)
            result.loc[i, "roi_fixed36_pct"] = fixed["roi_pct"]
            result.loc[i, "vs_DCA_fixed36_pp"] = fixed["vs_DCA_pp"]
    audit = pd.read_csv(
        source / "feature_audit.csv",
        parse_dates=[
            "origin",
            "fit_end",
            "fit_label_end",
            "early_stop_start",
            "threshold_end",
            "threshold_label_end",
        ],
    )
    assert (audit.fit_label_end < audit.early_stop_start).all()
    assert (audit.threshold_label_end < audit.origin).all()
    selected = result.loc[result.selected_by_dev].copy()
    # This picks features, sampling and model using development scores only, within each horizon.
    eligible = selected.loc[selected.model.ne("selected_single")]
    champion = (
        eligible.sort_values(["dev_AP0", "dev_brier", "model"], ascending=[False, True, True])
        .groupby("horizon", sort=True)
        .head(1)
        .sort_values("horizon")
    )
    retrospective = (
        result.loc[result.model.isin(MODELS)]
        .sort_values(
            ["roi_pct", "AP0", "brier", "model", "candidate"],
            ascending=[False, False, True, True, True],
        )
        .groupby("horizon", sort=True)
        .head(1)
        .sort_values("horizon")
    )
    paired = selected.loc[selected.model.isin(MODELS)].pivot(
        index=["horizon", "features", "model"],
        columns="oversampling",
        values=["AP0", "roi_pct", "F1_0"],
    )
    deltas = pd.DataFrame(
        {
            f"delta_{metric}": paired[metric][True] - paired[metric][False]
            for metric in ["AP0", "roi_pct", "F1_0"]
        }
    ).reset_index()
    output = ROOT / "docs/experiments/multihorizon-grid"
    output.mkdir(parents=True, exist_ok=True)
    for directory in (source, output):
        result.to_csv(directory / "all_combinations.csv", index=False, lineterminator="\n")
        selected.to_csv(directory / "selected_models.csv", index=False, lineterminator="\n")
        champion.to_csv(directory / "development_winners.csv", index=False, lineterminator="\n")
        retrospective.to_csv(
            directory / "retrospective_roi_winners.csv", index=False, lineterminator="\n"
        )
        deltas.to_csv(directory / "paired_sampling_deltas.csv", index=False, lineterminator="\n")
    for filename in ("feature_audit.csv", "manifest.json", "ensemble_validation.csv"):
        (output / filename).write_text(
            (source / filename).read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
    selection_rows = []
    for _, row in audit.loc[
        audit.features.eq("selected8") & audit.origin.ge("2005-01-01")
    ].iterrows():
        selection_rows.extend(dict(horizon=row.horizon, feature=c) for c in json.loads(row.columns))
    frequency = (
        pd.DataFrame(selection_rows)
        .groupby(["horizon", "feature"])
        .size()
        .rename("test_folds_selected")
        .reset_index()
    )
    frequency.sort_values(["horizon", "test_folds_selected"], ascending=[True, False]).to_csv(
        output / "feature_selection_frequency.csv", index=False, lineterminator="\n"
    )
    versions = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm"]
    }
    versions["python"] = platform.python_version()
    (output / "versions.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8", newline="\n"
    )
    columns = dict(
        horizon="Horizon (months)",
        model="Model",
        features="Features",
        oversampling="ROS",
        AP0="Test AP0",
        ROC_AUC="ROC-AUC",
        F1_0="F1_0",
        roi_pct="ROI %",
        vs_DCA_pp="vs DCA (pp)",
        roi_fixed36_pct="ROI, fixed 36m wait %",
    )
    all_columns = {**columns, "brier": "Brier", "precision0": "Precision0", "recall0": "Recall0"}
    baseline = result.dca_roi_pct.iloc[0]
    h36 = champion.loc[champion.horizon.eq(36)].iloc[0]
    max_roi = retrospective.loc[retrospective.roi_pct.idxmax()]
    section = f"""### Multi-horizon grid search: models, features and oversampling

This expanded experiment evaluates **1,476 hyperparameter configurations**, each with its own out-of-sample metrics and ROI. There are **216 model/horizon/feature/sampling groups** (6 × 6 × 3 × 2). Selecting one grid candidate per group gives 216 tuned models; another 36 probability ensembles and 36 single-model selectors bring the selected summary to **288 rows**. The ensemble search evaluates 57 equal-weight subsets per group, or 2,052 development scores; only each group's selected ensemble is evaluated as an ensemble in the test table.

**The table below selects the horizon-specific model, feature policy and sampling policy exclusively by development AP0**, with lower Brier breaking ties. These choices are fixed before the 2005 test begins; test ROI does not choose the models. Across all individual grid candidates, the highest *retrospective* ROI is {max_roi.roi_pct:.2f}% ({max_roi.model}, {int(max_roi.horizon)} months, {max_roi.features}, ROS {'on' if max_roi.oversampling else 'off'}). That is a result selected after observing the test, not evidence that this configuration would have been chosen in advance.

{table(champion, columns)}

Of these six horizon-specific development selections, **{int(champion.vs_DCA_pp.gt(0).sum())} outperform DCA** in the test. The 36-month development choice is {h36.model}, feature policy `{h36.features}`, ROS {'on' if h36.oversampling else 'off'}: AP0 {h36.AP0:.3f} and ROI {h36.roi_pct:.2f}%, versus DCA's {baseline:.2f}%. Choosing the horizon itself after inspecting this table is still a retrospective decision. The retrospective ROI table breaks equal-ROI ties by higher AP0, then lower Brier and deterministic model/grid order.

**Downloads:** [All 1,548 rows, including rejected grid candidates](docs/experiments/multihorizon-grid/all_combinations.csv) · [288 selected configurations](docs/experiments/multihorizon-grid/selected_models.csv) · [Full selected-model tables](docs/experiments/multihorizon-grid/results.md) · [Feature audit by fold](docs/experiments/multihorizon-grid/feature_audit.csv) · [Paired ROS differences](docs/experiments/multihorizon-grid/paired_sampling_deltas.csv).

#### Targets and chronological protocol

Targets are 6, 12, 24, 36, 48 and 60-month forward price returns: class 0 means `Close[t+h] <= Close[t]`. All configurations evaluate the same **192 monthly signals, January 2005–December 2020**, and are valued on **2 January 2026**. Shortening the signal period allows the full 60-month target to mature. These scores and ROIs are therefore **not directly comparable** with the preceding 2005–2022 experiment.

{table(champion, dict(horizon="Horizon (months)", n="Test months", n0="Unfavorable months", prevalence0="Class-0 prevalence / AP reference"))}

The fitting history starts in January 1962. Hyperparameter selection uses pooled annual walk-forward predictions from **1982–1999** (216 months), common to every horizon. Even the 60-month outcome of December 1999 matures before January 2005. Years 2000–2004 are excluded from grid scoring; their observations may enter later fitting windows once their labels mature. The test has 16 annual expanding folds. Both outer and fit-to-validation purges use the target horizon. Each historical training window reserves its last 20% for validation, divided equally between XGBoost early stopping and threshold fitting; fitting ends another horizon before validation starts. The two internal validation halves are adjacent, so their forward outcomes overlap. This does not expose the test, but they are not independent validation samples.

Parameters remain fixed after development selection; estimators and alert thresholds are refitted annually. AP0 uses the actual class-0 probability column. F1/precision/recall and ROI use `p0 > threshold_p0`, where the threshold maximizes **class-1 F1** in that fold's historical threshold window (0.1–0.9 grid, with the existing one-class fallback). Thus the search optimizes ranking, not portfolio ROI or class-0 recall. Test scores for rejected candidates are provided for inspection and must not be treated as an additional tuning set.

#### Feature policies and sampling

- **previous:** the five established candidates: equity risk premium, credit spread, 12-month unemployment change, M2 year-over-year growth and permits year-over-year growth.
- **broad:** the union of these five, the 39 configured macro/valuation candidates and eight horizon-dependent technical candidates (EMA distances/spread, RSI, horizon momentum, 12-month drawdown and momentum change), deduplicated. Each feature must have at least 60 nonmissing observations and nonzero variation in the **unsampled fit window**. The audit records the actual available columns in every fold.
- **selected8:** starts from the broad pool and keeps up to eight variables ranked by absolute Spearman association with the target in the unsampled fit window. Greedy screening rejects a variable with absolute Spearman correlation ≥0.90 to an already selected variable. This is a predefined feature-selection heuristic, not a claim to have found an optimal variable set.

The 60-observation eligibility rule applies to all three policies, including `previous`; it differs from the earlier hybrid experiment's 60% coverage rule. Archived unemployment, M2, permits and credit-spread features replace revised values where archived observations exist. Other variables and archival gaps retain the repository's lagged/revised histories. **This expanded experiment uses one hybrid panel, not a fully point-in-time panel**; it does not repeat the archived-only sensitivity test. Derived features involving those macro series can still depend on revised raw histories. Inherited engineered-variable definitions are retained. Historical revisions and the previous research that informed candidate choice can still bias results.

**Inherited feature-definition limitation:** `equity_risk_premium` is currently `1 / P_E - DGS10`, although [DGS10 is expressed in percent](https://fred.stlouisfed.org/series/DGS10); subtracting a decimal yield consistently would require `DGS10 / 100`. Also, `curve_slope` is `DGS10 - T10Y3M`, even though [T10Y3M already represents the 10-year minus 3-month spread](https://fred.stlouisfed.org/series/T10Y3M), and `earnings_growth_12m` measures P/E changes rather than earnings changes. This experiment retains these numerical definitions for comparison with the existing model; their financial names should not be taken literally. Correcting their definitions would constitute a new experiment and could change the ranking.

Both sampling settings use identical calendar folds and feature selection. ROS retains all fitting rows and duplicates class 0 with replacement to a 1:1 count (seed 42); it is a no-op if class 0 is absent or is already at least as frequent as class 1. Validation and test rows are never resampled. Model seeds and grids are fixed; logistic loss remains in the boosting/logistic models. This comparison does not change operational pipeline defaults.

#### Hyperparameter grids

| Model | Grid | Candidates per group |
| --- | --- | ---: |
| XGBoost | max_depth = 1, 2, 4; min_child_weight = 5, 10 | 6 |
| Logistic L2 | C = 0.01, 0.1, 1, 10, 100 | 5 |
| Extra Trees | max_depth = 2, 4, 8; min_samples_leaf = 4, 8, 16 | 9 |
| Random Forest | max_depth = 2, 4, 8; min_samples_leaf = 4, 8, 16 | 9 |
| HistGradientBoosting | max_leaf_nodes = 3, 7, 15; learning_rate = 0.025, 0.05 | 6 |
| LightGBM | num_leaves = 3, 7, 15; learning_rate = 0.025, 0.05 | 6 |

Other settings use [the six model definitions](scripts/compare_ensembles.py): XGBoost has up to 5,000 trees and 100-round early stopping; forests have 200 trees; HistGradientBoosting and LightGBM have 100 iterations. Logistic regression uses median imputation, scaling and L2; forests use median imputation. These transformations fit only the fitting rows. Ensembles select equal-weight subsets of two to six already tuned models by development AP0/Brier; the single-model selector uses the same criterion within each horizon/feature/sampling group.

#### Economic accounting and metrics

Each portfolio receives **100 units per signal, 19,200 total**. With no alert, that contribution buys at the next trading session's close. An alert holds that contribution in cash at 0% until the first session after the target horizon, then buys. Existing holdings are retained. All holdings use the same final valuation date. `roi_pct = 100 × (wealth / 19200 − 1)` is **cumulative, not annualized**; `vs_DCA_pp` is the difference in ROI percentage points. DCA earns **{baseline:.2f}%** under this calendar. Dividends, costs and taxes are excluded.

Changing the target also changes the cash waiting period in `roi_pct`. To isolate the signal comparison under a common execution rule, `roi_fixed36_pct` and `vs_DCA_fixed36_pp` rerun every set of alerts with a **36-month cash wait**, irrespective of target horizon. They are also valued on 2 January 2026 with the same contributions. All postponed purchases occur before valuation, and cash contributions remain in the accounting throughout.

The CSVs include AP0, AP0/prevalence lift, ROC-AUC, Brier, log loss, class-0 precision/recall/F1, detections, false alarms, precision/recall@24, final wealth and both ROIs. Top-24 statistics are retrospective ranking diagnostics, not executable alert policies. Raw AP0 should not be compared across horizons without considering their different unfavorable-month prevalence. Labels overlap; 192 months do not constitute 192 independent episodes. Multiple configurations share the same test period and test leaders are exploratory, especially after earlier research on these dates. No untouched confirmation period or confidence intervals are claimed.

#### Reproduction and verification

```bash
python -m scripts.compare_full_grid --vintages models/reference/archived_features.csv --out-dir ../estudio-oversampling/grid-multihorizonte --workers 6
python -m scripts.report_full_grid --input ../estudio-oversampling/grid-multihorizonte
python -m pytest tests/test_full_grid.py tests/test_ensembles.py tests/test_tuning.py -q
```

The first command resumes completed model groups only when its code/data fingerprint matches. Use a new output directory for changed inputs. It writes all individual candidate probabilities, selected internal-validation probabilities, feature/date audits and market purchase schedules. Full predictions remain in the companion study directory; compact results are copied into `docs/experiments/multihorizon-grid`. The archived-feature snapshot is bundled in `models/reference/archived_features.csv`. The report command recomputes **every test metric and ROI** from the saved predictions, checks the 1,476 + 72 rows, verifies parameter selection and temporal assertions, and publishes these tables. [Input fingerprint](docs/experiments/multihorizon-grid/manifest.json) · [Library versions](docs/experiments/multihorizon-grid/versions.json).

"""
    (output / "methodology.md").write_text(
        section.replace("(docs/experiments/multihorizon-grid/", "(").replace(
            "(scripts/", "(../../../scripts/"
        ),
        encoding="utf-8",
        newline="\n",
    )
    full = "# Selected configurations\n\nAll metrics are test results for 2005–2020; parameter selection uses 1982–1999. ROI is cumulative. See the repository README for assumptions.\n\n"
    for h, group in selected.groupby("horizon"):
        full += f"## {h}-month target\n\n" + table(group, all_columns) + "\n\n"
    full += (
        "## Highest retrospective ROI within each target\n\nThese are selected using test ROI and are exploratory, not development-selected winners.\n\n"
        + table(retrospective, columns)
        + "\n"
    )
    (output / "results.md").write_text(full, encoding="utf-8", newline="\n")
    print(table(champion, columns))
    print("\nRetrospective ROI leaders:\n" + table(retrospective, columns))
    print("\nVerified all 1,548 rows and published results.")


if __name__ == "__main__":
    main()
