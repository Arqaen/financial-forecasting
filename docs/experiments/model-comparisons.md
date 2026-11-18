# Model comparison archive

Historical detailed comparison tables, retained for audit. The current selected model is described in the [project README](../../README.md#selected-model-and-experiments).

### Oversampling and model comparison

This section records the earlier fixed-parameter, 36-month comparison. The multi-horizon grid search below uses a different evaluation calendar and feature eligibility rule; its results are reported separately.

**Random Forest with 1:1 oversampling ranked risk best in the hybrid panel.** It placed 17 of the 34 unfavorable months among its 24 highest risk scores (AP₀ 0.577). **XGBoost without oversampling gave the best purchase alerts** under the project's historical threshold rule. It detected 19 unfavorable months with six false alarms (F1₀ 0.644) and earned 317.57% cumulative ROI. Buying every month earned 302.28%. In the archived-only panel, every configuration had zero hits among its top 24 and finished below the monthly purchase benchmark.

#### Training and evaluation

The operational pipeline uses **XGBoost with binary logistic loss and 1:1 random oversampling**. After the chronological split, the sampler keeps every fit row and draws extra class-0 rows with replacement until their count matches class 1. It uses seed 42 and equal per-row weights by default. Early stopping, threshold selection and testing retain the observed frequencies. The paired control uses the original fit rows. [Sampler](../../models/src/tuning.py) · [Pipeline defaults](../../models/src/config.py)

The six candidates are depth-2 XGBoost, standardized L2 logistic regression, Extra Trees, HistGradientBoosting, Random Forest and LightGBM. Both forests use 200 trees, maximum depth 4 and at least eight samples per leaf. HistGradientBoosting and LightGBM use up to seven leaves, 100 boosting iterations and a learning rate of 0.05. Each annual ensemble averages the class-0 probabilities of two to six candidates with equal weights. It selects members by AP₀ on earlier walk-forward predictions whose 36-month outcomes have already matured, using Brier score to break ties. The single-model selector applies the same rule to individual candidates. Until both classes have matured, the ensemble uses all six models and the selector uses XGBoost. [Model definitions and selection](../../scripts/compare_ensembles.py)

Both panels evaluate the same **216 monthly forecast origins from January 2005 to December 2022**, in 18 annual blocks with a 36-month purge. Class 0 means the index closes at or below its starting price 36 months later. All 34 such origins occur in 2005–2008. Adjacent labels share 35 months of returns, which limits what this episode establishes about future downturns.

- **Hybrid panel.** The coverage filter uses only history before each test block and requires more than 60% coverage. Archived releases supply unemployment, M2, permits and credit spreads where available. Earlier gaps and the P/E component of the equity risk premium use the repository's historical series, which may contain revisions.
- **Archived-only panel.** A feature needs at least 60 archived observations in the fit block. Only unemployment and M2 qualify in 2005–2008. The change in scores across panels reflects both the available inputs and the selection rule.

#### Reading the scores

AP₀ (average precision) and ROC-AUC use the probability assigned to class 0. Higher values indicate better ranking. Brier is the mean squared probability error and lower is better. Hits@24 counts unfavorable months among the 24 highest scores over the evaluation period, with earlier dates breaking ties. This is a retrospective ranking diagnostic. F1₀, detections and false alarms use an alert threshold set on historical validation data by maximizing F1 for class 1. A false alarm postpones a purchase whose 36-month return was positive.

Cumulative ROI is `100 × (final wealth / 21,600 − 1)`. The simulation contributes 100 units for each of the 216 signals. It buys at the close of the next trading session, or holds that contribution in cash at 0% after an alert and invests it on the first session after the 36-month horizon. All portfolios are valued on **2 January 2026** using index prices, excluding dividends, fees and taxes. Unconditional monthly purchases finish at **86,892.71**. The best hybrid configuration finishes at **90,195.14**, a gain of **15.29 percentage points of ROI** or **3.80% in final wealth** over that benchmark.

#### Hybrid panel results

| Model | Oversampling | AP₀ | ROC-AUC | Brier | F1₀ | Hits@24 | Detected / false alarms | ROI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | Off | 0.171 | 0.511 | 0.144 | 0.644 | 2 | 19 / 6 | 317.57% |
| XGBoost | 1:1 | 0.299 | 0.652 | 0.187 | 0.169 | 8 | 7 / 42 | 285.76% |
| Logistic regression (L2) | Off | 0.121 | 0.335 | 0.347 | 0.365 | 0 | 19 / 51 | 306.10% |
| Logistic regression (L2) | 1:1 | 0.116 | 0.301 | 0.550 | 0.158 | 0 | 13 / 118 | 275.98% |
| Extra Trees | Off | 0.149 | 0.495 | 0.135 | 0.522 | 0 | 24 / 34 | 306.30% |
| Extra Trees | 1:1 | 0.326 | 0.788 | 0.223 | 0.413 | 4 | 26 / 66 | 297.22% |
| HistGradientBoosting | Off | 0.258 | 0.712 | 0.134 | 0.161 | 6 | 5 / 23 | 293.76% |
| HistGradientBoosting | 1:1 | 0.349 | 0.782 | 0.133 | 0.179 | 9 | 7 / 37 | 286.22% |
| Random Forest | Off | 0.163 | 0.543 | 0.146 | 0.468 | 0 | 22 / 38 | 297.99% |
| Random Forest | 1:1 | 0.577 | 0.880 | 0.154 | 0.337 | 17 | 17 / 50 | 291.41% |
| LightGBM | Off | 0.274 | 0.706 | 0.130 | 0.133 | 7 | 4 / 22 | 293.24% |
| LightGBM | 1:1 | 0.314 | 0.745 | 0.141 | 0.182 | 8 | 7 / 36 | 287.64% |
| Probability ensemble | Off | 0.152 | 0.510 | 0.148 | 0.391 | 0 | 17 / 36 | 296.35% |
| Probability ensemble | 1:1 | 0.548 | 0.870 | 0.180 | 0.323 | 16 | 16 / 49 | 290.09% |
| Historical single-model selector | Off | 0.171 | 0.511 | 0.144 | 0.644 | 2 | 19 / 6 | 317.57% |
| Historical single-model selector | 1:1 | 0.577 | 0.880 | 0.154 | 0.337 | 17 | 17 / 50 | 291.41% |

#### Archived-only panel results

<details>
<summary>Show the 16 archived-only configurations</summary>

| Model | Oversampling | AP₀ | ROC-AUC | Brier | F1₀ | Hits@24 | Detected / false alarms | ROI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | Off | 0.127 | 0.274 | 0.207 | 0.000 | 0 | 0 / 18 | 298.07% |
| XGBoost | 1:1 | 0.098 | 0.123 | 0.333 | 0.229 | 0 | 11 / 51 | 297.40% |
| Logistic regression (L2) | Off | 0.098 | 0.168 | 0.229 | 0.000 | 0 | 0 / 23 | 297.59% |
| Logistic regression (L2) | 1:1 | 0.097 | 0.155 | 0.297 | 0.000 | 0 | 0 / 37 | 288.22% |
| Extra Trees | Off | 0.111 | 0.296 | 0.146 | 0.000 | 0 | 0 / 24 | 288.76% |
| Extra Trees | 1:1 | 0.104 | 0.240 | 0.221 | 0.000 | 0 | 0 / 51 | 280.47% |
| HistGradientBoosting | Off | 0.122 | 0.339 | 0.234 | 0.000 | 0 | 0 / 26 | 296.77% |
| HistGradientBoosting | 1:1 | 0.115 | 0.286 | 0.277 | 0.000 | 0 | 0 / 30 | 295.51% |
| Random Forest | Off | 0.111 | 0.283 | 0.195 | 0.000 | 0 | 0 / 10 | 299.40% |
| Random Forest | 1:1 | 0.110 | 0.258 | 0.248 | 0.000 | 0 | 0 / 30 | 295.87% |
| LightGBM | Off | 0.107 | 0.225 | 0.239 | 0.000 | 0 | 0 / 28 | 296.69% |
| LightGBM | 1:1 | 0.111 | 0.263 | 0.279 | 0.000 | 0 | 0 / 35 | 294.81% |
| Probability ensemble | Off | 0.098 | 0.176 | 0.217 | 0.000 | 0 | 0 / 24 | 296.85% |
| Probability ensemble | 1:1 | 0.098 | 0.165 | 0.247 | 0.000 | 0 | 0 / 47 | 285.21% |
| Historical single-model selector | Off | 0.098 | 0.168 | 0.229 | 0.000 | 0 | 0 / 23 | 297.59% |
| Historical single-model selector | 1:1 | 0.097 | 0.155 | 0.297 | 0.000 | 0 | 0 / 37 | 288.22% |

</details>

#### Reproducing the comparison

From the repository root, using the companion study's vintage CSV and economic simulation script:

```bash
python -m scripts.compare_ensembles --both --vintages ../estudio-oversampling/fuentes/variables_pit.csv --out-dir ../estudio-oversampling/comparacion-seis-modelos
python ../estudio-oversampling/comparacion-seis-modelos/comparar_decisiones.py
```

The first command writes `metrics.csv` and `predictions.csv`, including each forecast's probability, alert threshold, members and features. The second joins the resulting alerts to purchase prices and writes `comparacion_2005_2022.csv` with final wealth and the monthly purchase benchmark. The commands above assume the sibling `estudio-oversampling` workspace directory. With a standalone checkout, `python -m scripts.compare_ensembles --both` runs the comparison on the repository's revised series. Use `--oversampling` or `--no-oversampling` for a single policy.

### Multi-horizon grid search: models, features and oversampling

This expanded experiment evaluates **1,476 hyperparameter configurations**, each with its own out-of-sample metrics and ROI. There are **216 model/horizon/feature/sampling groups** (6 × 6 × 3 × 2). Selecting one grid candidate per group gives 216 tuned models; another 36 probability ensembles and 36 single-model selectors bring the selected summary to **288 rows**. The ensemble search evaluates 57 equal-weight subsets per group, or 2,052 development scores; only each group's selected ensemble is evaluated as an ensemble in the test table.

**The table below selects the horizon-specific model, feature policy and sampling policy exclusively by development AP0**, with lower Brier breaking ties. These choices are fixed before the 2005 test begins; test ROI does not choose the models. Across all individual grid candidates, the highest *retrospective* ROI is 354.87% (hist_gb, 36 months, previous, ROS off). That is a result selected after observing the test, not evidence that this configuration would have been chosen in advance.

| Horizon (months) | Model | Features | ROS | Test AP0 | ROC-AUC | F1_0 | ROI % | vs DCA (pp) | ROI, fixed 36m wait % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | ensemble | broad | 1:1 | 0.268 | 0.529 | 0.000 | 330.85 | -1.17 | 331.06 |
| 12 | ensemble | selected8 | 1:1 | 0.172 | 0.460 | 0.000 | 330.15 | -1.86 | 333.29 |
| 24 | ensemble | previous | 1:1 | 0.109 | 0.348 | 0.000 | 324.69 | -7.32 | 322.10 |
| 36 | lightgbm | previous | 1:1 | 0.747 | 0.788 | 0.687 | 351.58 | 19.57 | 351.58 |
| 48 | ensemble | selected8 | 1:1 | 0.252 | 0.581 | 0.530 | 325.40 | -6.61 | 327.62 |
| 60 | ensemble | selected8 | 1:1 | 0.676 | 0.930 | 0.394 | 321.31 | -10.70 | 327.36 |

Of these six horizon-specific development selections, **1 outperform DCA** in the test. The 36-month development choice is lightgbm, feature policy `previous`, ROS on: AP0 0.747 and ROI 351.58%, versus DCA's 332.01%. Choosing the horizon itself after inspecting this table is still a retrospective decision. The retrospective ROI table breaks equal-ROI ties by higher AP0, then lower Brier and deterministic model/grid order.

**Downloads:** [All 1,548 rows, including rejected grid candidates](../experiments/multihorizon-grid/all_combinations.csv) · [288 selected configurations](../experiments/multihorizon-grid/selected_models.csv) · [Full selected-model tables](../experiments/multihorizon-grid/results.md) · [Feature audit by fold](../experiments/multihorizon-grid/feature_audit.csv) · [Paired ROS differences](../experiments/multihorizon-grid/paired_sampling_deltas.csv).

#### Targets and chronological protocol

Targets are 6, 12, 24, 36, 48 and 60-month forward price returns: class 0 means `Close[t+h] <= Close[t]`. All configurations evaluate the same **192 monthly signals, January 2005–December 2020**, and are valued on **2 January 2026**. Shortening the signal period allows the full 60-month target to mature. These scores and ROIs are therefore **not directly comparable** with the preceding 2005–2022 experiment.

| Horizon (months) | Test months | Unfavorable months | Class-0 prevalence / AP reference |
| --- | --- | --- | --- |
| 6 | 192 | 47 | 0.245 |
| 12 | 192 | 36 | 0.188 |
| 24 | 192 | 28 | 0.146 |
| 36 | 192 | 34 | 0.177 |
| 48 | 192 | 38 | 0.198 |
| 60 | 192 | 27 | 0.141 |

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

Other settings use [the six model definitions](../../scripts/compare_ensembles.py): XGBoost has up to 5,000 trees and 100-round early stopping; forests have 200 trees; HistGradientBoosting and LightGBM have 100 iterations. Logistic regression uses median imputation, scaling and L2; forests use median imputation. These transformations fit only the fitting rows. Ensembles select equal-weight subsets of two to six already tuned models by development AP0/Brier; the single-model selector uses the same criterion within each horizon/feature/sampling group.

#### Economic accounting and metrics

Each portfolio receives **100 units per signal, 19,200 total**. With no alert, that contribution buys at the next trading session's close. An alert holds that contribution in cash at 0% until the first session after the target horizon, then buys. Existing holdings are retained. All holdings use the same final valuation date. `roi_pct = 100 × (wealth / 19200 − 1)` is **cumulative, not annualized**; `vs_DCA_pp` is the difference in ROI percentage points. DCA earns **332.01%** under this calendar. Dividends, costs and taxes are excluded.

Changing the target also changes the cash waiting period in `roi_pct`. To isolate the signal comparison under a common execution rule, `roi_fixed36_pct` and `vs_DCA_fixed36_pp` rerun every set of alerts with a **36-month cash wait**, irrespective of target horizon. They are also valued on 2 January 2026 with the same contributions. All postponed purchases occur before valuation, and cash contributions remain in the accounting throughout.

The CSVs include AP0, AP0/prevalence lift, ROC-AUC, Brier, log loss, class-0 precision/recall/F1, detections, false alarms, precision/recall@24, final wealth and both ROIs. Top-24 statistics are retrospective ranking diagnostics, not executable alert policies. Raw AP0 should not be compared across horizons without considering their different unfavorable-month prevalence. Labels overlap; 192 months do not constitute 192 independent episodes. Multiple configurations share the same test period and test leaders are exploratory, especially after earlier research on these dates. No untouched confirmation period or confidence intervals are claimed.

#### Reproduction and verification

```bash
python -m scripts.compare_full_grid --vintages ../estudio-oversampling/fuentes/variables_pit.csv --out-dir ../estudio-oversampling/grid-multihorizonte --workers 6
python -m scripts.report_full_grid --input ../estudio-oversampling/grid-multihorizonte
python -m pytest tests/test_full_grid.py tests/test_ensembles.py tests/test_tuning.py -q
```

The first command resumes completed model groups only when its code/data fingerprint matches. Use a new output directory for changed inputs. It writes all individual candidate probabilities, selected internal-validation probabilities, feature/date audits and market purchase schedules. Full predictions remain in the companion study directory; compact results are copied into `docs/experiments/multihorizon-grid`. A standalone checkout also needs the external archived-feature CSV used by this command. The report command recomputes **every test metric and ROI** from the saved predictions, checks the 1,476 + 72 rows, verifies parameter selection and temporal assertions, and publishes these tables. [Input fingerprint](../experiments/multihorizon-grid/manifest.json) · [Library versions](../experiments/multihorizon-grid/versions.json).

### Historical XGBoost baseline without oversampling

This run uses the five default features, depth-2 XGBoost trees, equal class weights, and a probability threshold selected on historical validation data to maximize F1 for class 1. The complete-case modeling data starts in 1962; after the initial training period, 41 annual walk-forward folds evaluate January 1982 through December 2022.

![Walk-forward class predictions, S&P 500 prices and upward probabilities for shallow-weight1-f1_1](../images/shallow_weight1_f1_1/classification.png)

*The upper panel shows the predicted class alongside the S&P 500 closing price; green points mark correct predictions and red points mark errors. The lower panel shows the probability of a positive 36-month return. These are pooled walk-forward predictions, not in-sample fitted results.*

![Walk-forward confusion matrix for shallow-weight1-f1_1](../images/shallow_weight1_f1_1/confusion_matrix.png)

| Class 0 detection metric | Pooled walk-forward result |
| :--- | ---: |
| Evaluated months | 492 |
| Actual negative months | 71 (14.4%) |
| Detected negatives | 32 of 71 |
| Missed negatives | 39 |
| False alarms for class 0 (actual 1, predicted 0) | 34 |
| Recall₀ | 45.1% |
| Precision₀ | 48.5% |
| F1₀ | 0.467 |
| AP₀ | 0.307 |

*Rows are actual classes and columns are predictions. TN/FP/FN/TP use class 1 as the positive label, so the 34 false alarms for class 0 appear in the FN cell. These pooled metrics differ from averages across individual folds. MLflow experiment: `class-zero-policies-h36`; run ID: `b33cfa49d6a54d2e9bc40d3900897558`.*

> [!WARNING]
> **Known Limitations:**
> Monthly 36-month forward labels overlap, so the 492 evaluated months are not independent observations. Reported metrics are point estimates without confidence intervals and should be read as directional, not conclusive. Feature selection was informed by full-sample analysis, so walk-forward results are optimistic.

---
