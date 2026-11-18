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

**Downloads:** [All 1,548 rows, including rejected grid candidates](all_combinations.csv) · [288 selected configurations](selected_models.csv) · [Full selected-model tables](results.md) · [Feature audit by fold](feature_audit.csv) · [Paired ROS differences](paired_sampling_deltas.csv).

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

**Historical feature definitions.** The published grid used `1 / P_E - DGS10` for `equity_risk_premium`. Current training uses `1 / P_E - DGS10 / 100` because [DGS10 is expressed in percent](https://fred.stlouisfed.org/series/DGS10). The [corrected LightGBM results](../../../models/artifacts/selected_lightgbm_erp_corrected/verification.json) cover the selected 36-month configuration. In the published grid, `curve_slope` uses `DGS10 - T10Y3M`, and `earnings_growth_12m` measures P/E changes. [T10Y3M is the 10-year minus 3-month spread](https://fred.stlouisfed.org/series/T10Y3M).

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

Other settings use [the six model definitions](../../../scripts/compare_ensembles.py): XGBoost has up to 5,000 trees and 100-round early stopping; forests have 200 trees; HistGradientBoosting and LightGBM have 100 iterations. Logistic regression uses median imputation, scaling and L2; forests use median imputation. These transformations fit only the fitting rows. Ensembles select equal-weight subsets of two to six already tuned models by development AP0/Brier; the single-model selector uses the same criterion within each horizon/feature/sampling group.

#### Economic accounting and metrics

Each portfolio receives **100 units per signal, 19,200 total**. With no alert, that contribution buys at the next trading session's close. An alert holds that contribution in cash at 0% until the first session after the target horizon, then buys. Existing holdings are retained. All holdings use the same final valuation date. `roi_pct = 100 × (wealth / 19200 − 1)` is **cumulative, not annualized**; `vs_DCA_pp` is the difference in ROI percentage points. DCA earns **332.01%** under this calendar. Dividends, costs and taxes are excluded.

Changing the target also changes the cash waiting period in `roi_pct`. To isolate the signal comparison under a common execution rule, `roi_fixed36_pct` and `vs_DCA_fixed36_pp` rerun every set of alerts with a **36-month cash wait**, irrespective of target horizon. They are also valued on 2 January 2026 with the same contributions. All postponed purchases occur before valuation, and cash contributions remain in the accounting throughout.

The CSVs include AP0, AP0/prevalence lift, ROC-AUC, Brier, log loss, class-0 precision/recall/F1, detections, false alarms, precision/recall@24, final wealth and both ROIs. Top-24 statistics are retrospective ranking diagnostics, not executable alert policies. Raw AP0 should not be compared across horizons without considering their different unfavorable-month prevalence. Labels overlap; 192 months do not constitute 192 independent episodes. Multiple configurations share the same test period and test leaders are exploratory, especially after earlier research on these dates. No untouched confirmation period or confidence intervals are claimed.

#### Reproduction and verification

The commands below reproduce the published grid from commit `05770b1`, where `broad` and `selected8` can use lagged housing starts and vehicle sales in levels. The current candidate pool uses their year-over-year growth rates. [The paired Random Forest comparison](../../../scripts/compare_macro_growth.py) uses this historical grid as its baseline and checks that the evaluation windows match.

```bash
python -m scripts.compare_full_grid --vintages models/reference/archived_features.csv --out-dir ../estudio-oversampling/grid-multihorizonte --workers 6
python -m scripts.report_full_grid --input ../estudio-oversampling/grid-multihorizonte
python -m pytest tests/test_full_grid.py tests/test_ensembles.py tests/test_tuning.py -q
```

The first command resumes completed model groups only when its code/data fingerprint matches. Use a new output directory for changed inputs. It writes all individual candidate probabilities, selected internal-validation probabilities, feature/date audits and market purchase schedules. Full predictions remain in the companion study directory; compact results are copied into `docs/experiments/multihorizon-grid`. The archived-feature snapshot is bundled in `models/reference/archived_features.csv`. The report command recomputes **every test metric and ROI** from the saved predictions, checks the 1,476 + 72 rows, verifies parameter selection and temporal assertions, and publishes these tables. [Input fingerprint](manifest.json) · [Library versions](versions.json).
