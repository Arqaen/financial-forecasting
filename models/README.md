# Financial Machine Learning & Quantitative Prediction Engine

A time-aware, modular machine learning research framework for predicting macroeconomic regimes, financial market directions, and simulating quantitative investment strategies with temporal purging and explicit data-revision limitations.

The default estimator is **LightGBM, 36 months, five features and 1:1 random oversampling**. See the [selection rationale and results](../README.md#selected-model-and-experiments). `python -m scripts.train_selected_model` evaluates and refits this configuration with corrected ERP units. Add `--historical-erp` to reproduce the published experiment. The general pipeline retains XGBoost through `--model xgboost`; random search is available only for that legacy estimator.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data["1. Data Ingestion & Alignment"]
        A[Raw CSVs: 28 Market & Macro Series] --> B[to_monthly_last Resampling]
        B --> C[Consolidated Monthly DataFrame]
    end

    subgraph Features["2. Feature Engineering & Target Labelling"]
        C --> D[Publication Release Lags: GDP, M2, Housing...]
        D --> E[Technical Indicators: EMAs, RSI, ROC]
        E --> F[Macro Ratios: ERP, Spreads, Liquidity Impulse]
        F --> G[Target Construction: Binary Direction & Log Returns]
    end

    subgraph Validation["3. Temporal Cross-Validation Engine"]
        G --> H[Purged Walk-Forward Cross-Validation]
        G --> I[10-Year Out-of-Sample Final Rollout]
        H --> J[Internal Validation Early Stopping & F1 Threshold Tuning]
        I --> J
    end

    subgraph Evaluation["4. Diagnostics, Explainability & Strategy Simulation"]
        J --> K[Probabilistic Calibration: ECE & Brier Decomposition]
        J --> L[SHAP Tree Explanations: Global, Interactions & Waterfall]
        J --> M[Strategy Simulations: DCA vs Value Averaging vs ML Signal]
    end
```

---

## 📁 Package Organization

```text
models/
├── src/                          # Core modular package
│   ├── config.py                 # Centralized configuration, horizons, paths & hyperparameters
│   ├── data_loader.py            # Ingestion, cleaning, frequency resampling & dataset merging
│   ├── features.py               # Technical indicators, macroeconomic release lags & target labelling
│   ├── metrics.py                # Probabilistic (ECE, Brier) and financial (CAGR, Sharpe, MDD) metrics
│   ├── simulation.py             # Strategy simulators (DCA, Value Averaging Modified, ML Allocation)
│   ├── plots.py                  # High-resolution diagnostic figures, calibration plots & scorecards
│   ├── tuning.py                 # Time-aware random search with temporal validation
│   ├── tracking.py               # MLflow execution, stage, artifact and model tracking
│   ├── walk_forward.py           # Walk-forward validation engine with Purging and Embargo
│   ├── rollout.py                # 10-year out-of-sample rollout & regime stability evaluation
│   └── explainability.py         # Full model fitting, native export & in-sample SHAP interpretation
│
├── data/                         # Historical market and macroeconomic data files
├── metrics/                      # Generated evaluation artifacts (scorecards, plots, tables)
├── backtest/                     # Classic strategy comparison artifacts
│
├── run_pipeline.py               # Main CLI orchestrator with stage-by-stage execution
├── predictions.py                # Clean entrypoint for full end-to-end pipeline execution
├── compare_strategies_simple.py  # Rule-based (RSI, MA) backtesting benchmark
└── get_data.py                   # Data acquisition and scraping utilities
```

---

## 🔬 Key Methodological Pillars

### 1. Publication Lags and Data Revisions
Macroeconomic indicators (e.g. GDP, Unemployment, Housing Starts, Money Supply) are shifted backwards according to real-world publication calendars before being combined with market prices.

### 2. Purged Walk-Forward Temporal Cross-Validation
Financial time series violate the I.I.D. assumption. To prevent information leakage caused by overlapping multi-month forward returns, training folds are purged of overlapping observations prior to each test fold.

### 3. Validation-Based Decision Threshold Optimization
Walk-forward and rollout select the threshold on a historical validation block within the training window. Use `--threshold-class 0` to maximize F1 for negative alerts, or `--threshold-class 1` to maximize F1 for increases. Compare the number of detected negatives and false alarms alongside `AP_0` and `F1_0` in MLflow.

### 4. Rigorous Probabilistic Calibration
Beyond standard accuracy, predictions are evaluated through:
- **Expected Calibration Error (ECE)**
- **Brier Score Decomposition:** $\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$
- **Decile Calibration Tables & Empirical Positive Capture Curves**

> [!WARNING]
> **Known Limitations:**
> The 36-month horizon over monthly data yields ~20 independent observations and an 82.9% base rate; reported metrics are point estimates without confidence intervals and should be read as directional, not conclusive. Feature selection was informed by full-sample analysis, so walk-forward results are optimistic.

### 5. SHAP Interpretability
Using `TreeExplainer`, model decisions are fully explained through:
- Global feature importance and beeswarm distributions
- Non-linear feature dependence and automatic interaction plots
- Waterfall decomposition for the latest live market observation

---

## 🚀 Usage & CLI Guide

MLflow tracking is enabled by default. Each execution stores configuration and dataset
metadata, stage diagnostics, validation metrics, predictions and the final selected model.
See the [MLflow guide](../docs/mlflow.md) for the UI, Docker, configuration and Model Registry.
Add `--no-mlflow` to run without tracking.

### Run Complete Pipeline End-to-End
```bash
python models/predictions.py
```
*or via the CLI orchestrator:*
```bash
python models/run_pipeline.py --stage all
```

### Run Specific Stages
```bash
# 1. Exploratory Data Analysis & Feature Correlation Ranking:
python models/run_pipeline.py --stage eda

# 2. Walk-Forward Cross-Validation & Scorecards:
python models/run_pipeline.py --stage walk-forward

# 3. 10-Year Out-of-Sample Final Rollout:
python models/run_pipeline.py --stage rollout

# 4. Final Model Fitting & SHAP Interpretability:
python models/run_pipeline.py --stage explain
```

### Advanced CLI Arguments
```bash
python models/run_pipeline.py \
    --stage walk-forward \
    --horizon 36 \
    --start-date 1950-01-01 \
    --min-history 0.6 \
    --model xgboost --random-search
```

### Data Acquisition & Updates (`get_data.py`)
All 34 required historical datasets are stored directly in `models/data/`. To download or refresh datasets:
```bash
# Refresh all downloadable market and macroeconomic datasets:
python models/get_data.py --all

# Download or update specific data sources:
python models/get_data.py --source spy    # S&P 500 (SPY)
python models/get_data.py --source cape   # Shiller CAPE ratio
python models/get_data.py --source pmi    # ISM Manufacturing PMI
python models/get_data.py --source fred   # All St. Louis Fed FRED series
python models/get_data.py --source fred --fred-id M2SL  # Specific FRED series
```

---

## 📊 Generated Diagnostic Artifacts (`metrics/`)

For committed examples, see the [LightGBM figure gallery](../docs/images/README.md). Regenerate the seven reference figures with `python -m scripts.export_lightgbm_figures` from the repository root. Historical thesis images are archived separately. The general pipeline can produce different dates and scores because it uses its own dataset calendar.

| File | Description |
| --- | --- |
| `correlation_heatmap.png` | Pearson correlation matrix across active predictors and target |
| `spearman_rank_corr.png` | Monotonic rank correlation bar chart against forward returns |
| `walk_forward_metrics_scorecard.png` | Comprehensive average classification and probabilistic metrics table |
| `walk_forward_baselines_scorecard.png` | Model performance compared against constant base-rate baselines |
| `walk_forward_classification.png` | Timeline of price, forecast probability step curves, and hits/misses |
| `walk_forward_calibration.png` | Quantile calibration curve vs theoretical ideal |
| `walk_forward_roc_pr.png` | Dual ROC and Precision-Recall evaluation with baseline comparisons |
| `walk_forward_equity_curve_directional.png` | Continuous probability-weighted equity trajectory vs Buy & Hold |
| `roi_strategies_walk_forward.png` | Cumulative ROI (%) comparison: DCA vs Value Averaging vs ML Signal |
| `final_rollout_subperiod_metrics.png` | Subperiod stability breakdown (Early vs Late out-of-sample) |
| `shap_summary_cls.png` | Global SHAP beeswarm feature importance distribution |
| `shap_last_prediction_cls.png` | SHAP waterfall explaining the last labeled training row (in-sample) |
