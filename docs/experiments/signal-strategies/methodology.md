# Classification, regression and investment policies

Run from the repository root:

```powershell
.venv/Scripts/python -m scripts.compare_signal_strategies
```

This experiment changes the response variable and the investment policy while retaining the selected 36-month horizon, original five features and hybrid feature panel. It does not repeat the six-horizon feature search or replace the default model automatically. Outputs are research backtests, not live forecasts.

## Prediction and selection calendar

- Annual purged walk-forward folds reproduce the reference study: fitting history begins in 1962; development predictions cover 1982–1999 and evaluation predictions cover 2005–2020. The five intervening years are not scored. Their matured observations may enter subsequent training windows.
- Each fold uses the existing fitting, reserved early-stopping and threshold-validation windows. No model here uses early stopping; both tasks retain the same fit dates, even though that leaves the reserved window unused. The latest permitted target ends before the fold origin; the fit targets also mature before the internal validation begins.
- Development strategy wealth is valued at the last available close on or before 31 December 2004. No 2005–2026 price enters strategy selection. Evaluation wealth is valued on 2 January 2026, reproducing the existing study.
- The selected model/policy pair maximizes development cumulative ROI, including DCA as a candidate. Ties favor less turnover and then the stable combination identifier. The largest evaluation ROI is reported separately as a **retrospective maximum**, never relabeled as the development selection.
- Development returns are themselves optimized and are not unbiased performance estimates. The evaluation period has already informed prior research, including horizon/feature choices. Reusing it for this comparison does not create a fresh holdout or eliminate multiple-comparison bias.

## Models and signals

There are 26 fitted configurations:

- Two classifiers retain the selected LightGBM hyperparameters (100 trees, 3 leaves, learning rate 0.025, minimum leaf size 15 and L2 regularization 1), with/without class-0 oversampling.
- Twenty-four regressors predict `log(Close[t+36] / Close[t])`. The grid crosses L2/L1 loss, 3/7/15 leaves, learning rates 0.025/0.05, and negative-return oversampling off/on. Other hyperparameters match the classifier; seed 42 is fixed. L1 targets a conditional median and L2 targets a conditional mean of log returns; neither directly predicts expected simple ROI. [LightGBM objective definitions](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective).
- Oversampling duplicates fitting rows with non-positive returns until their count matches positive-return rows. Regression retains each duplicated row's **continuous target**. This changes the training distribution and can bias return levels; it is an experimental loss emphasis, not probability calibration or synthetic interpolation. Validation/test distributions are unchanged.
- Each configuration supplies a natural alert and a validation-tuned alert. Classification uses `P(0) > 0.5` or the existing past-validation F1-of-class-1 cutoff. Regression uses predicted log return `<= 0` or a cutoff chosen by F1 of positive returns on past validation, from 50 fixed points between -0.4 and 0.8; single-class validation falls back to zero.
- Fractional exposure is fixed beforehand: classification uses `1-P(0)`; regression uses `clip(expm1(predicted_log_return/3) / 0.10, 0, 1)`. This maps a 10% predicted annualized price return to full exposure. It is a heuristic allocation, not a calibrated probability. Fractional policies do not use the binary alert, so their natural/tuned rows intentionally repeat the same policy outcome.

## Equal-budget investment policies

All accounts start empty. Each receives 100 units at the first trading-session close strictly after each signal month-end: 21,600 in development and 19,200 in evaluation. Trades cannot execute at the close used to calculate the signal. No leverage or short selling is allowed.

| Policy | Action |
| --- | --- |
| DCA | Invest each contribution immediately. |
| `delay_3/6/12/24/36` | Invest favorable contributions immediately; wait the specified months on an alert, then buy regardless of later signals. |
| `half_delay_12/36` | On an alert, buy half now and the other half after the specified delay. |
| `wait_good_12/36` | On an alert, hold that contribution until the next favorable monthly signal, or until its maximum delay expires, whichever occurs first. |
| `cash_forever` | Keep flagged contributions in cash through valuation; existing holdings stay invested. |
| `scaled_delay_12` | Invest the fractional exposure immediately and delay the remaining contribution for 12 months. |
| `rebalance_binary` | Trade the entire portfolio to 0% equity on an alert and 100% otherwise. |
| `rebalance_scaled` | Trade the entire portfolio to the fractional exposure each signal month. |

The 13 signal policies crossed with 26 configurations and two alert rules produce 676 rows, plus DCA: **677 comparisons**. Repeated outcomes are retained to make the factorial table explicit, not counted as independent experiments.

After the last signal, contribution-only policies honor their pending queues; they do not invent new signals. Whole-portfolio policies revert to full equity at the first trading session after the following month-end (January 2000/2021) and then hold. This specified stopping rule is part of the result, not a continuous strategy evaluated through 2026. Pending cash is always included in wealth, even if not yet invested at valuation.

## Measures and sensitivity

- Cumulative ROI: `100 * (terminal wealth / total external contributions - 1)`. It is not an annual return and should not be compared between development and evaluation as though holding periods were equal.
- XIRR: annualized money-weighted return using the actual contribution dates and terminal marked wealth; no fictitious initial lump sum.
- Maximum drawdown and annualized time-weighted return: computed from daily account returns after removing external contributions, so deposits cannot hide losses. The account begins at the first contribution; risk includes the post-signal holding period.
- Turnover: gross traded notional divided by total contributions. Cash share is the mean over observed trading days.
- Predictive measures: AP0, ROC-AUC, F1/recall of class 0 and false alarms; regression additionally has log-return MAE/RMSE/R², rank correlation and natural sign accuracy. Regression risk is minus predicted log return, **not a probability**; no Brier/log-loss is reported for it.
- Base case: price index returns, cash earns 0%, no transaction costs, dividends or taxes. Holdings are marked at terminal close without a hypothetical liquidation fee.
- The saved shortlist is re-evaluated with 10/25 basis points per traded notional and with a hypothetical constant 2% cash yield plus 10 bp costs. These are sensitivity assumptions, not historical financing or broker cost estimates. Each scenario compares with DCA under the same costs.
- Cohorts restart an empty account with contributions only in the named interval. They retain the same within-development/evaluation terminal date; their ROIs are not directly comparable across differently aged cohorts. Signal-following stops at each cohort end using the same stopping rule. Cohorts show concentration and are not additive pieces of the full portfolio, especially for whole-portfolio rebalancing.

## Evidence and limitations

`predictions.csv`, `fold_audit.csv`, `model_grid.csv`, `all_combinations.csv`, `shortlist.csv`, `sensitivity.csv`, `cohorts.csv` and `portfolio_history.csv.gz` preserve the evidence. `manifest.json` records source fingerprints, software versions and selection identifiers. Tests check cash conservation, next-session execution, unchanged earlier wealth after future signal changes, external-flow-adjusted drawdown, fees, continuous-target sampling and development-only selection. The script reproduces the saved classifier probabilities and the prior 351.5842% strategy / 332.0114% DCA ROI before reporting a successful run.

The panel contains revised macroeconomic values and archived features; it is not fully point-in-time. Existing feature definitions are deliberately preserved for this controlled comparison, including the documented earnings-yield/bond-yield unit mismatch in ERP. Fixing these inputs requires a new evaluation. At 36 months, all 34 negative evaluation labels occur in the 2005–2008 episode. Overlapping forward labels and repeated configurations are not independent observations. Costs, dividends, true cash yields and taxes could materially alter the apparent advantage of frequently selling or waiting. A candidate that wins here still requires a corrected point-in-time evaluation and fresh, untouched evidence before a deployment decision.
