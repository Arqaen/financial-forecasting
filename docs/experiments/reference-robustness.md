# Robustness of the 36-month LightGBM reference

The historical edge over monthly dollar-cost averaging (DCA) depends on the
2005–2008 episode and the equity risk premium (ERP). The 0.949 ROC-AUC comes
from retrospective walk-forward predictions on a period used in the research.
It requires confirmation on future forecasts with dated inputs. The audit fixes
the horizon, classifier, oversampling, threshold rule and purchase delay.

## Dependence between outcomes

The 192 signals cover January 2005–December 2020. Class 0 means a non-positive
S&P 500 price return from month-end to 36 months later. Adjacent labels share
35 months of price changes. All 34 class-0 dates fall between September 2005 and
September 2008. Scoring every 36th month gives only five or six non-overlapping
return windows per starting offset, with zero to two class-0 windows. Those 36
offsets reuse the same market history.

| Signal cohort | Months | Class 0 | ROC-AUC | False alarms | Cumulative ROI vs DCA within cohort | Contribution to the full-sample ROI gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2005–2008 | 48 | 34 | 0.799 | 7 | +59.88 pp | +14.97 pp |
| 2009–2020 | 144 | 0 | Undefined | 5 | −2.27 pp | −1.70 pp |
| 2005–2020 | 192 | 34 | 0.949 | 12 | +13.27 pp | +13.27 pp |

Each cohort starts with zero holdings and uses the same terminal date. Its
contribution to the full result divides its incremental terminal wealth by
the **full** 19,200 of contributions. In the 1982–1999 development window the
reference AP₀ was 0.061 against class-0 prevalence of 0.079.

As a sensitivity check, 5,000 resamples for each block length (36, 48 and 60
months) draw contiguous circular blocks of paired, per-contribution payoffs.
Blocks preserve local dependence and can wrap from the last month to the first.
At 36 months, the 2.5th and 97.5th percentiles of the cumulative ROI gap were
−4.11 and +43.66 pp. Across block lengths, 24.2%–24.4% of draws had a
non-positive gap, and 7.7%–9.8% contained zero class-0 months. These are
descriptive resampling statistics from a single historical path. The audit
keeps the valuation date fixed and leaves model selection and fitting outside
the resampling.

## Point-in-time inputs

Publication lags approximate when revised series could first have been
available. The archive uses observations from vintages available at each
decision date. Its year-over-year features use both observations from the same
vintage. The archived credit spread uses the latest common BAA and AAA month
available on the decision date. The ERP still uses historical S&P 500 P/E
without release provenance. [ALFRED real-time data](https://fred.stlouisfed.org/docs/api/fred/alfred.html) · [Philadelphia Fed M2 vintages](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/m2)

| Feature | Archived months in 1962–1981 (240) | 1982–1999 (216) | 2005–2020 (192) |
| --- | ---: | ---: | ---: |
| ERP | 0 | 0 | 0 |
| Credit spread | 0 | 37 | 191 |
| Unemployment change | 240 | 216 | 192 |
| M2 growth | 128 | 216 | 192 |
| Building permits growth | 0 | 5 | 192 |

The audit refits the same classifier on the original folds and scores the same
dates. In the archive-only run, missing values stay missing. A candidate
feature needs at least 60 nonmissing observations and more than one distinct
value in the fitting block. The first test fold uses unemployment and M2.
Feature availability changes across folds.

| Inputs | Development AP₀ | 2005–2020 AP₀ | 2005–2020 ROC-AUC | Class-0 hits / 34 | False alarms | Cumulative ROI vs DCA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid reference with ERP | 0.061 | 0.765 | 0.949 | 16 | 12 | +13.27 pp |
| Hybrid excluding ERP | 0.141 | 0.167 | 0.427 | 27 | 58 | −3.27 pp |
| Archived observations only | 0.080 | 0.160 | 0.316 | 0 | 16 | −4.45 pp |

The ERP exclusion shows how much the result depends on that input under this
model and calendar. The archive-only run also changes feature availability and
missingness in fitting, so its difference combines those effects with vintage
substitution. Confirming the five-feature model requires a dated P/E or
earnings archive with point-in-time index membership and a documented release
calendar.

## Investment results

Each policy receives 100 at the next trading session after each signal. DCA
buys immediately. The signal policy defers flagged contributions for 36 months
and keeps existing holdings. Both portfolios use the same valuation date,
2 January 2026. Fees are proportional to executed purchases with fractional
shares and no terminal sale. The 2% cash scenario is hypothetical, with
interest retained as cash. The simulation excludes taxes.

| Execution series | Cost per purchase | Cash yield | Signal / DCA XIRR | Signal / DCA max drawdown | Cumulative ROI vs DCA |
| --- | ---: | ---: | ---: | ---: | ---: |
| S&P 500 price index | 0 bp | 0% | 11.20% / 10.98% | −38.53% / −56.78% | +13.27 pp |
| S&P 500 price index | 25 bp | 0% | 11.18% / 10.96% | −38.62% / −56.83% | +13.24 pp |
| S&P 500 price index | 10 bp | 2% | 11.21% / 10.97% | −36.72% / −56.80% | +14.46 pp |
| SPY adjusted close | 0 bp | 0% | 13.02% / 12.86% | −37.53% / −55.19% | +12.34 pp |
| SPY adjusted close | 25 bp | 0% | 13.00% / 12.84% | −37.63% / −55.24% | +12.31 pp |
| SPY adjusted close | 10 bp | 2% | 13.03% / 12.85% | −35.74% / −55.21% | +13.53 pp |

SPY adjusted closes approximate reinvested distributions and the ETF's realized
tracking of the index. This sensitivity holds the original price-index labels
and signals fixed. With zero fees and cash yield, the SPY result amounts to
**+0.16 pp annualized XIRR** and **−0.22 pp annualized time-weighted return**
versus DCA. Starting SPY contributions in 2009 gives **−2.98 pp cumulative
ROI** versus DCA. [SPY fund documentation](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-500-etf-trust-spy) · [S&P index return definitions](https://www.spglobal.com/spdji/en/methodology/article/index-mathematics-methodology/)

## Prospective validation protocol

1. Before the first new signal, record a dated commit and a manifest for a single
   primary candidate. Freeze the feature definitions and sources, LightGBM
   parameters, fitting window, oversampling seed, threshold rule, costs and
   purchase policy. Fix the evaluation end date and treatment of missing inputs
   before seeing results. Retain DCA as the investment benchmark and use the
   matured training prevalence as a probability baseline.
2. Store a timestamped snapshot at each month-end. For every observation retain
   its reference month, actual release or vintage timestamp, source and data
   hash. Select the values available by the decision cutoff, including both
   yearly comparison observations from that vintage. Use dated earnings and
   Treasury yields for ERP. If suitable earnings vintages are unavailable,
   freeze a separately specified archive-only candidate before starting.
3. Commit the score, alert, model hash, data snapshot hash and decision time
   before the next-session execution. Follow the frozen refit rule using only
   labels already mature at the time. Keep original forecasts when data are
   revised. Mark missed signals and apply the prespecified fallback equally
   throughout the evaluation.
4. Score forecasts once their 36-month labels mature. At the prespecified
   end, report class-0 AP against prevalence, Brier against the probability
   baseline, and results by distinct episode. Report undefined metrics where
   only one class occurs. Report net XIRR, time-weighted return, terminal
   wealth and flow-adjusted drawdown against DCA with identical contributions
   and execution assumptions. A calendar with too few distinct adverse
   episodes gives inconclusive evidence even if the monthly row count is large.
   The first month-end forecast made after freezing needs a further 36 months
   to mature.

## Reproduce

Run from the repository root with the project dependencies installed.

```bash
python -m scripts.audit_reference_robustness
```

The command uses bundled prices, predictions and archived observations. To
include the SPY sensitivity, save daily adjusted closes with US market dates.
The historical run used these dates and the report records the input SHA-256
hashes. [Yahoo Finance SPY historical data](https://finance.yahoo.com/quote/SPY/history/)

```python
from pathlib import Path

import yfinance as yf

out = Path("models/metrics/spy_adjusted_2005_2026.csv")
out.parent.mkdir(parents=True, exist_ok=True)
spy = yf.download(
    "SPY", start="2005-01-01", end="2026-01-03", auto_adjust=False, multi_level_index=False
)
spy[["Adj Close"]].to_csv(out, index_label="Date")
```

```bash
python -m scripts.audit_reference_robustness --spy-csv models/metrics/spy_adjusted_2005_2026.csv
```
