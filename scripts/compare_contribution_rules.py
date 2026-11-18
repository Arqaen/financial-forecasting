"""Compare class/probability purchase sizing with variable and equal funding.

Run python -m scripts.compare_contribution_rules. Reuses saved OOS predictions.
"""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from models.src.config import DATA_DIR
from scripts.compare_signal_strategies import OUT as SIGNAL_OUT
from scripts.compare_signal_strategies import next_session
from scripts.train_selected_model import ROOT

OUT = ROOT / "docs/experiments/contribution-rules"
RULES = {
    "dca": "100 every month",
    "class_1_0": "100 if class 1; 0 if class 0",
    "class_2_0": "200 if class 1; 0 if class 0",
    "class_3_0": "300 if class 1; 0 if class 0",
    "class_2_1": "200 if class 1; 100 if class 0",
    "class_2_half": "200 if class 1; 50 if class 0",
    "class_1_half": "100 if class 1; 50 if class 0",
    "probability_1": "100 * P(class 1)",
    "probability_2": "200 * P(class 1)",
    "probability_3": "300 * P(class 1)",
    "probability_floor": "50 + 100 * P(class 1)",
    "probability_squared": "200 * P(class 1)^2",
    "probability_tiers": "0 / 50 / 100 / 200 at P1 <0.4 / <0.6 / <0.8 / >=0.8",
    "probability_high": "200 if P(class 1) >= 0.6; otherwise 0",
}


def requested_amount(frame, rule):
    p1 = 1 - frame.p0
    good = ~frame.alert
    if rule == "dca":
        return pd.Series(100.0, index=frame.index)
    class_rules = {
        "class_1_0": (100, 0),
        "class_2_0": (200, 0),
        "class_3_0": (300, 0),
        "class_2_1": (200, 100),
        "class_2_half": (200, 50),
        "class_1_half": (100, 50),
    }
    if rule in class_rules:
        bull, bear = class_rules[rule]
        return pd.Series(np.where(good, bull, bear), index=frame.index, dtype=float)
    if rule in ("probability_1", "probability_2", "probability_3"):
        return 100 * int(rule[-1]) * p1
    if rule == "probability_floor":
        return 50 + 100 * p1
    if rule == "probability_squared":
        return 200 * p1**2
    if rule == "probability_tiers":
        return pd.Series(
            np.select([p1 < 0.4, p1 < 0.6, p1 < 0.8], [0, 50, 100], default=200),
            index=frame.index,
            dtype=float,
        )
    if rule == "probability_high":
        return pd.Series(np.where(p1 >= 0.6, 200, 0), index=frame.index, dtype=float)
    raise ValueError(rule)


def simulate_amounts(requested, close, terminal, funding):
    """Buy-only account. Equal funding caps purchases at cash already available.

    No retroactive budget normalization, initial reserve or future contributions.
    All accounts stop buying after their last signal and retain any unused cash.
    """
    if funding not in ("variable", "equal"):
        raise ValueError(funding)
    if not requested.index.is_unique or not requested.index.is_monotonic_increasing:
        raise ValueError("Signals must be unique and ordered")
    if not np.isfinite(requested).all() or (requested < 0).any():
        raise ValueError("Purchase requests must be finite and non-negative")
    cash = 0.0
    records = []
    for date, desired in requested.items():
        flow = float(desired) if funding == "variable" else 100.0
        cash += flow
        purchase = min(float(desired), cash)
        cash -= purchase
        execution = next_session(close.index, date)
        if execution > pd.Timestamp(terminal):
            raise ValueError("Terminal precedes a contribution")
        records.append((date, execution, desired, flow, purchase, cash))
    ledger = pd.DataFrame(
        records,
        columns=["signal_date", "execution_date", "requested", "flow", "purchase", "cash_after"],
    )
    prices = close.loc[ledger.execution_date.iloc[0] : terminal]
    events = (
        ledger.groupby("execution_date")[["flow", "purchase"]]
        .sum()
        .reindex(prices.index, fill_value=0)
    )
    contributed = events.flow.cumsum()
    cash_daily = contributed - events.purchase.cumsum()
    if cash_daily.min() < -1e-7:
        raise AssertionError("Purchase schedule borrowed cash")
    cash_daily = cash_daily.clip(lower=0)
    shares = (events.purchase / prices).cumsum()
    wealth = shares * prices + cash_daily
    total = contributed.iloc[-1]
    roi = 100 * (wealth.iloc[-1] / total - 1) if total else np.nan
    previous = wealth.shift(1)
    returns = ((wealth - events.flow) / previous - 1).where(previous > 0, 0)
    nav = (1 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1
    dates = events.index[events.flow > 0]
    xirr = np.nan
    if len(dates) and total:
        times = (dates - dates[0]).days.to_numpy() / 365.25
        years = (prices.index[-1] - dates[0]).days / 365.25
        flows = events.loc[dates, "flow"].to_numpy()

        def npv(rate):
            return wealth.iloc[-1] * np.exp(-rate * years) - np.sum(flows * np.exp(-rate * times))

        if years > 0 and npv(-1) * npv(1) < 0:
            xirr = 100 * np.expm1(brentq(npv, -1, 1))
    result = dict(
        roi_pct=roi,
        contributed=total,
        wealth=wealth.iloc[-1],
        profit=wealth.iloc[-1] - total,
        deployed=events.purchase.sum(),
        terminal_cash=cash_daily.iloc[-1],
        xirr_pct=xirr,
        max_drawdown_pct=100 * drawdown.min(),
        peak_monthly_contribution=events.flow.max(),
        peak_monthly_purchase=events.purchase.max(),
        capped_months=int((ledger.purchase < ledger.requested - 1e-8).sum()),
    )
    return result, ledger


def choose_development(rows):
    return rows.sort_values(["dev_roi_pct", "strategy_id"], ascending=[False, True]).iloc[0]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = SIGNAL_OUT / "predictions.csv"
    pred = pd.read_csv(prediction_path, parse_dates=["date"])
    pred = pred.loc[pred.model_id.str.startswith("classifier_")]
    close = (
        pd.read_csv(DATA_DIR / "sp500.csv", skiprows=[1], index_col="Date", parse_dates=True)
        .Close.dropna()
        .sort_index()
    )
    dev_terminal = close.loc[:"2004-12-31"].index[-1]
    test_terminal = next_session(close.index, "2025-12-31")
    rows, ledgers = [], []
    for model_id, group in pred.groupby("model_id"):
        frame = group.set_index("date").sort_index()
        for cutoff in ("natural", "tuned"):
            frame["alert"] = frame[f"{cutoff}_alert"].astype(bool)
            for rule in RULES:
                if rule == "dca" and (model_id != "classifier_ros1" or cutoff != "tuned"):
                    continue
                desired = requested_amount(frame, rule)
                for funding in ("variable", "equal"):
                    strategy_id = f"{model_id}__{cutoff}__{rule}__{funding}"
                    row = dict(
                        strategy_id=strategy_id,
                        model_id=model_id,
                        cutoff=cutoff,
                        rule=rule,
                        description=RULES[rule],
                        funding=funding,
                    )
                    for period, interval, terminal in [
                        ("dev", slice("1982", "1999"), dev_terminal),
                        ("test", slice("2005", "2020"), test_terminal),
                    ]:
                        result, ledger = simulate_amounts(
                            desired.loc[interval], close, terminal, funding
                        )
                        row.update({f"{period}_{k}": v for k, v in result.items()})
                        ledgers.append(ledger.assign(strategy_id=strategy_id, period=period))
                    rows.append(row)
    table = pd.DataFrame(rows)
    for period in ("dev", "test"):
        baseline = table.loc[table.rule.eq("dca"), f"{period}_roi_pct"].iloc[0]
        table[f"{period}_vs_dca_pp"] = table[f"{period}_roi_pct"] - baseline
    picks = {}
    for funding in ("variable", "equal"):
        group = table.loc[table.funding.eq(funding)]
        picks[f"{funding}_development"] = choose_development(group).strategy_id
        picks[f"{funding}_retrospective"] = (
            group.sort_values(["test_roi_pct", "strategy_id"], ascending=[False, True])
            .iloc[0]
            .strategy_id
        )
    table["development_selected"] = table.strategy_id.isin(
        [picks["variable_development"], picks["equal_development"]]
    )
    table.to_csv(OUT / "all_combinations.csv", index=False, lineterminator="\n")
    pd.concat(ledgers).to_csv(OUT / "purchase_ledgers.csv.gz", index=False, lineterminator="\n")
    current = table.loc[table.model_id.eq("classifier_ros1") & table.cutoff.eq("tuned")]
    current.to_csv(OUT / "current_classifier.csv", index=False, lineterminator="\n")
    manifest = dict(
        selection=picks,
        combinations=len(table),
        rules=RULES,
        development="1982-1999; valued 2004-12-31",
        evaluation="2005-2020; valued 2026-01-02",
        source_sha256={
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                prediction_path,
                DATA_DIR / "sp500.csv",
                ROOT / "scripts/compare_contribution_rules.py",
            ]
        },
    )
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    render(current)
    print(
        current[
            [
                "rule",
                "funding",
                "test_contributed",
                "test_wealth",
                "test_roi_pct",
                "test_terminal_cash",
            ]
        ].to_string(index=False)
    )
    print("SELECTIONS")
    print(
        table.loc[
            table.strategy_id.isin(picks.values()),
            ["strategy_id", "dev_vs_dca_pp", "test_roi_pct", "test_vs_dca_pp"],
        ].to_string(index=False)
    )
    return table


def render(current):
    plt.switch_backend("Agg")
    pivot = current.pivot(index="rule", columns="funding", values="test_roi_pct").reindex(RULES)
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    y = np.arange(len(pivot))
    for offset, funding, color in [(-0.18, "variable", "#287b8e"), (0.18, "equal", "#c27a27")]:
        values = pivot[funding]
        ax.barh(
            y + offset,
            values,
            height=0.34,
            label=(
                "Variable external contributions"
                if funding == "variable"
                else "Equal 100/month; cash included"
            ),
            color=color,
        )
        for i, value in enumerate(values):
            ax.text(value + 2, i + offset, f"{value:.1f}%", va="center", fontsize=8)
    ax.set_yticks(
        y,
        [
            "Probability bands: 0 / 50 / 100 / 200" if r == "probability_tiers" else RULES[r]
            for r in pivot.index
        ],
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Cumulative ROI (%)",
        title="Class/probability purchase sizing · current LightGBM classifier\nSignals 2005–2020; valuation 2 Jan 2026; cash 0%; gross price returns",
    )
    ax.set_xlim(0, pivot.max().max() * 1.22)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), fontsize=9, ncol=2)
    fig.savefig(OUT / "contribution_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
