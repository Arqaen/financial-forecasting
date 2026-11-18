"""Reproduce the published comparison of signals and equal-budget trading policies.

Run: python -m scripts.compare_signal_strategies
Development economics stop in 2004; evaluation never selects the development winner.
"""

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from scipy.optimize import brentq
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from models.src.config import DATA_DIR, DEFAULT_FEATURES, LIGHTGBM_PARAMS
from models.src.data_loader import load_raw_dataset
from models.src.estimators import build_classifier
from models.src.features import create_targets
from models.src.tuning import oversample_class_zero
from scripts.compare_full_grid import make_folds, predict_fold, prepare_grid_values
from scripts.train_selected_model import ARCHIVED_FEATURES, HISTORICAL_OUTPUT, ROOT

OUT = ROOT / "docs/experiments/signal-strategies"
HORIZON = 36
REG_GRID = {
    "objective": ["regression", "regression_l1"],
    "num_leaves": [3, 7, 15],
    "learning_rate": [0.025, 0.05],
}
POLICIES = (
    *[f"delay_{n}" for n in (3, 6, 12, 24, 36)],
    "half_delay_12",
    "half_delay_36",
    "wait_good_12",
    "wait_good_36",
    "cash_forever",
    "scaled_delay_12",
    "rebalance_binary",
    "rebalance_scaled",
)


def load_panel():
    raw = load_raw_dataset()
    archived = pd.read_csv(ARCHIVED_FEATURES, index_col="fecha", parse_dates=True)
    x = prepare_grid_values(raw, archived, HORIZON, historical_erp=True)[DEFAULT_FEATURES]
    target = create_targets(raw, HORIZON).loc["1962":"2020"]
    assert target.target_reg.notna().all()
    close = (
        pd.read_csv(DATA_DIR / "sp500.csv", skiprows=[1], index_col="Date", parse_dates=True)
        .Close.dropna()
        .sort_index()
    )
    assert close.index.is_unique and (close > 0).all()
    return x, target, close


def sample_regression(x, y):
    """Duplicate non-positive-return FIT rows, preserving their continuous targets."""
    positions = pd.DataFrame({"position": np.arange(len(y))}, index=x.index)
    sampled, _ = oversample_class_zero(positions, (y > 0).astype(int))
    indices = sampled.position.to_numpy()
    return x.iloc[indices], y.iloc[indices]


def regression_threshold(actual, prediction):
    if (actual > 0).nunique() < 2:
        return 0.0
    thresholds = np.linspace(-0.4, 0.8, 50)
    scores = [f1_score(actual > 0, prediction >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def make_predictions(x, target):
    folds, audit = make_folds(x, target.target.astype(int), HORIZON, "previous")
    predictions, specs = {}, []
    for sampling in (False, True):
        model_id = f"classifier_ros{int(sampling)}"
        frames = [
            predict_fold(
                build_classifier("lightgbm", **LIGHTGBM_PARAMS), {}, "lightgbm", f, sampling
            )[0]
            for f in folds
        ]
        p = pd.concat(frames)
        p["risk"] = p.p0
        p["natural_alert"] = p.p0 > 0.5
        p["tuned_alert"] = p.alert
        p["exposure"] = 1 - p.p0
        p["actual_log_return"] = target.target_reg.reindex(p.index)
        predictions[model_id] = p
        specs.append(
            dict(
                model_id=model_id,
                task="classification",
                oversampling=sampling,
                params=json.dumps(LIGHTGBM_PARAMS, sort_keys=True),
            )
        )
    reference = pd.read_csv(
        HISTORICAL_OUTPUT / "walk_forward_predictions.csv", index_col="date", parse_dates=True
    )
    np.testing.assert_allclose(predictions["classifier_ros1"].p0, reference.p0, rtol=1e-10)
    np.testing.assert_array_equal(predictions["classifier_ros1"].alert, reference.alert)
    for sampling in (False, True):
        for candidate, params in enumerate(ParameterGrid(REG_GRID)):
            model_id = f"regressor_ros{int(sampling)}_{candidate:02d}"
            frames = []
            for fold in folds:
                xf, _, _, _, xv, _, xt, yt = fold
                yf = target.target_reg.loc[xf.index]
                if sampling:
                    xf, yf = sample_regression(xf, yf)
                model = LGBMRegressor(**{**LIGHTGBM_PARAMS, **params, "n_jobs": 1})
                model.fit(xf, yf)
                pv = model.predict(xv)
                pt = model.predict(xt)
                threshold = regression_threshold(target.target_reg.loc[xv.index], pv)
                frames.append(
                    pd.DataFrame(
                        dict(
                            actual=yt,
                            actual_log_return=target.target_reg.loc[xt.index],
                            predicted_log_return=pt,
                            threshold_log_return=threshold,
                            risk=-pt,
                            natural_alert=pt <= 0,
                            tuned_alert=pt < threshold,
                            # Fixed, uncalibrated allocation rule; not a probability.
                            exposure=np.clip(np.expm1(pt / 3) / 0.10, 0, 1),
                        ),
                        index=xt.index,
                    )
                )
            predictions[model_id] = pd.concat(frames)
            specs.append(
                dict(
                    model_id=model_id,
                    task="regression",
                    oversampling=sampling,
                    params=json.dumps({**LIGHTGBM_PARAMS, **params, "n_jobs": 1}, sort_keys=True),
                )
            )
            print(f"Fitted {model_id}: {params}", flush=True)
    return predictions, pd.DataFrame(specs), pd.DataFrame(audit)


def next_session(index, date):
    position = index.searchsorted(pd.Timestamp(date), side="right")
    if position >= len(index):
        raise ValueError(f"No execution session after {date}")
    return index[position]


def purchase_schedule(signals, close, policy, terminal, amount=100.0):
    """Schedule purchases causally: later good signals can only release cash later."""
    contributions, buys = {}, {}

    def add(mapping, date, value):
        mapping[date] = mapping.get(date, 0.0) + value

    for date, row in signals.iterrows():
        execution = next_session(close.index, date)
        add(contributions, execution, amount)
        if policy == "dca":
            add(buys, execution, amount)
            continue
        if policy == "cash_forever":
            if not row.alert:
                add(buys, execution, amount)
            continue
        delay = int(policy.rsplit("_", 1)[1])
        if policy.startswith("scaled_"):
            fraction = float(row.exposure)
        elif policy.startswith("half_"):
            fraction = 0.5 if row.alert else 1.0
        else:
            fraction = 0.0 if row.alert else 1.0
        add(buys, execution, amount * fraction)
        due_signal = date + pd.offsets.MonthEnd(delay)
        if policy.startswith("wait_good_"):
            good = signals.loc[(signals.index > date) & (signals.index <= due_signal)]
            good = good.loc[~good.alert]
            if len(good):
                due_signal = good.index[0]
        due = next_session(close.index, due_signal)
        if due <= terminal:
            add(buys, due, amount * (1 - fraction))
    return contributions, buys


def simulate(signals, close, policy, terminal, fee_bps=0.0, cash_rate=0.0):
    """Cash-account backtest, no borrowing, with external-flow-adjusted risk metrics.

    Fees apply to purchases/sales; terminal values are marked, not liquidated.
    After the final signal, whole-portfolio policies revert to 100% equity at the
    next month's execution. Contribution-only policies honor their remaining queues.
    """
    terminal = pd.Timestamp(terminal)
    start = next_session(close.index, signals.index.min())
    prices = close.loc[start:terminal]
    if prices.empty or prices.index[-1] != terminal:
        raise ValueError("Terminal must be an available close")
    fee = fee_bps / 10000
    if not 0 <= fee < 1 or cash_rate <= -1:
        raise ValueError("Invalid fee or cash rate")
    rebalancing = policy.startswith("rebalance_")
    if rebalancing:
        contributions = {next_session(close.index, d): 100.0 for d in signals.index}
        allocations = {
            next_session(close.index, d): (
                float(r.exposure) if policy.endswith("scaled") else float(not r.alert)
            )
            for d, r in signals.iterrows()
        }
        end_policy = next_session(close.index, signals.index[-1] + pd.offsets.MonthEnd(1))
        if end_policy <= terminal:
            allocations[end_policy] = 1.0
        buys = {}
    else:
        contributions, buys = purchase_schedule(signals, close, policy, terminal)
        allocations = {}
    cash, shares, total, fees, turnover = 0.0, 0.0, 0.0, 0.0, 0.0
    previous_wealth, previous_date = 0.0, prices.index[0]
    records = []
    for date, price in prices.items():
        days = (date - previous_date).days
        cash *= (1 + cash_rate) ** (days / 365.25)
        flow = contributions.get(date, 0.0)
        cash += flow
        total += flow
        before = shares * price + cash
        if date in allocations:
            weight = allocations[date]
            current = shares * price
            # Solve post-fee equity/(wealth-fee*trade) = desired weight.
            delta = (weight * before - current) / (
                1 + weight * fee if weight * before >= current else 1 - weight * fee
            )
            trade_fee = abs(delta) * fee
            shares += delta / price
            cash -= delta + trade_fee
        else:
            budget = buys.get(date, 0.0)
            delta = budget / (1 + fee)
            trade_fee = delta * fee
            shares += delta / price
            cash -= budget
        if cash < -1e-7 or shares < -1e-7:
            raise AssertionError("Strategy borrowed cash or sold short")
        cash, shares = max(cash, 0.0), max(shares, 0.0)
        fees += trade_fee
        turnover += abs(delta)
        wealth = shares * price + cash
        # External contribution is deposited at the trading close; gains precede flow.
        daily_return = (
            (wealth - flow) / previous_wealth - 1 if previous_wealth else wealth / flow - 1
        )
        records.append((date, wealth, total, cash, flow, daily_return, fees, turnover))
        previous_wealth, previous_date = wealth, date
    history = pd.DataFrame(
        records,
        columns=[
            "date",
            "wealth",
            "contributed",
            "cash",
            "flow",
            "daily_return",
            "fees",
            "turnover",
        ],
    ).set_index("date")
    nav = (1 + history.daily_return).cumprod()
    drawdown = nav / np.maximum(nav.cummax(), 1.0) - 1
    years = (history.index[-1] - history.index[0]).days / 365.25
    flow_dates = history.index[history.flow > 0]
    times = (flow_dates - history.index[0]).days.to_numpy() / 365.25
    flows = history.loc[flow_dates, "flow"].to_numpy()
    final_wealth = history.wealth.iloc[-1]

    def npv(log_rate):
        return final_wealth * np.exp(-log_rate * years) - np.sum(flows * np.exp(-log_rate * times))

    irr = np.expm1(brentq(npv, -1.0, 1.0)) if years > 0 and npv(-1.0) * npv(1.0) < 0 else np.nan
    result = dict(
        roi_pct=100 * (final_wealth / total - 1),
        wealth=final_wealth,
        contributed=total,
        xirr_pct=100 * irr,
        max_drawdown_pct=100 * drawdown.min(),
        twr_annual_pct=100 * (nav.iloc[-1] ** (1 / years) - 1) if years else np.nan,
        mean_cash_pct=100 * (history.cash / history.wealth).mean(),
        terminal_cash=history.cash.iloc[-1],
        fees=fees,
        turnover_multiple=turnover / total,
    )
    history["nav"] = nav
    history["drawdown_pct"] = 100 * drawdown
    return result, history


def signal_metrics(frame, rule):
    negative = frame.actual.eq(0)
    alert = frame[f"{rule}_alert"]
    result = dict(
        n=len(frame),
        n0=int(negative.sum()),
        alerts=int(alert.sum()),
        AP0=average_precision_score(negative, frame.risk),
        ROC_AUC=roc_auc_score(negative, frame.risk),
        F1_0=f1_score(negative, alert, zero_division=0),
        recall0=float((negative & alert).sum() / negative.sum()),
        false_alarms=int((~negative & alert).sum()),
    )
    if "predicted_log_return" in frame:
        y, p = frame.actual_log_return, frame.predicted_log_return
        result.update(
            MAE_log=mean_absolute_error(y, p),
            RMSE_log=np.sqrt(mean_squared_error(y, p)),
            R2_log=r2_score(y, p),
            spearman=y.corr(p, method="spearman"),
            sign_accuracy=float(((y > 0) == (p > 0)).mean()),
        )
    return result


def select_development(table):
    """Only development columns enter selection; deterministic tie break."""
    return table.sort_values(
        ["dev_roi_pct", "dev_turnover_multiple", "combination"], ascending=[False, True, True]
    ).iloc[0]


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", FutureWarning)
        x, target, close = load_panel()
        predictions, specs, audit = make_predictions(x, target)
    specs.to_csv(out / "model_grid.csv", index=False, lineterminator="\n")
    audit.to_csv(out / "fold_audit.csv", index=False, lineterminator="\n")
    pd.concat([p.assign(model_id=k) for k, p in predictions.items()]).to_csv(
        out / "predictions.csv", index_label="date", lineterminator="\n"
    )
    terminal_dev = close.loc[:"2004-12-31"].index[-1]
    terminal_test = next_session(close.index, "2025-12-31")
    assert terminal_dev < pd.Timestamp("2005-01-31")
    baseline_signals = predictions["classifier_ros1"].assign(alert=False)
    baselines = {}
    for name, dates, terminal in [
        ("dev", slice("1982", "1999"), terminal_dev),
        ("test", slice("2005", "2020"), terminal_test),
    ]:
        baselines[name] = simulate(baseline_signals.loc[dates], close, "dca", terminal)[0]
    rows = []
    spec_map = specs.set_index("model_id")
    for model_id, frame in predictions.items():
        for rule in ("natural", "tuned"):
            p = frame.assign(alert=frame[f"{rule}_alert"])
            dm, tm = signal_metrics(p.loc["1982":"1999"], rule), signal_metrics(
                p.loc["2005":"2020"], rule
            )
            for policy in POLICIES:
                dev, _ = simulate(p.loc["1982":"1999"], close, policy, terminal_dev)
                test, _ = simulate(p.loc["2005":"2020"], close, policy, terminal_test)
                rows.append(
                    dict(
                        combination=f"{model_id}__{rule}__{policy}",
                        model_id=model_id,
                        task=spec_map.loc[model_id, "task"],
                        oversampling=spec_map.loc[model_id, "oversampling"],
                        rule=rule,
                        policy=policy,
                        **{f"dev_{k}": v for k, v in {**dm, **dev}.items()},
                        **{f"test_{k}": v for k, v in {**tm, **test}.items()},
                        dev_vs_dca_pp=dev["roi_pct"] - baselines["dev"]["roi_pct"],
                        test_vs_dca_pp=test["roi_pct"] - baselines["test"]["roi_pct"],
                    )
                )
        print(f"Evaluated strategies: {model_id}", flush=True)
    table = pd.DataFrame(rows)
    dca = dict(
        combination="dca",
        model_id="none",
        task="baseline",
        oversampling=False,
        rule="none",
        policy="dca",
        dev_vs_dca_pp=0.0,
        test_vs_dca_pp=0.0,
    )
    for name, result in baselines.items():
        dca.update({f"{name}_{k}": v for k, v in result.items()})
    table = pd.concat([pd.DataFrame([dca]), table], ignore_index=True)
    winner = select_development(table)
    table["selected_by_development"] = table.combination.eq(winner.combination)
    table.to_csv(out / "all_combinations.csv", index=False, lineterminator="\n")
    picks = {
        "development_winner": winner.combination,
        "dca": "dca",
        "current_reference": "classifier_ros1__tuned__delay_36",
    }
    for task in ("classification", "regression"):
        group = table.loc[table.task.eq(task)]
        picks[f"{task}_development_winner"] = select_development(group).combination
        picks[f"{task}_retrospective_test_max"] = (
            group.sort_values("test_roi_pct", ascending=False).iloc[0].combination
        )
    selected = table.set_index("combination").loc[list(dict.fromkeys(picks.values()))].reset_index()
    selected["roles"] = selected.combination.map(
        lambda c: ", ".join(k for k, v in picks.items() if v == c)
    )
    selected.to_csv(out / "shortlist.csv", index=False, lineterminator="\n")
    reference = selected.loc[selected.combination.eq(picks["current_reference"])].iloc[0]
    np.testing.assert_allclose(reference.test_roi_pct, 351.58423923727213, rtol=1e-10)
    np.testing.assert_allclose(baselines["test"]["roi_pct"], 332.01140198982193, rtol=1e-10)
    sensitivities, cohorts, histories = [], [], []
    for _, row in selected.iterrows():
        p = (
            baseline_signals
            if row.policy == "dca"
            else predictions[row.model_id].assign(
                alert=predictions[row.model_id][f"{row.rule}_alert"]
            )
        )
        for fee_bps, cash_rate in [(0, 0), (10, 0), (25, 0), (10, 0.02)]:
            scores, history = simulate(
                p.loc["2005":"2020"], close, row.policy, terminal_test, fee_bps, cash_rate
            )
            base, _ = simulate(
                baseline_signals.loc["2005":"2020"], close, "dca", terminal_test, fee_bps, cash_rate
            )
            sensitivities.append(
                dict(
                    combination=row.combination,
                    fee_bps=fee_bps,
                    cash_rate=cash_rate,
                    **scores,
                    vs_dca_pp=scores["roi_pct"] - base["roi_pct"],
                )
            )
            if fee_bps == 0 and cash_rate == 0:
                histories.append(history.assign(combination=row.combination))
        # Restart each cohort with no capital; same terminal within each comparison.
        for start, end in [
            (1982, 1989),
            (1990, 1999),
            (2005, 2008),
            (2009, 2014),
            (2015, 2020),
            (2009, 2020),
        ]:
            dates = slice(str(start), str(end))
            terminal = terminal_dev if end < 2000 else terminal_test
            scores, _ = simulate(p.loc[dates], close, row.policy, terminal)
            base, _ = simulate(baseline_signals.loc[dates], close, "dca", terminal)
            cohorts.append(
                dict(
                    combination=row.combination,
                    cohort=f"{start}-{end}",
                    **scores,
                    vs_dca_pp=scores["roi_pct"] - base["roi_pct"],
                )
            )
    pd.DataFrame(sensitivities).to_csv(out / "sensitivity.csv", index=False, lineterminator="\n")
    pd.DataFrame(cohorts).to_csv(out / "cohorts.csv", index=False, lineterminator="\n")
    pd.concat(histories).to_csv(
        out / "portfolio_history.csv.gz", index_label="date", lineterminator="\n"
    )
    sources = [
        ARCHIVED_FEATURES,
        DATA_DIR / "sp500.csv",
        HISTORICAL_OUTPUT / "walk_forward_predictions.csv",
        Path(__file__),
    ]
    manifest = dict(
        horizon=HORIZON,
        features=DEFAULT_FEATURES,
        models=len(specs),
        combinations=len(table),
        development_signals="1982-1999",
        development_terminal=str(terminal_dev.date()),
        test_signals="2005-2020",
        test_terminal=str(terminal_test.date()),
        selection="Maximum development ROI including DCA; tie: lower turnover, then ID",
        picks=picks,
        python=platform.python_version(),
        lightgbm=lightgbm.__version__,
        sklearn=sklearn.__version__,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    refresh_fingerprints(out)
    render_chart(selected, out)
    print(
        selected[
            [
                "roles",
                "combination",
                "dev_vs_dca_pp",
                "test_roi_pct",
                "test_vs_dca_pp",
                "test_max_drawdown_pct",
            ]
        ].to_string(index=False)
    )
    return table


def refresh_fingerprints(out):
    """Include the complete local input/code dependency set in the source record."""
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = [
        *sorted(DATA_DIR.glob("*.csv")),
        *sorted((ROOT / "models/src").glob("*.py")),
        ARCHIVED_FEATURES,
        HISTORICAL_OUTPUT / "walk_forward_predictions.csv",
        ROOT / "scripts/compare_full_grid.py",
        ROOT / "scripts/compare_ensembles.py",
        ROOT / "scripts/train_selected_model.py",
        Path(__file__),
    ]
    manifest["source_sha256"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def render_chart(selected, out):
    plt.switch_backend("Agg")

    def label(row):
        if row.policy == "dca":
            return "DCA"
        task = "Regression" if row.task == "regression" else "Classification"
        role = (
            "development choice" if "development_winner" in row.roles else "retrospective maximum"
        )
        if "current_reference" in row.roles:
            role = "current reference"
        policy = {
            "wait_good_36": "Wait for favorable signal (up to 36m)",
            "delay_36": "Delay flagged purchases by 36m",
            "rebalance_binary": "Switch entire portfolio equity/cash",
        }.get(row.policy, row.policy)
        return f"{task}: {role}\n{policy}"

    labels = selected.apply(label, axis=1)
    fig, axes = plt.subplots(
        1, 2, figsize=(15, max(4.5, len(selected) * 0.6)), layout="constrained"
    )
    for ax, column, title in [
        (axes[0], "dev_vs_dca_pp", "Development: selected here (1982–1999)"),
        (axes[1], "test_vs_dca_pp", "Evaluation: already explored (2005–2020)"),
    ]:
        values = selected[column].to_numpy()
        ax.barh(
            np.arange(len(selected)),
            values,
            color=["#287b8e" if v >= 0 else "#bc5548" for v in values],
        )
        ax.set_yticks(np.arange(len(selected)), labels if ax is axes[0] else [])
        ax.axvline(0, color="0.3", lw=0.8)
        ax.invert_yaxis()
        ax.set(title=title, xlabel="Cumulative ROI difference vs DCA (percentage points)")
        low = min(selected.dev_vs_dca_pp.min(), selected.test_vs_dca_pp.min(), 0)
        high = max(selected.dev_vs_dca_pp.max(), selected.test_vs_dca_pp.max(), 0)
        margin = max(high - low, 1) * 0.2
        ax.set_xlim(low - margin, high + margin)
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:+.2f}",
                (v, i),
                xytext=(4 if v >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9,
            )
    fig.suptitle(
        "LightGBM classification vs regression · equal monthly contributions\nGross price returns; cash at 0%; different holding periods across panels",
        fontsize=13,
    )
    fig.savefig(out / "strategy_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    run(parser.parse_args().out_dir)
