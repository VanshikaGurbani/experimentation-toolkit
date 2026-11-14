"""End-to-end analysis. Every number in docs/RESULTS.md and the memo comes from `run_all`."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import config, data
from .adjust import aipw_effect
from .balance import assignment_auc, fit_propensity
from .cuped import cuped_compare, fit_theta
from .decision import Guardrail, decide
from .inference import EffectEstimate, compare_proportions, compare_samples
from .power import (
    mde_for_n,
    power_curve,
    power_proportions,
    sample_size_proportions,
)
from .qini import percentile_ci, qini_bootstrap, qini_curve
from .srm import srm_test
from .uplift import CausalForest, ResponseModel, SLearner, TLearner


@dataclass
class Settings:
    aa_reps: int = config.AA_REPS
    aa_n_per_arm: int = config.AA_N_PER_ARM
    power_reps: int = config.POWER_REPS
    decision_reps: int = config.DECISION_REPS
    n_boot: int = config.BOOTSTRAP_REPS
    cf_train_n: int = config.CAUSAL_FOREST_TRAIN_N
    cf_eval_n: int = config.CAUSAL_FOREST_EVAL_N
    run_causal_forest: bool = True
    seed: int = config.SEED

    @classmethod
    def smoke(cls) -> Settings:
        return cls(
            aa_reps=200,
            aa_n_per_arm=2000,
            power_reps=200,
            decision_reps=100,
            n_boot=10,
            cf_train_n=8000,
            cf_eval_n=8000,
        )


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _est(e: EffectEstimate) -> dict:
    return e.as_dict()


def _wilson(k: int, n: int) -> list[float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return [float(ci.low), float(ci.high)]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=data.ROOT,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------- design and effect


def stage_design(con) -> dict:
    arms = data.run_sql(con, "03_arm_summary.sql").set_index("treatment")
    n_c, n_t = int(arms.loc[0, "n"]), int(arms.loc[1, "n"])
    srm = srm_test([n_c, n_t], config.DESIGN_RATIO_CONTROL_TREATMENT)

    # How well would the check catch lost data? Counts-only arithmetic on the design ratio.
    sens = []
    for total, lost in ((n_c + n_t, 0.001), (n_c + n_t, 0.01), (20_000, 0.01), (20_000, 0.10)):
        exp_c = total * 0.15
        exp_t = total * 0.85
        r = srm_test(
            [round(exp_c), round(exp_t * (1 - lost))], config.DESIGN_RATIO_CONTROL_TREATMENT
        )
        sens.append(
            {
                "total_users": total,
                "treated_lost_share": lost,
                "p_value": r.p_value,
                "flagged": r.flagged,
            }  # fmt: skip
        )

    bal = data.run_sql(con, "04_feature_balance.sql").set_index("treatment")
    smd = []
    for i in range(12):
        m0, m1 = bal.loc[0, f"m{i}"], bal.loc[1, f"m{i}"]
        s0, s1 = bal.loc[0, f"s{i}"], bal.loc[1, f"s{i}"]
        smd.append(float((m1 - m0) / np.sqrt((s0**2 + s1**2) / 2)))
    splits = data.run_sql(con, "05_split_summary.sql")
    return {
        "arms": {
            "control": {k: int(arms.loc[0, k]) for k in ("n", "visits", "conversions", "exposed")},
            "treatment": {
                k: int(arms.loc[1, k]) for k in ("n", "visits", "conversions", "exposed")
            },
        },
        "srm": {
            "observed": list(srm.observed),
            "expected": list(srm.expected),
            "observed_treated_share": srm.observed_shares[1],
            "chi2": srm.chi2,
            "p_value": srm.p_value,
            "flagged": srm.flagged,
        },
        "srm_sensitivity": sens,
        "max_abs_smd": float(np.max(np.abs(smd))),
        "smd": smd,
        "split_sizes": {
            str(r.split) + "_" + ("treated" if r.treatment == 1 else "control"): int(r.n)
            for r in splits.itertuples()
        },
    }


def stage_adjusted(train: dict, val: dict, test: dict) -> tuple[dict, np.ndarray]:
    """Balance check and covariate-adjusted (AIPW) effect on the held-out test split.

    Nuisance models (propensity, outcome under each arm) are fitted on the train split with
    early stopping on the validation split, so the test split is used only for estimation.
    """
    import lightgbm as lgb

    from .uplift import DEFAULT_PARAMS

    prop = fit_propensity(train["X"], train["t"], (val["X"], val["t"]), seed=config.SEED)
    e_test = prop.predict_proba(test["X"])[:, 1]
    balance = assignment_auc(test["t"], e_test)

    mu = {}
    for arm in (0, 1):
        m = lgb.LGBMClassifier(random_state=config.SEED + arm, n_jobs=-1, **DEFAULT_PARAMS)
        tr, va = train["t"] == arm, val["t"] == arm
        m.fit(
            train["X"][tr],
            train["visit"][tr],
            eval_X=val["X"][va],
            eval_y=val["visit"][va],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        mu[arm] = m.predict_proba(test["X"])[:, 1]
    y, t = test["visit"], test["t"]
    adj = aipw_effect(y, t, mu[0], mu[1], e_test)
    raw = compare_samples(y[t == 1].astype(float), y[t == 0].astype(float))
    return {
        "balance": {
            "auc": balance.auc,
            "null_se": balance.null_se,
            "z": balance.z,
            "p_value": balance.p_value,
            "threshold": balance.threshold,
            "flagged": balance.flagged,
            "n_test": balance.n_test,
            "propensity_range_1_99": [
                float(np.quantile(e_test, 0.01)),
                float(np.quantile(e_test, 0.99)),
            ],
        },
        "visit_adjusted": _est(adj),
        "visit_raw_test": _est(raw),
    }, e_test


def stage_effect(design: dict, adjusted: dict) -> dict:
    a, b = design["arms"]["control"], design["arms"]["treatment"]
    srm = srm_test(design["srm"]["observed"], config.DESIGN_RATIO_CONTROL_TREATMENT)
    raw = compare_proportions(b["visits"], b["n"], a["visits"], a["n"])
    conv = compare_proportions(b["conversions"], b["n"], a["conversions"], a["n"])
    adj = EffectEstimate(**adjusted["visit_adjusted"])
    # Pre-specified rule: if the balance check flags, the covariate-adjusted estimate is primary.
    basis = "adjusted" if adjusted["balance"]["flagged"] else "raw"
    primary = adj if basis == "adjusted" else raw

    rows = []
    for thr in config.THRESHOLD_SENSITIVITY:
        dp, dr = decide(primary, thr, srm=srm), decide(raw, thr, srm=srm)
        rows.append(
            {
                "threshold": thr,
                "action": dp.action,
                "sample_scale": dp.sample_scale,
                "action_raw": dr.action,
            }
        )
    dprim = decide(primary, config.PRACTICAL_THRESHOLD, srm=srm)
    draw = decide(raw, config.PRACTICAL_THRESHOLD, srm=srm)
    guard = decide(
        primary, config.PRACTICAL_THRESHOLD, srm=srm,
        guardrails=[Guardrail("conversion rate", conv, tolerance=0.0)],
    )  # fmt: skip
    return {
        "visit": _est(primary),
        "primary_basis": basis,
        "visit_raw": _est(raw),
        "visit_adjusted": _est(adj),
        "conversion": _est(conv),
        "exposed_share_of_treated": b["exposed"] / b["n"],
        "practical_threshold": config.PRACTICAL_THRESHOLD,
        "decision": {
            "action": dprim.action,
            "reason": dprim.reason,
            "sample_scale": dprim.sample_scale,
        },
        "decision_raw": {"action": draw.action, "reason": draw.reason},
        "decision_with_conversion_guardrail": {"action": guard.action, "reason": guard.reason},
        "threshold_sensitivity": rows,
    }


# ---------------------------------------------------------------- power calculator


def stage_power(design: dict, effect: dict) -> dict:
    p_c = effect["visit_raw"]["mean_control"]
    lift = effect["visit"]["diff"]
    n_c, n_t = design["arms"]["control"]["n"], design["arms"]["treatment"]["n"]
    ratio = n_t / n_c
    rows = []
    for label, mde, rel in (
        ("0.5 percentage points", 0.005, False),
        ("measured lift (raw)", effect["visit_raw"]["diff"], False),
        ("measured lift (adjusted)", effect["visit_adjusted"]["diff"], False),
        ("10% relative", 0.10, True),
    ):
        eq = sample_size_proportions(p_c, mde, relative=rel)
        sk = sample_size_proportions(p_c, mde, relative=rel, ratio=ratio)
        rows.append(
            {
                "label": label,
                "absolute_effect": eq.absolute_effect,
                "n_per_arm_equal": eq.n_control,
                "total_equal": eq.total,
                "total_85_15": sk.total,
                "n_control_85_15": sk.n_control,
                "n_treatment_85_15": sk.n_treatment,
            }
        )
    grid = [int(x) for x in np.unique(np.geomspace(500, 100_000, 25).astype(int))]
    return {
        "baseline_rate": p_c,
        "design_ratio_treat_to_control": ratio,
        "table": rows,
        "mde_of_actual_experiment": mde_for_n(p_c, n_c, n_t),
        "curve": {"n_per_arm": grid, "power": power_curve(p_c, lift, grid)},
    }


# ---------------------------------------------------------------- checks of the toolkit itself


def stage_validation(vt: np.ndarray, vc: np.ndarray, effect: dict, st: Settings) -> dict:
    """Simulation checks on real rows: are the test and the calculator calibrated?"""
    rng = np.random.default_rng(st.seed)
    out: dict = {}

    # A/A: split the treated arm into two random halves. Truth: no difference.
    n = min(st.aa_n_per_arm, len(vt) // 2)
    fp = 0
    for _ in range(st.aa_reps):
        idx = rng.choice(len(vt), size=2 * n, replace=False)
        e = compare_proportions(int(vt[idx[:n]].sum()), n, int(vt[idx[n:]].sum()), n)
        fp += e.p_value < config.ALPHA
    out["aa"] = {
        "n_per_arm": n,
        "reps": st.aa_reps,
        "false_positive_rate": fp / st.aa_reps,
        "ci95": _wilson(fp, st.aa_reps),
        "nominal": config.ALPHA,
    }

    # Power: draw samples of the calculator's recommended size from the real arms.
    p_c, lift = effect["visit_raw"]["mean_control"], effect["visit_raw"]["diff"]
    planned = sample_size_proportions(p_c, lift)
    cases = []
    for label, n_arm in (
        ("planned n", planned.n_control),
        ("half of planned n", planned.n_control // 2),
    ):
        n_arm = min(n_arm, len(vc), len(vt))
        hits = 0
        for _ in range(st.power_reps):
            xc = int(vc[rng.choice(len(vc), size=n_arm, replace=False)].sum())
            xt = int(vt[rng.choice(len(vt), size=n_arm, replace=False)].sum())
            e = compare_proportions(xt, n_arm, xc, n_arm)
            hits += (e.p_value < config.ALPHA) and e.diff > 0
        cases.append(
            {
                "label": label,
                "n_per_arm": n_arm,
                "nominal_power": power_proportions(p_c, lift, n_arm),
                "empirical_power": hits / st.power_reps,
                "ci95": _wilson(hits, st.power_reps),
                "reps": st.power_reps,
            }
        )
    out["power"] = cases

    # Decision-rule behavior as the sample grows.
    ops = []
    for mult in config.DECISION_N_MULTIPLIERS:
        n_arm = min(planned.n_control * mult, len(vc), len(vt))
        tally = {"SHIP": 0, "RUN_LONGER": 0, "DONT_SHIP": 0}
        for _ in range(st.decision_reps):
            xc = int(vc[rng.choice(len(vc), size=n_arm, replace=False)].sum())
            xt = int(vt[rng.choice(len(vt), size=n_arm, replace=False)].sum())
            d = decide(compare_proportions(xt, n_arm, xc, n_arm), config.PRACTICAL_THRESHOLD)
            tally[d.action] += 1
        ops.append(
            {
                "n_per_arm": n_arm,
                "multiplier": mult,
                "reps": st.decision_reps,
                **{k.lower(): v / st.decision_reps for k, v in tally.items()},
            }
        )
    out["decision"] = ops
    out["planned_n_per_arm"] = planned.n_control
    return out


# ---------------------------------------------------------------- CUPED


def _cuped_row(label: str, comp) -> dict:
    return {
        "label": label,
        "diff": comp.adjusted.diff,
        "se": comp.adjusted.se,
        "ci_low": comp.adjusted.ci_low,
        "ci_high": comp.adjusted.ci_high,
        "variance_reduction": comp.variance_reduction,
        "sample_size_ratio": comp.sample_size_ratio,
    }


def stage_cuped_criteo(train: dict, val: dict, test: dict) -> dict:
    import lightgbm as lgb

    y_test = test["visit"].astype(np.float64)
    t_test = test["t"]
    base = compare_samples(y_test[t_test == 1], y_test[t_test == 0])
    theta = fit_theta(train["visit"].astype(np.float64), train["X"])
    lin = cuped_compare(y_test, test["X"], t_test, theta=theta)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=500,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=config.SEED, n_jobs=-1, verbose=-1,
    )  # fmt: skip
    model.fit(train["X"], train["visit"])
    pred_val = model.predict_proba(val["X"])[:, 1]
    theta_cupac = fit_theta(val["visit"].astype(np.float64), pred_val)
    pred_test = model.predict_proba(test["X"])[:, 1]
    cupac = cuped_compare(y_test, pred_test, t_test, theta=theta_cupac)
    return {
        "n_test": len(y_test),
        "raw": {
            "diff": base.diff,
            "se": base.se,
            "ci_low": base.ci_low,
            "ci_high": base.ci_high,
        },  # fmt: skip
        "linear": _cuped_row("CUPED, linear on f0 to f11", lin),
        "cupac": _cuped_row("CUPAC, LightGBM prediction of visit", cupac),
        "theta_linear": [float(x) for x in lin.theta],
    }


def stage_cuped_hillstrom(con, csv: Path) -> dict:
    data.build_hillstrom(con, csv)
    summ = data.run_sql(con, "08_hillstrom_summary.sql")
    counts = [int(x) for x in summ["n"]]
    srm = srm_test(counts, [1, 1, 1])
    df = con.execute("SELECT treatment, history, spend FROM hillstrom ORDER BY row_id").df()
    comp = cuped_compare(
        df["spend"].to_numpy(), df["history"].to_numpy(), df["treatment"].to_numpy()
    )
    corr = float(np.corrcoef(df["spend"], df["history"])[0, 1])
    return {
        "segments": summ["segment"].tolist(),
        "counts": counts,
        "srm_p_value": srm.p_value,
        "srm_flagged": srm.flagged,
        "n": len(df),
        "corr_spend_history": corr,
        "raw": {
            "diff": comp.raw.diff,
            "se": comp.raw.se,
            "ci_low": comp.raw.ci_low,
            "ci_high": comp.raw.ci_high,
        },  # fmt: skip
        "cuped": _cuped_row("CUPED, pre-period spend (history)", comp),
    }


# ---------------------------------------------------------------- uplift


def _curve_summary(y, t, scores: dict[str, np.ndarray], st: Settings, seed: int) -> dict:
    boot = qini_bootstrap(
        y, t, scores, n_boot=st.n_boot, n_bins=config.QINI_BINS,
        capture_fractions=config.CAPTURE_FRACTIONS, seed=seed,
    )  # fmt: skip
    out: dict = {}
    ref = "response_model" if "response_model" in scores else next(iter(scores))
    for name, s in scores.items():
        c = qini_curve(y, t, s, config.QINI_BINS)
        row = {
            "qini_coefficient": c.coefficient,
            "qini_ci95": percentile_ci(boot[name]["coefficient"]),
            "ate": c.ate,
            "curve_x": [float(v) for v in c.x[::5]],
            "curve_q": [float(v) for v in c.q[::5]],
        }
        for f in config.CAPTURE_FRACTIONS:
            row[f"capture@{f}"] = c.capture(f)
            row[f"capture@{f}_ci95"] = percentile_ci(boot[name][f"capture@{f}"])
        if name != ref:
            diff = boot[name]["coefficient"] - boot[ref]["coefficient"]
            row[f"qini_minus_{ref}"] = c.coefficient - qini_curve(y, t, scores[ref]).coefficient
            row[f"qini_minus_{ref}_ci95"] = percentile_ci(diff)
        out[name] = row
    return out


def stage_uplift(
    con, train: dict, val: dict, test: dict, st: Settings, e_test: np.ndarray | None = None
) -> dict:
    y_col = "visit"
    models = {
        "response_model": ResponseModel(seed=st.seed),
        "s_learner": SLearner(seed=st.seed),
        "t_learner": TLearner(seed=st.seed),
    }
    valset = (val["X"], val["t"], val[y_col])
    scores_test, scores_val, secs = {}, {}, {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(train["X"], train["t"], train[y_col], val=valset)
        secs[name] = time.time() - t0
        scores_val[name] = m.score(val["X"])
        scores_test[name] = m.score(test["X"])
        _log(f"fitted {name} in {secs[name]:.0f}s")

    val_qini = {
        n: qini_curve(val[y_col], val["t"], s, config.QINI_BINS).coefficient
        for n, s in scores_val.items()
    }
    uplift_models = [n for n in models if n != "response_model"]
    selected = max(uplift_models, key=lambda n: val_qini[n])

    _log("bootstrap on full test split")
    main = _curve_summary(test[y_col], test["t"], scores_test, st, st.seed)

    # Robustness: re-weight users by inverse propensity so arm imbalance does not tilt the curves.
    ipw = None
    if e_test is not None:
        e = np.clip(e_test, 0.01, 0.99)
        w = test["t"] / e + (1 - test["t"]) / (1 - e)
        ipw = {}
        for name, s in scores_test.items():
            c = qini_curve(test[y_col], test["t"], s, config.QINI_BINS, weights=w)
            ipw[name] = {
                "qini_coefficient": c.coefficient,
                "ate": c.ate,
                **{f"capture@{f}": c.capture(f) for f in config.CAPTURE_FRACTIONS},
            }

    # Observed uplift by decile, in SQL, for the model chosen on the validation split.
    df = pd.DataFrame(
        {"score": scores_test[selected], "treatment": test["t"], "visit": test[y_col]}
    )
    con.register("scores", df)
    deciles = data.run_sql(con, "06_uplift_deciles.sql")
    con.unregister("scores")

    out = {
        "outcome": y_col,
        "n_train": len(train["t"]),
        "n_val": len(val["t"]),
        "n_test": len(test["t"]),
        "fit_seconds": secs,
        "validation_qini": val_qini,
        "selected_on_validation": selected,
        "test": main,
        "test_ipw": ipw,
        "deciles_selected_model": deciles.to_dict(orient="list"),
    }

    # EconML causal forest on subsamples (slow), compared on a shared evaluation subset.
    if st.run_causal_forest and importlib.util.find_spec("econml") is not None:
        rng = np.random.default_rng(st.seed)
        tr_idx = rng.choice(
            len(train["t"]), size=min(st.cf_train_n, len(train["t"])), replace=False
        )
        ev_idx = np.sort(
            rng.choice(len(test["t"]), size=min(st.cf_eval_n, len(test["t"])), replace=False)
        )
        t0 = time.time()
        cf = CausalForest(seed=st.seed).fit(
            train["X"][tr_idx], train["t"][tr_idx], train[y_col][tr_idx]
        )
        fit_s = time.time() - t0
        t0 = time.time()
        cf_scores = cf.score(test["X"][ev_idx])
        score_s = time.time() - t0
        _log(f"causal forest fit {fit_s:.0f}s, scored {len(ev_idx)} rows in {score_s:.0f}s")
        sub_scores = {n: s[ev_idx] for n, s in scores_test.items()}
        sub_scores["causal_forest"] = cf_scores
        sub = _curve_summary(test[y_col][ev_idx], test["t"][ev_idx], sub_scores, st, st.seed + 1)
        out["causal_forest"] = {
            "n_train": len(tr_idx),
            "n_eval": len(ev_idx),
            "fit_seconds": fit_s,
            "score_seconds": score_s,
            "eval_subset": sub,
        }
    else:
        out["causal_forest"] = None
    return out


# ---------------------------------------------------------------- orchestration


def run_all(
    criteo_csv: Path,
    hillstrom_csv: Path,
    db_path: Path | None,
    out_json: Path,
    st: Settings | None = None,
    rebuild: bool = False,
    dataset_label: str = "Criteo Uplift Modeling Dataset v2.1",
) -> dict:
    st = st or Settings()
    t_start = time.time()
    con = data.connect(db_path)
    if rebuild or not data.has_table(con, "criteo"):
        _log("building DuckDB tables from CSV")
        data.build_criteo(con, criteo_csv)
    else:
        con.execute(data.sql_text("02_split.sql"))

    _log("design checks")
    design = stage_design(con)

    _log("loading splits")
    train, val, test = (data.fetch_split(con, s) for s in ("train", "val", "test"))
    _log("balance check and adjusted effect")
    adjusted, e_test = stage_adjusted(train, val, test)
    effect = stage_effect(design, adjusted)
    effect["balance"] = adjusted["balance"]
    effect["visit_raw_test"] = adjusted["visit_raw_test"]
    power = stage_power(design, effect)
    vt = np.concatenate([d["visit"][d["t"] == 1] for d in (train, val, test)])
    vc = np.concatenate([d["visit"][d["t"] == 0] for d in (train, val, test)])

    _log("simulation checks")
    validation = stage_validation(vt, vc, effect, st)
    _log("CUPED")
    cuped = {
        "criteo": stage_cuped_criteo(train, val, test),
        "hillstrom": stage_cuped_hillstrom(con, hillstrom_csv),
    }
    _log("uplift models")
    uplift = stage_uplift(con, train, val, test, st, e_test)

    results = {
        "meta": {
            "dataset": dataset_label,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "packages": {
                p: version(p)
                for p in ("numpy", "pandas", "scipy", "scikit-learn", "duckdb", "lightgbm")
            }
            | ({"econml": version("econml")} if importlib.util.find_spec("econml") else {}),
            "seed": st.seed,
            "alpha": config.ALPHA,
            "power": config.POWER,
            "settings": st.__dict__,
            "runtime_seconds": time.time() - t_start,
        },
        "design": design,
        "effect": effect,
        "power": power,
        "validation": validation,
        "cuped": cuped,
        "uplift": uplift,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    con.close()
    _log(f"wrote {out_json}")
    return results
