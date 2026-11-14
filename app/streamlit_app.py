"""Sample size calculator for A/B tests on a conversion rate.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expkit.power import mde_for_n, power_proportions, sample_size_proportions  # noqa: E402

st.set_page_config(page_title="A/B test sample size", layout="centered")
st.title("A/B test sample size")
st.write(
    "How many users does an experiment need? Enter the current conversion rate and the smallest "
    "improvement worth detecting."
)

col1, col2 = st.columns(2)
with col1:
    baseline_pct = st.number_input(
        "Baseline conversion rate (%)", min_value=0.01, max_value=99.0, value=4.0, step=0.1,
        help="The rate in the control group today, for example 4.0 for 4%.",
    )  # fmt: skip
    effect_kind = st.radio(
        "Minimum detectable effect is", ["Absolute (percentage points)", "Relative (%)"]
    )
    if effect_kind.startswith("Absolute"):
        mde_input = st.number_input("Minimum detectable effect (pp)", 0.01, 50.0, 0.5, 0.05)
        relative = False
    else:
        mde_input = st.number_input(
            "Minimum detectable effect (% of baseline)", 0.1, 500.0, 10.0, 0.5
        )
        relative = True
with col2:
    alpha_pct = st.select_slider("Significance level (%)", [1, 2, 5, 10], value=5)
    power_pct = st.select_slider("Power (%)", [70, 75, 80, 85, 90, 95], value=80)
    treat_share_pct = st.slider("Share of users in treatment (%)", 10, 90, 50)
    two_sided = st.checkbox("Two-sided test", value=True)
    daily_users = st.number_input("Eligible users per day (optional)", 0, 100_000_000, 0, 1000)

baseline = baseline_pct / 100
mde = mde_input / 100
ratio = treat_share_pct / (100 - treat_share_pct)

try:
    s = sample_size_proportions(
        baseline, mde, relative=relative, alpha=alpha_pct / 100, power=power_pct / 100,
        ratio=ratio, two_sided=two_sided,
    )  # fmt: skip
except ValueError as err:
    st.error(str(err))
    st.stop()

st.subheader("Result")
m1, m2, m3 = st.columns(3)
m1.metric("Control users", f"{s.n_control:,}")
m2.metric("Treatment users", f"{s.n_treatment:,}")
m3.metric("Total users", f"{s.total:,}")
st.write(
    f"To detect a change from **{baseline_pct:.2f}%** to **{s.treated_rate * 100:.2f}%** "
    f"({s.absolute_effect * 100:+.2f} percentage points, {s.relative_effect * 100:+.1f}% relative) "
    f"with {power_pct}% power at a {alpha_pct}% significance level."
)
if daily_users > 0:
    st.write(
        f"At {daily_users:,} eligible users per day, that is about "
        f"{math.ceil(s.total / daily_users)} days."
    )

st.subheader("Power by sample size")
grid = sorted(
    {max(50, int(s.n_control * f)) for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 3)}
)
power_by_n = [
    power_proportions(
        baseline, mde, n, max(1, round(n * ratio)), relative=relative, alpha=alpha_pct / 100,
        two_sided=two_sided,
    )
    * 100
    for n in grid
]  # fmt: skip
st.line_chart(pd.DataFrame({"Power (%)": power_by_n}, index=pd.Index(grid, name="Control users")))

st.subheader("What can this experiment detect?")
n_cont = st.number_input(
    "If you can only get this many control users", 100, 1_000_000_000, s.n_control, 1000
)
try:
    detectable = mde_for_n(
        baseline, int(n_cont), max(1, round(n_cont * ratio)), alpha=alpha_pct / 100,
        power=power_pct / 100, two_sided=two_sided,
    )  # fmt: skip
    st.write(f"The smallest lift detectable is **{detectable * 100:.2f} percentage points**.")
except ValueError as err:
    st.write(str(err))

with st.expander("How it is calculated"):
    st.write(
        "Normal approximation for a two-sample test of proportions (Fleiss formula, no continuity "
        "correction). Real tests with very small samples or rates near 0% or 100% may need exact "
        "methods. This calculator does not account for repeated looks at the data, multiple "
        "metrics, or variance reduction such as CUPED."
    )
