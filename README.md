# Experimentation Toolkit

*A small, tested Python toolkit for A/B tests, applied end to end to a real 14-million-user ad experiment*

## What is this?

Companies run experiments to decide things: does this ad, this email, this button make more people
visit or buy? Deciding well takes more than "the new version has a higher number". You need to know
how many users to test, whether the test ran as designed, how much of the difference is real, and
what to do when the answer is "not sure yet".

This project packages those steps as a reusable Python library, `expkit`, and uses it on the public
**Criteo uplift dataset** (14.0 million users, about 85% shown an ad campaign and 15% held back as
a comparison group). It ends with a one-page recommendation memo: ship, do not ship, or run longer,
and why. It is an independent portfolio project, not affiliated with Criteo.

## The result in one paragraph

<!-- HEADLINE:START -->
On 13,979,592 users from the Criteo uplift experiment, the ad campaign raised the visit rate by **+0.75 pp** (95% CI +0.71 to +0.78 pp; baseline 4.04%). The sample ratio check passed, but a model can predict assignment from the features (AUC 51.1% against 50% for a clean experiment), so the headline lift is the covariate-adjusted one and the raw difference of +1.03 pp overstates it. With a placeholder minimum worthwhile lift of +0.50 pp, the decision rule says **ship**. Treating only the top 10% of users ranked by the S-learner (LightGBM) captures 59% of the extra visits, against 51% for ranking by likelihood to visit. The A/A false positive rate measured 4.7% against 5% expected, and CUPAC cut the variance of the lift estimate by 30.5%.
<!-- HEADLINE:END -->

Full numbers: [docs/RESULTS.md](docs/RESULTS.md). The decision memo: [docs/MEMO.md](docs/MEMO.md).

## What is in the toolkit

| Piece | Module | What it does |
|---|---|---|
| Power and sample size | `expkit/power.py` | How many users per arm for a given baseline rate and minimum detectable effect, or what effect a given sample can detect. Handles unequal splits. |
| Sample ratio mismatch (SRM) | `expkit/srm.py` | Chi-square check that the arms are the size the design promised. If not, the data cannot be trusted. |
| Balance check | `expkit/balance.py` | Can the features predict who was treated? In a clean experiment they cannot. |
| Confidence intervals | `expkit/inference.py` | Difference in rates or means, absolute and relative, with 95% intervals. |
| Adjusted effect | `expkit/adjust.py` | Doubly robust (AIPW) estimate that corrects for arm imbalance. |
| Variance reduction | `expkit/cuped.py` | CUPED (and the CUPAC model-based variant): use pre-experiment information to get a tighter interval from the same users. |
| Decision rule | `expkit/decision.py` | SHIP, DO NOT SHIP, RUN LONGER or INVALID, by comparing the interval with a practical threshold, with guardrail metrics. |
| Uplift models | `expkit/uplift.py` | S-learner, T-learner (LightGBM), a "likelihood to visit" baseline, and an EconML causal forest. |
| Qini curve | `expkit/qini.py` | Evaluates who to target, with bootstrap confidence intervals. |
| Streamlit calculator | `app/streamlit_app.py` | Enter a baseline rate and minimum detectable effect, get the sample size. |

The data preparation is **SQL on DuckDB**, and the SQL is in [`sql/`](sql): loading and typing the
raw file, the train / validation / test split, arm summaries for the SRM check, the feature balance
check, and the uplift-by-decile table.

## How it was tested

- **Unit tests (99)** check each statistic against a textbook value, a second library (statsmodels,
  scipy) or a brute-force reference, and by simulation: the 95% intervals cover the truth about 95%
  of the time, the SRM check false-alarms at its stated rate, CUPED leaves the estimate unbiased,
  and the Qini curve of a random score is flat.
- **Leak and sanity tests**: shuffling the treatment labels destroys the uplift signal, shuffling
  the outcome destroys the Qini curve, the post-assignment `exposure` column is never used as a
  feature, and train, validation and test rows never overlap.
- **Checks on the real data** (in `docs/RESULTS.md`): an A/A test (splitting one arm in two should
  find nothing), and a power calibration (samples of the size the calculator recommends should
  detect the effect about as often as it promises).
- Every number in the results, the memo and the headline above is written by code from
  `docs/results/results.json`. Nothing is typed by hand. A test also checks that no em dash appears
  anywhere in the repository.

## What I found that I did not expect

The Criteo file is described as a randomized experiment, and the sample ratio check agrees: the
split is 85% / 15% almost exactly. But a model can still tell treated from control users from the
12 features slightly better than a coin flip. The raw difference between arms therefore mixes the
ad's effect with a small difference in who is in each arm. The covariate-adjusted estimate is
lower than the raw one, and the pre-specified rule in this project uses the adjusted estimate for
the decision when the balance check flags. See sections 1 and 2 of [docs/RESULTS.md](docs/RESULTS.md).

## Run it

Python 3.12 or 3.13 (3.14 is not supported by the pinned versions).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[causal,serve,dev]"   # on macOS or Linux: .venv/bin/python
expkit fetch-data      # downloads Criteo (about 311 MB) and Hillstrom (about 4 MB) into data/raw
expkit run-all         # about 8 minutes on a 16-core machine; rewrites docs/ and the README headline
pytest                 # 99 tests, no data download needed
ruff check . && ruff format --check .
streamlit run app/streamlit_app.py
```

`expkit run-all --fixture` runs the same pipeline on a small synthetic dataset (this is what CI does).

## Data

- **Criteo Uplift Modeling Dataset v2.1** (Diemert et al., 2018), 13,979,592 rows, licence
  CC BY-NC-SA 4.0 (non-commercial use, share-alike). Downloaded from the public mirror
  `criteo/criteo-uplift` on Hugging Face. Not stored in this repository.
- **Hillstrom MineThatData email test** (2008), 64,000 customers, used only to show CUPED with a
  true pre-period metric. Not stored in this repository.

## Limitations

- **Intent to treat, not exposure.** Only a small share of treated users actually saw the ad (the
  exact share is in RESULTS.md). The estimates are the effect of being assigned to the campaign.
- **Imbalance between arms** (above). The adjusted estimate relies on models of the features and
  removes only the imbalance those features reveal.
- **No costs or revenue.** The practical threshold that drives the ship decision is an
  assumption in `expkit/config.py`. The memo shows how the call changes with it.
- **The designed 85% / 15% allocation** is taken from the dataset's documentation, not proven by
  the file.
- **CUPED on Criteo uses features, not a pre-period metric**, because the file has none. On
  Hillstrom, which does have prior spend, the variance reduction is negligible.
- **Uplift results are modest.** See RESULTS.md for how the models compare with simply ranking
  users by likelihood to visit; the gap is small and the T-learner does not beat it. The exact
  capture percentages also move with the arm imbalance (inverse-propensity-weighted curves give
  higher shares for every model), so treat them as approximate. The conversion outcome (rare)
  was not modeled, only reported.
- **One experiment, one period**: no novelty, fatigue or seasonality checks are possible.
- **The Streamlit calculator** uses the normal approximation; very small samples or rates near 0%
  or 100% need exact methods, and repeated looks at the data are not corrected for.
- **EconML causal forest** was fitted on a 200,000-user subsample because forests are slow; the
  comparison with the LightGBM learners mixes training sizes.
- Not run on a second large dataset; no live deployment of the calculator yet.

## Repository layout

```
expkit/      package: power, srm, balance, inference, adjust, cuped, decision, uplift, qini,
             experiment (the pipeline), report (writes the docs), cli
sql/         the DuckDB SQL: load, split, summaries, balance, deciles
app/         Streamlit calculator
tests/       pytest suite (synthetic fixtures only)
docs/        RESULTS.md and MEMO.md (generated), results/ (results.json and figures)
```
