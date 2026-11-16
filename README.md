# Experimentation Toolkit

A tested Python toolkit for A/B tests (sample size, validity checks, adjusted effect, variance reduction, ship/no-ship decision, uplift targeting), applied end to end to a 14-million-user ad experiment from Criteo.

**Live demo:** [sample size calculator](https://experimentation-toolkit.streamlit.app/) (Streamlit)

## Results

<!-- HEADLINE:START -->
**Setup.** 13,979,592 users in the Criteo uplift experiment: 85% assigned to an ad campaign, the rest held out as control. Outcome: visiting the advertiser's website.

- **Effect:** the campaign raised the visit rate from 4.04% to 4.78%, a lift of **+0.75 pp** (95% CI +0.71 to +0.78 pp), about 7.5 extra visits per 1,000 users.
- **Validity:** the sample ratio check passed, but a model can predict assignment from the features (AUC 51.1%; 50% is a clean experiment). The covariate-adjusted lift is therefore the headline, and the raw difference of +1.03 pp overstates it.
- **Decision:** **ship**, against a placeholder minimum worthwhile lift of +0.50 pp (the data has no cost or revenue). The call changes at higher thresholds (notebook 02).
- **Targeting:** treating only the top 10% of users ranked by the S-learner (LightGBM) captures 59% of the extra visits, against 51% for ranking by likelihood to visit.
- **Toolkit checks:** the A/A false positive rate measured 4.7% against 5% expected, and CUPAC cut the variance of the lift estimate by 30.5%.
<!-- HEADLINE:END -->

Full tables: [docs/RESULTS.md](docs/RESULTS.md). Decision memo: [docs/MEMO.md](docs/MEMO.md).

![Raw versus adjusted lift](docs/figures/validity_forest_lift.png)

![Decision by threshold](docs/figures/validity_threshold_sensitivity.png)

![Qini curves with 95% bootstrap bands](docs/figures/uplift_qini_curves.png)

## What is inside

- **`expkit/`**: power and sample size, SRM and balance checks, confidence intervals, doubly robust (AIPW) adjustment, CUPED/CUPAC, decision rule, S-/T-learner uplift models and Qini curves
- **`sql/`**: the DuckDB data preparation
- **`notebooks/`**: five notebooks with saved outputs: [EDA](notebooks/01_eda_criteo.ipynb), [validity and effect](notebooks/02_experiment_validity_and_effect.ipynb), [CUPED, power and A/A](notebooks/03_cuped_power_and_aa.ipynb), [uplift and targeting](notebooks/04_uplift_and_targeting.ipynb), [Hillstrom](notebooks/05_hillstrom_sanity.ipynb)
- **`app/`**: the Streamlit calculator

**Tech stack:** Python, DuckDB, pandas, scikit-learn, LightGBM, EconML, Matplotlib, Jupyter, Streamlit, pytest, GitHub Actions

## Quick start

```bash
git clone https://github.com/VanshikaGurbani/experimentation-toolkit.git
cd experimentation-toolkit
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[causal,serve,dev,notebooks]"   # macOS or Linux: .venv/bin/python
expkit fetch-data      # downloads Criteo (about 311 MB) and Hillstrom (about 4 MB)
expkit run-all         # about 8 minutes; writes docs/ and the results above
pytest                 # no data download needed
```

Requires Python 3.12 or 3.13.

## Hardest part

The Criteo data is described as randomized and the SRM check agrees, yet a model can still predict assignment from the features slightly better than chance. The raw difference then overstates the lift. The fix was a rule fixed in code: when the balance check flags, the doubly robust estimate is the headline and the raw one is shown only for comparison.

## Limitations

- The ship decision uses a placeholder threshold, because the data has no cost or revenue.
- Estimates are the effect of being assigned the campaign; only a small share of treated users saw the ad.
- Uplift models beat the likelihood-to-visit baseline only modestly. One experiment, one period, one seed.

## Data and licence

Code: MIT ([LICENSE](LICENSE)). The datasets are not included and keep their own licences: **Criteo Uplift Modeling Dataset v2.1** (CC BY-NC-SA 4.0, mirror `criteo/criteo-uplift` on Hugging Face) and the **Hillstrom MineThatData** email test. Independent portfolio project, not affiliated with Criteo.
