# Resume bullets: Experimentation Toolkit

Every number below is read from `docs/RESULTS.md` (written by `expkit run-all` from
`docs/results/results.json`). Re-run it and confirm before sending anything out.

## Two bullets

* Built a Python A/B testing toolkit (power and sample size, sample ratio mismatch check, CUPED
  variance reduction, confidence intervals with a ship, don't ship or run longer decision rule)
  with SQL data prep in DuckDB, and applied it to 14 million users from the Criteo ad experiment;
  a model-based CUPED adjustment cut the variance of the lift estimate by 30.5%, and an A/A test
  on real rows measured a 4.7% false positive rate against the 5% expected.
* Found the experiment's arms were not balanced (a model predicts treatment from user features
  with 51.1% AUC, where 50% is clean), so the raw ad lift of +1.03 percentage points overstated
  the doubly robust (AIPW) estimate of +0.75 points (95% CI +0.71 to +0.78) in visit rate;
  recommended shipping in a one-page memo, and built uplift models with bootstrap Qini curves
  where targeting the top 10% of users captured 59% of extra visits versus 51% for ranking by
  likelihood to visit.

## What you must verify before using these

1. **Reproduce the numbers.** Run `expkit fetch-data` then `expkit run-all` and compare the
   figures in each bullet with `docs/RESULTS.md` (30.5%, 4.7%, 51.1%, +1.03, +0.75, +0.71 to
   +0.78, 59%, 51%). The full run takes about 8 minutes on a 16-core machine. Timings will
   differ on yours; the statistics should match because every seed is fixed.
2. **CI status.** The branch has not been pushed. Push it (your call), open the Actions tab and
   confirm the run is green before you mention CI. The workflow also runs Python 3.12, which was
   not available on the build machine (only 3.13 was run locally).
3. **Be ready to explain the terms.** Sample ratio mismatch, CUPED versus CUPAC (what covariate
   was used and why Criteo has no true pre-period metric), why the balance check uses a model
   AUC, what AIPW does and why it is "doubly robust", what a Qini curve measures, and why the
   top 10% capture share can differ between a weighted and an unweighted curve.
4. **The "ship" call depends on an assumption.** The 0.50 percentage point minimum worthwhile
   lift is a placeholder (the dataset has no costs). At 0.75 points the rule says run longer and
   at 1.00 point it says do not ship. Do not present the call as a business fact.
5. **Do not overstate the uplift models.** The S-learner beat the likelihood-to-visit ranking by a
   small margin and the T-learner did not beat it at all (RESULTS.md section 6). The 59% versus
   51% figure is for the top 10% only; at the top 30% both capture about 85%.
6. **Causal wording.** This is an intent-to-treat estimate (only 3.6% of treated users saw the
   ad) from an experiment whose arms turned out slightly imbalanced. Say "estimated lift", not
   "proved the ad caused".
7. **Data licence.** The Criteo dataset is CC BY-NC-SA 4.0 (non-commercial, share-alike). It is
   not stored in the repository. Fine for a portfolio, but do not redistribute it.
8. **Repository housekeeping.** The repo has an MIT `LICENSE` for the code. The Streamlit
   calculator is not deployed, so do not list a live link until one exists.
9. **Hillstrom CUPED.** The pre-period spend barely helped there (0.03% variance reduction).
   Do not claim CUPED helped on Hillstrom; the 30.5% is from Criteo with a model-based covariate.
