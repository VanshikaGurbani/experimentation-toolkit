-- Observed uplift by score decile for one model. `scores` has (score, treatment, visit) for the
-- test split. Decile 1 holds the users the model ranks as most responsive to treatment. In a
-- good model the measured uplift falls from decile 1 to decile 10.
WITH ranked AS (
    SELECT
        treatment,
        visit,
        ntile(10) OVER (ORDER BY score DESC) AS decile
    FROM scores
)
SELECT
    decile,
    count(*)::BIGINT AS n,
    count(*) FILTER (WHERE treatment = 1)::BIGINT AS n_treated,
    count(*) FILTER (WHERE treatment = 0)::BIGINT AS n_control,
    avg(visit) FILTER (WHERE treatment = 1) AS visit_rate_treated,
    avg(visit) FILTER (WHERE treatment = 0) AS visit_rate_control,
    avg(visit) FILTER (WHERE treatment = 1) - avg(visit) FILTER (WHERE treatment = 0) AS uplift
FROM ranked
GROUP BY decile
ORDER BY decile;
