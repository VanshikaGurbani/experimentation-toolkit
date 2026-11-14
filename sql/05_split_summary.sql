-- Size and outcome rates of each split by arm, to confirm the split is balanced.
SELECT
    split,
    treatment,
    count(*)::BIGINT AS n,
    avg(visit) AS visit_rate,
    avg(conversion) AS conversion_rate
FROM criteo_split
GROUP BY split, treatment
ORDER BY split, treatment;
