-- Arm sizes and outcomes for the Hillstrom test.
SELECT
    segment,
    count(*)::BIGINT AS n,
    avg(visit) AS visit_rate,
    avg(conversion) AS conversion_rate,
    avg(spend) AS mean_spend,
    avg(history) AS mean_history
FROM hillstrom
GROUP BY segment
ORDER BY segment;
