-- One row per arm: size, outcome counts, exposure share. Source of the SRM check and the ATE.
SELECT
    treatment,
    count(*)::BIGINT AS n,
    sum(visit)::BIGINT AS visits,
    sum(conversion)::BIGINT AS conversions,
    sum(exposure)::BIGINT AS exposed
FROM criteo
GROUP BY treatment
ORDER BY treatment;
