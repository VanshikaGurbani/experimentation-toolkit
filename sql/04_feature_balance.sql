-- Randomization check: feature means and standard deviations by arm. If assignment is truly
-- random, standardized differences should be close to zero.
SELECT
    treatment,
    count(*)::BIGINT AS n,
    avg(f0) AS m0, avg(f1) AS m1, avg(f2) AS m2, avg(f3) AS m3, avg(f4) AS m4, avg(f5) AS m5,
    avg(f6) AS m6, avg(f7) AS m7, avg(f8) AS m8, avg(f9) AS m9, avg(f10) AS m10, avg(f11) AS m11,
    stddev_samp(f0) AS s0, stddev_samp(f1) AS s1, stddev_samp(f2) AS s2, stddev_samp(f3) AS s3,
    stddev_samp(f4) AS s4, stddev_samp(f5) AS s5, stddev_samp(f6) AS s6, stddev_samp(f7) AS s7,
    stddev_samp(f8) AS s8, stddev_samp(f9) AS s9, stddev_samp(f10) AS s10,
    stddev_samp(f11) AS s11
FROM criteo
GROUP BY treatment
ORDER BY treatment;
