-- Deterministic 40 / 10 / 50 split by hashing the row id (train, validation, test).
-- Train fits models and covariate adjustments, validation is for early stopping only, and
-- test is touched once for evaluation. The hash makes the split reproducible and independent
-- of row order and of the outcome.
CREATE OR REPLACE VIEW criteo_split AS
SELECT
    *,
    CASE
        WHEN hash(row_id) % 100 < 40 THEN 'train'
        WHEN hash(row_id) % 100 < 50 THEN 'val'
        ELSE 'test'
    END AS split
FROM criteo;
