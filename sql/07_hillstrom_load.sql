-- Hillstrom email test (second dataset, used to show CUPED with a real pre-period metric).
-- `history` is dollars spent in the previous 12 months, measured before the email was sent.
-- `segment` is the randomized arm: Mens E-Mail, Womens E-Mail or No E-Mail (about 1/3 each).
CREATE OR REPLACE TABLE hillstrom AS
SELECT
    row_number() OVER ()::BIGINT AS row_id,
    segment,
    CASE WHEN segment = 'No E-Mail' THEN 0 ELSE 1 END AS treatment,
    history::DOUBLE AS history,
    recency::INTEGER AS recency,
    visit::UTINYINT AS visit,
    conversion::UTINYINT AS conversion,
    spend::DOUBLE AS spend
FROM read_csv('{{CSV}}', header = true);
