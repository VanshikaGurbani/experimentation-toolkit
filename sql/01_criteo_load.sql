-- Load the raw Criteo uplift file into a typed table with a stable row id.
-- {{CSV}} is replaced with the path of the (gzipped) CSV by expkit.data.
-- f0..f11 are anonymized user features. `treatment` is the randomized assignment (about 85%
-- treated). `visit` and `conversion` are outcomes. `exposure` happens AFTER assignment (only
-- some treated users actually saw the ad), so it must never be used as a model feature.
CREATE OR REPLACE TABLE criteo AS
SELECT
    row_number() OVER ()::BIGINT AS row_id,
    f0::FLOAT AS f0, f1::FLOAT AS f1, f2::FLOAT AS f2, f3::FLOAT AS f3,
    f4::FLOAT AS f4, f5::FLOAT AS f5, f6::FLOAT AS f6, f7::FLOAT AS f7,
    f8::FLOAT AS f8, f9::FLOAT AS f9, f10::FLOAT AS f10, f11::FLOAT AS f11,
    treatment::UTINYINT AS treatment,
    conversion::UTINYINT AS conversion,
    visit::UTINYINT AS visit,
    exposure::UTINYINT AS exposure
FROM read_csv('{{CSV}}', header = true);
