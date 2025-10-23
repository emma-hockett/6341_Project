# EDA Summary
_Last updated: 2025-10-23_

## Dataset Overview
- Source: HMDA-like loan application dataset (2024 slice used in notebook).
- Features in schema (retained in modeling table): **88**.
- Notable high-cardinality identifiers: `lei`, geographic codes (`census_tract`, `county_code`, `state_code`).
- Rows: (refer to notebook for exact count; large-scale dataset ~8M rows).

## Target Balance & Baseline
Target is binary (denied_flag). There is class imbalance as only 24.3% of applications are denied.

## Missing Value Patterns
Top missingness (first 15 features):
- `co_applicant_ethnicity_5`: 99.99% missing
- `co_applicant_ethnicity_4`: 99.99% missing
- `applicant_ethnicity_5`: 99.99% missing
- `applicant_ethnicity_4`: 99.99% missing
- `co_applicant_race_5`: 99.99% missing
- `applicant_race_5`: 99.98% missing
- `co_applicant_race_4`: 99.98% missing
- `co_applicant_ethnicity_3`: 99.95% missing
- `applicant_race_4`: 99.94% missing
- `applicant_ethnicity_3`: 99.86% missing
- `co_applicant_race_3`: 99.84% missing
- `multifamily_affordable_units`: 99.73% missing
- `applicant_race_3`: 99.56% missing
- `co_applicant_race_2`: 97.86% missing
- `co_applicant_ethnicity_2`: 97.36% missing

Practical treatment:
- Transform ethnicity/race slots 1–5 into multi-hot encoding
- Treat “not applicable” program features as categorical levels (e.g., `intro_rate_period` NA → fixed-rate).
- Add missing flags and impute for key numerics: `income`, `property_value`, `loan_term`, `debt_to_income_ratio`, `combined_loan_to_value_ratio` (see recommendations).

## Categorical Target Association
(Top associations by Cramér’s V)
- `debt_to_income_ratio`: Cramér’s V = 0.468
- `loan_purpose`: Cramér’s V = 0.251
- `applicant_credit_scoring_model`: Cramér’s V = 0.234
- `initially_payable_to_institution`: Cramér’s V = 0.228
- `census_tract`: Cramér’s V = 0.208
- `manufactured_home_secured_property_type`: Cramér’s V = 0.189
- `manufactured_home_land_property_interest`: Cramér’s V = 0.182
- `construction_method`: Cramér’s V = 0.173
- `lien_status`: Cramér’s V = 0.165
- `co_applicant_credit_scoring_model`: Cramér’s V = 0.149
- `open_end_line_of_credit`: Cramér’s V = 0.147

> Flags: very strong association for `debt_to_income_ratio`, and moderate for product/program features (`loan_purpose`, credit scoring model, manufactured home fields).

## Numeric–Target Association (Point-biserial)
- `intro_rate_period`: r = -0.068 (p=0.0e+00)
- `multifamily_affordable_units`: r = -0.065 (p=4.4e-24)
- `loan_term`: r = -0.049 (p=0.0e+00)
- `loan_amount`: r = -0.025 (p=0.0e+00)
- `property_value`: r = -0.006 (p=0.0e+00)
- `combined_loan_to_value_ratio`: r = 0.001 (p=2.2e-03)
- `income`: r = 0.001 (p=9.4e-02)

> Note: Small magnitudes are expected given heavy skew and program effects. Signal often strengthens after log transforms (e.g., `loan_amount`, `property_value`, `income`).

## Categorical Spread of Denial Rates
Features with large within-feature denial-rate range (top examples):
- `county_code`: range = 100.00% over 3223 classes
- `census_tract`: range = 100.00% over 83892 classes
- `prepayment_penalty_term`: range = 100.00% over 36 classes
- `co_applicant_race_5`: range = 86.96% over 16 classes
- `applicant_race_4`: range = 78.66% over 17 classes
- `initially_payable_to_institution`: range = 76.38% over 4 classes
- `applicant_race_5`: range = 75.00% over 17 classes
- `co_applicant_race_3`: range = 70.76% over 17 classes
- `applicant_credit_scoring_model`: range = 63.47% over 15 classes
- `manufactured_home_land_property_interest`: range = 60.41% over 6 classes
- `co_applicant_race_2`: range = 59.29% over 17 classes

## Redundancy / Multicollinearity
- Numeric–numeric high-correlation pairs were screened; no pairs above threshold were found. Expect mathematical linkage among `loan_amount`, `property_value`, and CLTV-derived quantities.

## Transformations & Encodings Needed
- **Log-scale**: `loan_amount`, `income`, `property_value` (right-skewed). Keep raw + log if model family benefits both.
- **Outlier handling**: clip winsorize at 99.5–99.9% for heavy-tailed monetary fields; treat single extreme values as data errors where documented (e.g., `intro_rate_period` > 60 months → NA).
- **Categorical encoding**: one-hot encode low-cardinality categoricals; consider collapsing rare categories (<0.1%) into “Other” per earlier flags; maintain `_exempt` boolean flags as separate features.
- **Missing flags + contextual imputation**:

  - `income`: add `income_missing`; impute by **median within loan_type** or via **loan_amount ÷ median (loan_amount/income)** by segment.

  - `property_value`: add flag; impute via **loan_amount ÷ median LTV** by segment.

  - `loan_term`: add flag; impute **360** for closed-end loans; leave NA for open-end/reverse mortgage.


## Bias / Sampling Considerations
- Missing income and property value vary by **loan_type**, suggesting programmatic missingness rather than random error; keep missing flags to capture this structure.
- Extremely high denial/approval rates in certain categories (perfect classes) may reflect product definitions or institutional policies; review for potential **target leakage**.
- Geographic features (tract/county) have very high cardinality; avoid one-hot; consider dropping if not central to the modeling objective.

