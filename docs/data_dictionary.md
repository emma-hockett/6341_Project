# HMDA 2024 – Data Dictionary

This document describes the typed and cleaned dataset at data/interim/hmda_2024_typed.parquet. It covers every retained column, the binary target mapping, and a list of removed columns.

## Retained columns
## Retained Columns

| column_name | data_type | description | transformations | notes |
|--------------|-----------|--------------|-----------------|-------|
| activity_year | Int16 | Filing year of the HMDA record. | Typed per schema. | Single year in current slice. |
| lei | string[pyarrow] | Legal Entity Identifier for the reporting institution. | Trimmed/typed. | Identifier string; high cardinality. |
| loan_type | category | Type of loan (e.g., conventional, FHA, VA, RHS/FSA). | Cast to categorical. | Encoded numeric category in source. |
| loan_purpose | category | Purpose of the loan (home purchase, home improvement, refinancing). | Cast to categorical. | Encoded numeric category in source. |
| preapproval | category | Whether a preapproval request was involved. | Cast to categorical. | Encoded numeric category in source. |
| construction_method | category | Site-built vs. manufactured housing. | Cast to categorical. | Encoded numeric category in source. |
| occupancy_type | category | Occupancy status of the property (principal, second, investment). | Cast to categorical. | Encoded numeric category in source. |
| loan_amount | Float32 | Loan amount as reported. | Typed; numeric cleaning. | Positive numeric. |
| action_taken | category | Lender’s action/outcome for the application. | Cast to categorical. | Used to derive `approved_flag`. |
| state_code | category | State/territory code of the property. | Cast to categorical; trimmed. | Code string. |
| county_code | category | County code of the property. | Cast to categorical; trimmed. | Code string; has `_exempt` flag. |
| census_tract | category | Census tract identifier for the property. | Cast to categorical; trimmed. | Code string. |
| applicant_ethnicity_1..5 | category | Applicant ethnicity fields (multiple responses). | Cast to categorical. | Encoded numeric categories. |
| co_applicant_ethnicity_1..5 | category | Co-applicant ethnicity fields (multiple responses). | Cast to categorical. | Encoded numeric categories. |
| applicant_ethnicity_observed | category | Whether applicant ethnicity was observed or reported. | Cast to categorical. | Encoded numeric category. |
| co_applicant_ethnicity_observed | category | Whether co-applicant ethnicity was observed or reported. | Cast to categorical. | Encoded numeric category. |
| applicant_race_1..5 | category | Applicant race fields (multiple responses). | Cast to categorical. | Encoded numeric categories. |
| co_applicant_race_1..5 | category | Co-applicant race fields (multiple responses). | Cast to categorical. | Encoded numeric categories. |
| applicant_race_observed | category | Whether applicant race was observed or reported. | Cast to categorical. | Encoded numeric category. |
| co_applicant_race_observed | category | Whether co-applicant race was observed or reported. | Cast to categorical. | Encoded numeric category. |
| applicant_sex | category | Applicant sex. | Cast to categorical. | Encoded numeric category. |
| co_applicant_sex | category | Co-applicant sex. | Cast to categorical. | Encoded numeric category. |
| applicant_sex_observed | category | Whether applicant sex was observed or reported. | Cast to categorical. | Encoded numeric category. |
| co_applicant_sex_observed | category | Whether co-applicant sex was observed or reported. | Cast to categorical. | Encoded numeric category. |
| applicant_age | category | Applicant age (HMDA-coded band). | Cast to categorical; trimmed. | Ordinal by policy, but stored categorical. |
| applicant_age_above_62 | category | Indicator band for age ≥ 62 (HMDA-coded). | Cast to categorical. | Ordinal by policy, but stored categorical. |
| co_applicant_age | category | Co-applicant age (HMDA-coded band). | Cast to categorical. | Ordinal by policy, but stored categorical. |
| co_applicant_age_above_62 | category | Indicator band for co-applicant age ≥ 62. | Cast to categorical. | Ordinal by policy, but stored categorical. |
| income | Float64 | Applicant income (thousands USD). | Typed; cleaned negative/sentinel; standardized nulls. | Continuous; zeros retained, negatives fixed. |
| lien_status | category | Lien status (first lien, subordinate, etc.). | Cast to categorical. | Encoded numeric category. |
| applicant_credit_scoring_model | category | Credit score model used for applicant. | Cast to categorical. | Encoded numeric category. |
| co_applicant_credit_scoring_model | category | Credit score model used for co-applicant. | Cast to categorical. | Encoded numeric category. |
| prepayment_penalty_term | category | Prepayment penalty term (coded, may be exempt). | Cast to categorical. | Has `_exempt` flag. |
| debt_to_income_ratio | category | DTI ratio band (coded). | Cast to categorical. | Has `_exempt` flag. |
| combined_loan_to_value_ratio | category | CLTV ratio band (coded). | Cast to categorical. | Has `_exempt` flag. |
| loan_term | category | Loan term (months band/coded). | Cast to categorical. | Has `_exempt` flag. |
| intro_rate_period | category | Introductory rate period (months band/coded). | Cast to categorical. | Has `_exempt` flag. |
| balloon_payment | category | Balloon payment feature. | Cast to categorical. | Encoded numeric category. |
| interest_only_payment | category | Interest-only payment feature. | Cast to categorical. | Encoded numeric category. |
| negative_amortization | category | Negative amortization feature. | Cast to categorical. | Encoded numeric category. |
| other_non_amortizing_features | category | Other non-amortizing features. | Cast to categorical. | Encoded numeric category. |
| property_value | Float32 | Property value (reported). | Typed; exempt split applied. | Continuous; has `_exempt` flag. |
| manufactured_home_secured_property_type | category | Secured property type (for manufactured housing). | Cast to categorical. | Encoded numeric category. |
| manufactured_home_land_property_interest | category | Land property interest (manufactured housing). | Cast to categorical. | Encoded numeric category. |
| total_units | category | Total dwelling units on the property. | Cast to categorical. | Count band/coded. |
| multifamily_affordable_units | category | Number of affordable units (for multifamily). | Cast to categorical; exempt split applied. | Has `_exempt` flag. |
| submission_of_application | category | Submission channel (e.g., direct, retail). | Cast to categorical. | Encoded numeric category. |
| initially_payable_to_institution | category | Whether the loan is initially payable to the institution. | Cast to categorical. | Encoded numeric category. |
| reverse_mortgage | category | Whether the loan is a reverse mortgage. | Cast to categorical. | Encoded numeric category. |
| open_end_line_of_credit | category | Whether the loan is an open-end line of credit. | Cast to categorical. | Encoded numeric category. |
| business_or_commercial_purpose | category | Whether loan purpose is business/commercial. | Cast to categorical. | Encoded numeric category. |
| combined_loan_to_value_ratio_exempt | bool[pyarrow] | Flag: CLTV reported as “Exempt”. | Derived from exempt split. | Companion to `combined_loan_to_value_ratio`. |
| county_code_exempt | bool[pyarrow] | Flag: County code reported as “Exempt”. | Derived from exempt split. | Companion to `county_code`. |
| debt_to_income_ratio_exempt | bool[pyarrow] | Flag: DTI reported as “Exempt”. | Derived from exempt split. | Companion to `debt_to_income_ratio`. |
| intro_rate_period_exempt | bool[pyarrow] | Flag: Intro rate period reported as “Exempt”. | Derived from exempt split. | Companion to `intro_rate_period`. |
| loan_term_exempt | bool[pyarrow] | Flag: Loan term reported as “Exempt”. | Derived from exempt split. | Companion to `loan_term`. |
| multifamily_affordable_units_exempt | bool[pyarrow] | Flag: Affordable units reported as “Exempt”. | Derived from exempt split. | Companion to `multifamily_affordable_units`. |
| prepayment_penalty_term_exempt | bool[pyarrow] | Flag: Prepayment penalty term reported as “Exempt”. | Derived from exempt split. | Companion to `prepayment_penalty_term`. |
| property_value_exempt | bool[pyarrow] | Flag: Property value reported as “Exempt”. | Derived from exempt split. | Companion to `property_value`. |
| approved_flag | bool[pyarrow] | Binary target: lender approval decision for modeling. | Derived from `action_taken` via clean config. | See “Target Variable” section. |
## Target Variable

Column: approved_flag</br>
Definition: True = approved, False = denied (used only for rows where a lender decision exists).

Mapping source: clean.yaml
- Approved: action_taken ∈ {1, 2, 8}
- Denied: action_taken ∈ {3, 7}
- Excluded from modeling: action_taken ∈ {4, 5, 6} (withdrawn, file incomplete, purchased loan)

These sets are specified under clean.action_taken (approved/denied/exclude).

Justification:
Codes 1, 2, and 8 represent lender approvals (loan originated; approved but not accepted; preapproval approved).
Codes 3 and 7 are explicit denials (application denied; preapproval denied).
Codes 4, 5, and 6 do not reflect an underwriting decision outcome on an application (withdrawn, incomplete, purchased) and are therefore excluded from the target.

## Columns Removed During Cleaning

The following fields were dropped per schema.yaml (role: drop). Where relevant, a typical reason is given (pricing/post-decision leakage, auxiliary administrative codes, or out of scope for the modeling target).
- purchaser_type — Secondary market purchaser type (not part of decision; potential leakage).
- rate_spread — Pricing-related; post-decision/derived; out of scope.
- hoepa_status — Compliance flag; not needed for approval modeling.
- denial_reason_1..4 — Reasons present only when denied; post-outcome information (leakage).
- total_loan_costs, total_points_and_fees, origination_charges, discount_points, lender_credits, interest_rate — Pricing/fees that are typically finalized post-decision; potential leakage; also frequently “Exempt” in raw HMDA.
- aus_1..5 — Automated Underwriting System result codes; ancillary and inconsistently populated across reporters.

## Transformations Applied (Summary)
- Whitespace & null-like normalization: Standardized blanks/“NA” tokens to pd.NA.
- Exempt split: For fields that may carry the literal “Exempt”, created paired boolean flags *_exempt and nulled the base value on exempt rows.
- Type enforcement: Converted per schema.yaml dtypes (integers/floats/strings/categoricals/booleans).
- Negative income handling: Negative values corrected (typo fixes).
- Target creation: approved_flag derived from action_taken based on configuration above.

## Access & Review
- File location: docs/data_dictionary.md.
- Source of truth for roles/dtypes: configs/schema.yaml.  ￼
- Target mapping configuration: configs/clean.yaml.