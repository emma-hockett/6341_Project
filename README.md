# Project Structure

This repository contains our group project for the Applied Machine Learning course.
We’re predicting mortgage loan denials using the 2024 HMDA dataset.

The repo is structured for collaborative development, reproducibility, and clarity.
Below is a quick guide to how everything fits together.

## Directory Overview

### /data

This folder is ignored by Git to prevent huge files from being committed.
Each teammate should create the same subfolder structure locally:

```text
data/
├─ raw/         # Original HMDA CSVs (download from CFPB)
├─ interim/     # Cleaned / typed data saved as Parquet/Feather
└─ processed/   # Final modeling dataset (ready for training)
```
#### Workflow

1.	Download the 2024 HMDA modified LAR CSV and place it in data/raw/.
2.  Run the cleaning and feature engineering scripts in /scripts to generate the processed file.
3.  Everyone uses the same processed dataset for modeling — no need to re-run preprocessing.

Note: These folders contain .gitkeep files so the structure appears in GitHub, but the actual data files are not tracked.

### /src

Reusable Python code lives here, organized as a small internal package. Anything that will be reused across notebooks goes here — this prevents code duplication and merge conflicts.

### /notebooks

Exploratory and narrative notebooks — one per major step:
```text
notebooks/
├─ 01_step_name.ipynb
├─ 02_step_name.ipynb
└─ 03_step_name.ipynb
```
### /scripts

Command-line entry points for reproducible processing. Each script reads settings from /configs.

### /configs

Configuration files (.yaml) specifying:

- File paths and column lists
- Feature inclusion/exclusion
- Model hyperparameters
- Train/validation/test splits

This keeps model runs reproducible without editing code.

### /models

Stores output from training.

### /reports

All generated plots, tables, and figures used in the paper or presentation.
```text
reports/
├─ figures/
└─ tables/
```

## Workflow Summary

1. Data setup  
   - Download raw data → place in `data/raw/`
