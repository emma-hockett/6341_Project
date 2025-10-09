# Project Overview

This repository contains our group project for the Applied Machine Learning course.
We’re predicting mortgage loan denials using the 2024 HMDA dataset.

The repo is structured for collaborative development, reproducibility, and clarity.
Below is a quick guide to how everything fits together.

## Workflow Summary

1. Initial Setup

   These instructions will help with the first time setup of your local development environment.

   1. Git Authentication
      <br/><br/>
      Once you have access to the team project, follow the instructions at https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account to set up SSH authentication. You will need this to push changes from your local computer to the remote GitHub repository.<br/><br/>
   2. Local Git Clone
      <br/><br/>
      After your authentication is set up, the next step is to set up a local clone of the project repository. Instructions available at https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository. 
      <br/><br/>
   3. Install Dependencies
      <br/><br/>
      Once you have the repo cloned, you'll need to install the required packages to execute the code.  Run the command below.  If you are using a virtual environment (venv), note that you'll need to run the command within the venv.
      ```bash
      pip install -r requirements.txt
      ```
   4. Data Download
      <br/><br/>
      Given the size of the dataset, we are not committing it to GitHub.  Instead, the files are available on OneDrive.  The raw dataset has been converted to Parquet - the file size is 90% smaller - so download the Parquet version for initial data exploration and processing.
      <br/><br/>
      The /data directories are replicated at https://cometmail-my.sharepoint.com/:f:/g/personal/dal425714_utdallas_edu/Es1MVn6LCCtJrHwz3lLyNdkBzEjYYGaxvWqGu2Ojncywfg?e=XeJlY4. The initial Parquet file is located at data/interim/2024_combined_mlar_header.parquet. Download and place in the corresponding folder in your local git project.
      <br/><br/>
2. Project Updates
   <br/><br/>
   When adding or updating files within the project, follow these steps.
   <br/><br/>
   1. Data Updates
      <br/><br/>
      Given the size of the data files, we can store them in the OneDrive folder above.  If you've created a new data file, upload it to OneDrive and update the README.md file to indicate usage.
      <br/><br/>
   2. Dependency Updates
      <br/><br/>
      If you write code that requires a new library, add the library to the requirements.txt.  This will help others on the team quickly install updates and recreate your code.

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