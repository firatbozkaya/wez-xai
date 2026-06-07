# WEZ-XAI

**Firat Bozkaya**  
Student No: **24411004**  
Galatasaray University, Institute of Science  
INF539 - Explainable Artificial Intelligence

This repository contains the project report and reproducible code for an explainable machine learning study on Weapon Engagement Zone (WEZ) prediction in Beyond-Visual-Range air combat.

Repository: https://github.com/firatbozkaya/wez-xai

## Report

The submitted report is:

```text
report/FiratBozkaya.pdf
```

The report follows the IEEE two-column conference format and includes the problem formulation, dataset audit, methodology, experimental results, XAI analysis, discussion, and conclusion.

## Project Summary

The project formulates WEZ prediction as a supervised regression task on the public Kuroswiski 2024 WEZ dataset. The primary target is `RMax`, with `RNez` evaluated as a portability check.

The workflow compares an inherently interpretable model with black-box models, then explains the selected black-box model using local, global, and model-specific XAI methods.

Models:

- Ridge regression baseline
- Explainable Boosting Machine (EBM)
- Random Forest
- XGBoost
- XGBoost with monotone constraints

Explanation methods:

- LIME for local model-agnostic explanations
- KernelSHAP for local model-agnostic explanations
- ALE for global model-agnostic explanations
- TreeSHAP interactions and monotone constraints for model-specific interpretation

## Key Results

- Best model: XGBoost with monotone constraints
- Factorial OOD MAE: 1.101 nm
- Factorial OOD Hit@2nm: 84.6%
- EBM OOD MAE: 1.754 nm
- Monotone constraints improve OOD MAE by 0.08 nm over unconstrained XGBoost
- Physics-consistency audit: 6/6 audited features pass
- Split-conformal nominal 90% intervals reach 67.1% empirical coverage on the Factorial OOD set

## Dataset

The raw dataset files are stored in `wez_dataset/`:

```text
FactorialExperiment.csv
RandomExperiment_1000_NEZ.csv
RandomExperiment_1000_RMAX.csv
RandomExperiment_1000_WEZ.csv
```

The report uses the Random WEZ file for training and cross-validation, and the Factorial file as an out-of-distribution test set for `RMax`.

## Repository Structure

```text
src/          Reusable data, feature, model, and metric code
notebooks/    Python scripts for EDA, modeling, XAI, and additional analyses
tests/        Integrity tests for data handling, feature engineering, and metrics
report/       IEEE LaTeX report, figures, tables, and bibliography
wez_dataset/  Raw CSV dataset files
run_all.py    End-to-end reproduction script
```

## Reproducibility

Create an environment and install the dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the tests:

```bash
pytest -q
```

Regenerate the processed data, figures, and tables:

```bash
python run_all.py
```

Compile the report with Tectonic:

```bash
cd report
tectonic FiratBozkaya.tex
```

The final PDF will be written to:

```text
report/FiratBozkaya.pdf
```

## Notes

Generated cache files, build logs, local environment files, and private download scripts are intentionally excluded from the repository. The raw CSV files, source code, figures, tables, and final report are included so that the project can be inspected and reproduced.
