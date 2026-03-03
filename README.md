# Hiring Bias and Fairness Analysis (AIF360)

Reproducible project for analyzing hiring bias and applying AIF360 reweighing across multiple protected attributes.

## Project Scope

This repository contains:
- Baseline disparity plots from observed hiring outcomes.
- Model training/evaluation for 5 classifiers.
- Fairness metric computation with AIF360.
- Before/after comparison with reweighing mitigation.
- Protected-attribute analyses for Gender, Age, MentalHealth, Country, and Accessibility.

## Repository Structure

- `hiring_bias_fairness_analysis.py`: Core training + fairness utility functions.
- `generate_hiring_bias_graphs.py`: Pre-mitigation descriptive plots.
- `generate_aif360_gender_protected_plots.py`: Gender-protected AIF360 run.
- `generate_aif360_age_protected_plots.py`: Age-protected AIF360 run.
- `generate_aif360_mental_health_protected_plots.py`: MentalHealth-protected AIF360 run.
- `generate_aif360_country_protected_plots.py`: Country-protected AIF360 run.
- `generate_aif360_accessibility_protected_plots.py`: Accessibility-protected AIF360 run.
- `plots/`: Generated charts and CSV outputs.
- `Hiring_bias_report.docx`: Main report pdf

## Environment

Tested with:
- Python `3.12.10`
- Core dependencies pinned in `requirements.txt`

`aif360` is loaded from the local `./AIF360` folder in this repository.
Setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Expected CSV file:
- `./hiring_bias_dataset.csv`

If your dataset is elsewhere, pass `--dataset-path`.

## Reproduce The Results

Run all commands from the repository root (the folder containing `README.md`).

### 1. Baseline (Before AIF360) plots

```powershell
python generate_hiring_bias_graphs.py --dataset-path "./stackoverflow_full.csv" --out-dir "./plots/Before_AIF360"
```

### 2. Gender-protected AIF360 (before vs after)

```powershell
python generate_aif360_gender_protected_plots.py --dataset-path "./stackoverflow_full.csv" --out-dir "./plots" --sample-size 30000 --random-state 42 --top-countries 15 --min-country-count 20
```

### 3. Age-protected AIF360 (before vs after)

```powershell
python generate_aif360_age_protected_plots.py --dataset-path "./stackoverflow_full.csv" --out-dir "./plots" --sample-size 30000 --random-state 42
```

### 4. MentalHealth-protected AIF360 (before vs after)

```powershell
python generate_aif360_mental_health_protected_plots.py --dataset-path "./stackoverflow_full.csv" --out-dir "./plots" --sample-size 30000 --random-state 42
```

### 5. Country-protected AIF360 (before vs after)

Default privileged country is `United States of America`.

```powershell
python generate_aif360_country_protected_plots.py --dataset-path "./stackoverflow_full.csv" --out-dir "./plots" --sample-size 30000 --random-state 42 --privileged-country "United States of America" --top-countries 15 --min-country-count 20
```

### 6. Accessibility-protected AIF360 (before vs after)

```powershell
python generate_aif360_accessibility_protected_plots.py --dataset-path "./stackoverflow_full.csv" --out-dir "./plots" --sample-size 30000 --random-state 42
```

## Output Locations

After running the scripts, check:
- `plots/`: Primary generated PNG/CSV outputs from scripts.
- `plots/images`: Curated chart images for reporting.
- `plots/metrics_CSV`: Curated metrics tables for reporting.
- `plots/Before_AIF360`: Baseline exploratory outputs.
- `plots/ignore`: Archive/intermediate outputs not required for final grading package.

## Models Used

- Logistic Regression
- Random Forest
- Decision Tree
- KNN
- Linear SVC

## Fairness Metrics Reported

- Statistical Parity Difference (SPD, closer to `0` is fairer)
- Disparate Impact (DI, closer to `1` is fairer)
- Average Odds Difference (closer to `0` is fairer)
- Equal Opportunity Difference (closer to `0` is fairer)
- Accuracy, Precision, Recall, F1

## Notes

- Randomness is controlled by `--random-state` (default `42`).
- Reweighing affects models that accept `sample_weight`; KNN remains unweighted.
- Optional package warnings (for example `fairlearn`) can be ignored for this project.



