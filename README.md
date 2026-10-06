# Diabetes Prediction: Pima Indians Dataset

## Introduction
This project prepares the Pima Indians Diabetes dataset for binary classification of diabetes from diagnostic measures. The focus is on a clean, leakage-free preprocessing workflow: handling medically invalid zero values, exploring distributions and outliers, and splitting and transforming the data correctly so that later model evaluation is unbiased.

Modeling is the next stage and is not part of this repository yet.

## Pipeline Overview

### Missing Value Handling
The dataset contains invalid zero values in physiological columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`), where a value of 0 is physically impossible.
- **Approach:** zeros are first mapped to `NaN`.
- **Scale of the problem:** `Insulin` is missing in about 49% of rows and `SkinThickness` in about 30%. `BloodPressure`, `BMI` and `Glucose` are missing in under 5%.
- **Imputation:** median imputation, fitted on the training set only. A comparison of strategies (see below) showed that KNN imputation gives no measurable benefit on this dataset, so the simpler method was chosen.

### Exploratory Data Analysis (EDA)
- **Class balance:** 65.1% no diabetes, 34.9% diabetes (moderate imbalance).
- **Histograms** by outcome reveal strong right skew in `Insulin` (skewness 2.17, reduced to -0.09 by a `log1p` transform).
- **IQR rule** (`Q1 - 1.5 * IQR`, `Q3 + 1.5 * IQR`) is used to count outliers. `Glucose` has none; `Insulin` has 24 high values, which look like genuine measurements rather than errors, so they are kept.

### Leakage-Preventing Preprocessing
- **Train/test split** (80/20, stratified by `Outcome`) is done *before* any imputation or scaling.
- **Median imputation** and **feature scaling** (`StandardScaler`, z-score: `z = (x - mean) / std`) are fitted on the training set only and then applied to the test set, so no information from the test set leaks into training.

### Missing-Data Strategy Comparison
To decide how to handle missing values, several strategies were compared using 5×5 repeated stratified cross-validation, with all preprocessing fitted inside the pipeline to avoid data leakage.

| Model / strategy | ROC-AUC | AUC std | F1 | Recall | Accuracy |
|---|---|---|---|---|---|
| LR, KNN (unscaled), Insulin raw | 0.8385 | 0.0303 | 0.6270 | 0.5627 | 0.7669 |
| LR, KNN (scaled), no Insulin | 0.8382 | 0.0306 | 0.6285 | 0.5664 | 0.7669 |
| LR, KNN (scaled), Insulin log1p | 0.8376 | 0.0303 | 0.6251 | 0.5656 | 0.7638 |
| LR, KNN (scaled), Insulin raw | 0.8375 | 0.0301 | 0.6290 | 0.5656 | 0.7677 |
| LR, KNN (scaled), Insulin winsorized | 0.8374 | 0.0304 | 0.6257 | 0.5649 | 0.7648 |
| LR, median imputation | 0.8371 | 0.0298 | 0.6272 | 0.5612 | 0.7680 |
| Random Forest, KNN (scaled) | 0.8301 | 0.0305 | 0.6335 | 0.5963 | 0.7601 |
| Random Forest, median imputation | 0.8295 | 0.0308 | 0.6457 | 0.6142 | 0.7651 |
| HistGradientBoosting, NaN as is | 0.8108 | 0.0300 | 0.6379 | 0.6210 | 0.7549 |
| HistGradientBoosting, no Insulin | 0.8100 | 0.0331 | 0.6261 | 0.6081 | 0.7474 |

**Takeaway:** all logistic regression variants perform the same within noise (AUC ≈ 0.837–0.839, std ≈ 0.03), and removing `Insulin` does not hurt performance. Simple imputation with scaling is sufficient.

## Next Steps
- Train and tune models on the prepared data.
- In medical diagnostics a false negative (a missed case of diabetes) is far costlier than a false positive, so evaluation will prioritize **Recall**, **F1-score** and **ROC-AUC** over raw accuracy, including classification threshold tuning.

## Data Overview
The project uses the **Pima Indians Diabetes Database** (768 instances, 9 columns), containing diagnostic measures for female patients of Pima Indian heritage aged 21 and older.

**Features:**
- `Pregnancies`: number of times pregnant
- `Glucose`: plasma glucose concentration (mg/dL)
- `BloodPressure`: diastolic blood pressure (mm Hg)
- `SkinThickness`: triceps skin fold thickness (mm)
- `Insulin`: 2-hour serum insulin (mu U/ml)
- `BMI`: body mass index (kg/m²)
- `DiabetesPedigreeFunction`: diabetes pedigree function (genetic likelihood)
- `Age`: age (years)

**Target variable:**
- `Outcome`: class variable (0 = no diabetes, 1 = diabetes)

## Project Structure
```text
project/
├── data/
│   └── pima-indians-diabetes.csv   # Raw dataset
├── notebooks/
│   ├── 01_data_loading.ipynb       # Data loading, zero-to-NaN mapping
│   ├── 02_eda.ipynb                # Distributions, skewness, outlier detection
│   └── 03_preprocessing.ipynb      # Train/test split, imputation, scaling
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```
