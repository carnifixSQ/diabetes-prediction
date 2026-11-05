# 🩺 Diabetes Prediction: Pima Indians Dataset

## 📌 Introduction

This project focuses on predicting diabetes using the **Pima Indians Diabetes Dataset** and machine learning classification algorithms.

The goal is to build a reliable binary classification pipeline while paying particular attention to data quality, leakage-free preprocessing, model comparison, and evaluation metrics relevant to medical diagnostics.

Since missing diabetes cases (false negatives) are particularly important in medical screening, the project prioritizes **Recall, F1-score, and ROC-AUC** rather than relying on accuracy alone.

## 📊 Dataset Overview

The project uses the **Pima Indians Diabetes Database**, containing 768 observations and 9 columns.

The dataset includes diagnostic measurements for female patients of Pima Indian heritage aged 21 and older.

**Features:**

- `Pregnancies` — number of times pregnant
- `Glucose` — plasma glucose concentration (mg/dL)
- `BloodPressure` — diastolic blood pressure (mm Hg)
- `SkinThickness` — triceps skin fold thickness (mm)
- `Insulin` — 2-hour serum insulin (mu U/ml)
- `BMI` — body mass index (kg/m²)
- `DiabetesPedigreeFunction` — diabetes pedigree function
- `Age` — age (years)

**Target variable:**

- `Outcome` — 0 = No Diabetes, 1 = Diabetes

## 🔍 Exploratory Data Analysis (EDA)

The exploratory analysis revealed several important characteristics of the dataset:

- **Class balance:** 65.1% No Diabetes and 34.9% Diabetes.
- **Missing values:** zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were treated as missing because they represent physiologically invalid measurements.
- **Missing data distribution:** `Insulin` has approximately 49% missing values, while `SkinThickness` has approximately 30%. Other affected features have less than 5%.
- **Skewness:** `Insulin` has a strong right-skewed distribution (skewness = 2.17). A `log1p` transformation reduces skewness to approximately -0.09.
- **Outliers:** the IQR method was used to identify outliers. High insulin measurements were retained because they may represent genuine observations rather than errors.

## ⚙️ Data Preprocessing

A leakage-free preprocessing pipeline was implemented.

Main steps:

1. Replacing physiologically invalid zero values with `NaN`.
2. Splitting the dataset into training and testing sets (80/20).
3. Stratifying the split by the target variable `Outcome`.
4. Applying median imputation to missing values.
5. Standardizing numerical features using `StandardScaler`.

**Important:** Imputation and scaling are fitted on the training data only. The same transformations are subsequently applied to the test data.

This prevents information from the test set from leaking into the training process.

### Missing-Data Strategy Comparison

Several missing-data strategies were evaluated using repeated stratified cross-validation (5 folds × 5 repeats).

The experiments compared:

- Median imputation.
- KNN imputation with and without feature scaling.
- Excluding the `Insulin` feature.
- Applying `log1p` transformation to `Insulin`.
- Winsorizing extreme insulin values.
- HistGradientBoosting with missing values handled natively.

The results showed that all Logistic Regression variants performed similarly, with ROC-AUC around 0.837–0.839.

Removing `Insulin` did not significantly reduce predictive performance. Median imputation with scaling was therefore selected as a simpler preprocessing strategy.

## 🤖 Model Training and Comparison

Three individual classification algorithms and an ensemble model were evaluated:

- **Logistic Regression (LogReg)**
- **Random Forest**
- **HistGradientBoosting (HistGB)**
- **Ensemble** — combined predictions from multiple models.

The dataset was divided into:

- Training set: 614 observations.
- Test set: 154 observations.

Model comparison was performed using cross-validation on the training set.

### Cross-Validation Results

| Model | ROC-AUC | ROC-AUC Std | PR-AUC |
|---|---:|---:|---:|
| LogReg | 0.8419 | 0.0069 | 0.7232 |
| Random Forest | 0.8360 | 0.0031 | 0.7099 |
| HistGB | 0.8270 | 0.0056 | 0.7045 |
| Ensemble | 0.8421 | 0.0039 | 0.7190 |

**Selected model: Logistic Regression.**

Although the Ensemble achieved a marginally higher ROC-AUC (0.8421 vs. 0.8419), the difference is negligible. Logistic Regression was selected as a simpler and more interpretable model with comparable predictive performance.

## 🎯 Classification Threshold Optimization

The default classification threshold of 0.5 is not necessarily optimal for medical screening.

A lower threshold increases the number of correctly identified diabetes cases (Recall), but also increases the number of false positive predictions.

To address this trade-off, the classification threshold was selected using cross-validation predictions on the training set.

The target Recall was set to **0.85**.

### Threshold Comparison

| Target Recall | Threshold | Actual Recall | Specificity | Precision |
|---:|---:|---:|---:|---:|
| 0.95 | 0.137 | 0.950 | 0.472 | 0.491 |
| 0.90 | 0.188 | 0.900 | 0.562 | 0.523 |
| 0.85 | 0.225 | 0.850 | 0.631 | 0.552 |
| 0.80 | 0.276 | 0.800 | 0.707 | 0.594 |
| 0.75 | 0.326 | 0.750 | 0.760 | 0.626 |
| 0.70 | 0.368 | 0.700 | 0.794 | 0.646 |
| 0.60 | 0.492 | 0.602 | 0.881 | 0.730 |

**Selected threshold: 0.225.**

This threshold provides a compromise between detecting as many diabetes cases as possible and limiting false positive predictions.

## 🧪 Final Test Results

The selected Logistic Regression pipeline was evaluated on the previously unseen test set.

**Test set composition:**
- Total observations: 154
- No Diabetes: 100
- Diabetes: 54

### Overall Performance

| Metric | Result |
|---|---:|
| ROC-AUC | 0.820 |
| Accuracy | 0.721 |
| Precision (Diabetes) | 0.565 |
| Recall (Diabetes) | 0.889 |
| F1-score (Diabetes) | 0.691 |
| Specificity | 0.630 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Diabetes | 0.913 | 0.630 | 0.746 | 100 |
| Diabetes | 0.565 | 0.889 | 0.691 | 54 |
| **Macro Avg** | **0.739** | **0.759** | **0.718** | **154** |
| **Weighted Avg** | **0.791** | **0.721** | **0.726** | **154** |

### 🔬 False Positives and False Negatives

At the selected threshold of 0.225:

- **False Negatives (FN): 6 out of 54** — 6 diabetes cases were missed.
- **False Positives (FP): 37 out of 100** — 37 patients without diabetes were incorrectly classified as positive.

For comparison, at the default threshold of 0.5, the model produced 9 false negatives and 35 false positives.

The optimized threshold therefore reduced the number of missed diabetes cases from 9 to 6, at the cost of 2 additional false positive predictions.

### Confidence Intervals

The estimated performance on the test set:

- **Recall:** 0.889 (95% CI: 0.78–0.95)
- **Specificity:** 0.630 (95% CI: 0.53–0.72)

These intervals illustrate the uncertainty associated with evaluating a model on a relatively small test sample.

## 📈 Key Findings

1. Logistic Regression achieved the best balance of predictive performance and model simplicity.
2. The Ensemble model achieved a nearly identical ROC-AUC, but did not demonstrate a meaningful advantage.
3. Median imputation proved sufficient despite substantial missingness in the `Insulin` feature.
4. Threshold optimization increased Recall from the default-threshold result of 0.833 to 0.889 on the test set.
5. The optimized model missed 6 diabetes cases instead of 9, although the number of false positives increased slightly.
6. The final ROC-AUC of 0.820 indicates reasonable discrimination between the two classes, but further improvements are possible.

## 🚀 Future Improvements

- Hyperparameter optimization for individual models.
- Further investigation of feature importance and model interpretability.
- Additional feature engineering.
- Testing alternative classification algorithms.
- Calibration of predicted probabilities.
- Evaluation on larger and more diverse datasets.
- Further optimization of the Recall–Specificity trade-off.

## 📁 Project Structure

```text
diabetes-prediction/
│
├── data/
│
├── notebooks/
│   ├── .ipynb_checkpoints/
│   │
│   ├── models/
│   │   ├── cv_metrics.csv
│   │   ├── diabetes_logreg_pipeline.joblib
│   │   ├── diabetes_randomforest_base_pipeline.joblib
│   │   ├── diabetes_randomforest_pipeline.joblib
│   │   └── test_metrics.json
│   │
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   └── 03_preprocessing_modeling.ipynb
│
├── LICENSE
├── README.md
└── requirements.txt
```

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.

The model is not a medical diagnostic tool and must not be used as a substitute for professional medical evaluation. Its predictions require clinical validation before any real-world application.
