# Diabetes Prediction: Pima Indians Dataset

## Introduction
This repository contains a production-grade machine learning pipeline for the binary classification of diabetes based on diagnostic measures. Rather than simply training a model, this project focuses on building a robust, leakage-free Data Science pipeline: from handling medically invalid missing values to evaluating models with business-aware metrics.

## Pipeline Capabilities

### Robust Data Imputation
The dataset contains invalid zero-values in physiological columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`). 
- **Approach:** Zeros are first mapped to `NaN`. 
- **Algorithm:** Utilizes `KNNImputer` ($K=5$) based on *Euclidean distance*. Unlike naive mean/median imputation, this preserves the multidimensional relationships and local structure of the data by finding the most similar patients to fill the gaps.

### Rigorous Exploratory Data Analysis (EDA)
Systematic outlier detection and distribution analysis. 
- Employs **Histograms** to identify *skewness* (e.g., right-skewed `Insulin` distributions).
- Utilizes **Boxplots** to mathematically define outliers beyond the $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$ thresholds.

### Leakage-Preventing Preprocessing
Strict adherence to machine learning best practices to ensure unbiased evaluation.
- **Train/Test Split:** Executed *before* any feature scaling to prevent **Data Leakage**.
- **Feature Scaling:** Applies `StandardScaler` (Z-score normalization: $z = \frac{x - \mu}{\sigma}$) exclusively to the training fold, then transforms the test fold, ensuring algorithms sensitive to feature magnitude (like KNN or Logistic Regression) perform optimally.

### Business-Aware Model Evaluation
In medical diagnostics, the cost of a **False Negative** (failing to detect diabetes) is significantly higher than a **False Positive**. 
- Evaluation prioritizes **Recall** and **F1-Score** over raw Accuracy, ensuring the model is optimized for real-world clinical utility.

## Data Overview
The project utilizes the **Pima Indians Diabetes Database** (768 instances, 9 features), containing diagnostic measures for female patients of Pima Indian heritage aged 21 and older.

**Features:**
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration (mg/dL)
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- **`SkinThickness`**: Triceps skin fold thickness (mm)
- **`Insulin`**: 2-Hour serum insulin (mu U/ml)
- **`BMI`**: Body mass index (kg/m²)
- **`DiabetesPedigreeFunction`**: Diabetes pedigree function (genetic likelihood)
- **`Age`**: Age (years)

**Target Variable:**
- `Outcome`: Class variable (0 = No Diabetes, 1 = Diabetes)

## Project Structure
```text
project/
├── data/
│   └── pima-indians-diabetes.csv   # Raw dataset
├── notebooks/
│   ├── 01_data_loading.ipynb       # Data ingestion, NaN mapping, KNN Imputation
│   └── 02_eda.ipynb                # Statistical visualization, outlier detection
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
