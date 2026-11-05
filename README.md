# 🩺 Diabetes Prediction — Pima Indians Dataset

## 📌 Introduction

This project focuses on predicting diabetes using the **Pima Indians Diabetes Dataset** and machine learning classification algorithms.

The main objective is to develop a complete machine learning pipeline, including data exploration, preprocessing, model comparison, classification threshold optimization, and final evaluation on an independent test set.

Special attention is given to medical screening requirements, where missing a positive case (False Negative) can be more costly than incorrectly identifying a healthy patient (False Positive).

Therefore, the project prioritizes **Recall, Specificity, F1-score, and ROC-AUC** rather than relying on accuracy alone.

## 📊 Dataset Overview

The project uses the **Pima Indians Diabetes Database**, containing 768 observations and 9 columns.

The dataset includes diagnostic measurements for female patients of Pima Indian heritage aged 21 and older.

**Features:**

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age (years) |

**Target variable:**

- `Outcome` — 0 = No Diabetes, 1 = Diabetes.

## 🔍 Exploratory Data Analysis (EDA)

The exploratory analysis was performed to understand feature distributions, missing values, class balance, and potential outliers.

### Class Distribution

- No Diabetes: 65.1%
- Diabetes: 34.9%

The dataset has a moderate class imbalance.

### Missing Value Handling

The following features contain physiologically invalid zero values:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

These values were replaced with `NaN` before further analysis.

Missing data distribution:

- `Insulin`: approximately 49%.
- `SkinThickness`: approximately 30%.
- `BloodPressure`, `BMI`, and `Glucose`: less than 5% each.

### Distribution Analysis

Histograms and boxplots were used to examine numerical feature distributions across both target classes.

A strong right skew was observed in `Insulin`:

- Original skewness: approximately 2.17.
- After `log1p`: approximately -0.09.

### Outlier Detection

The Interquartile Range (IQR) method was used:

\[
IQR = Q_3-Q_1
\]

\[
Lower=Q_1-1.5\cdot IQR
\]

\[
Upper=Q_3+1.5\cdot IQR
\]

Extreme values were investigated rather than automatically removed.

High insulin measurements were retained because they may represent genuine physiological observations rather than measurement errors.

Additional visualizations included:
- Feature distributions by target class.
- Boxplots for numerical features.
- Correlation matrix.

## ⚙️ Data Preprocessing

A leakage-free preprocessing workflow was implemented using Scikit-learn pipelines.

### Train/Test Split

The dataset was divided into:

| Dataset | Observations | Features |
|---|---:|---:|
| Training | 614 | 8 |
| Testing | 154 | 8 |

An 80/20 stratified split was used with:

- `test_size = 0.20`
- `random_state = 42`
- `stratify = y`

The test set contains 54 positive diabetes cases.

### Missing Value Imputation

Median imputation was selected as the primary strategy.

Additionally, `SimpleImputer` uses `add_indicator=True`, allowing models to receive information about which features originally contained missing values.

### Feature Transformations

For Logistic Regression, a logarithmic transformation is applied to three features:

- `Insulin`
- `DiabetesPedigreeFunction`
- `Age`

The transformation is:

\[
x'=\log(1+x)
\]

After transformation, missing values are imputed using the training-set median, and features are standardized using `StandardScaler`.

Random Forest uses median imputation with missing-value indicators but does not require feature scaling.

HistGradientBoosting handles missing values natively.

**Important:** All learned preprocessing operations are fitted within the training folds during cross-validation or on the complete training set during final model fitting. The test set is never used to fit preprocessing transformations.

## 🤖 Model Training and Comparison

Four classification approaches were evaluated:

1. **Logistic Regression (LogReg)**
2. **Random Forest**
3. **HistGradientBoosting (HistGB)**
4. **Ensemble (Soft Voting)**

### Model Configuration

| Model | Main Configuration |
|---|---|
| Logistic Regression | C=1.0, max_iter=2000 |
| Random Forest | 300 trees, min_samples_leaf=5, max_features=sqrt |
| HistGradientBoosting | max_depth=3, learning_rate=0.05, max_iter=100 |
| Ensemble | Soft Voting of all three classifiers |

The Ensemble combines predicted class probabilities from the three individual models.

### Cross-Validation Strategy

Model comparison was performed using repeated stratified 5-fold cross-validation on the training set.

- 5 folds per repetition.
- 5 independent repetitions.
- 25 total fold-level training runs per model.
- Out-of-fold probability predictions were used for evaluation.

The following metrics were calculated:

- **ROC-AUC:** ability to distinguish between the two classes.
- **ROC-AUC Std:** standard deviation across the five repeated cross-validation results.
- **Average Precision (AP):** summarizes precision-recall performance.

### Cross-Validation Results

| Model | ROC-AUC | ROC-AUC Std | AP |
|---|---:|---:|---:|
| LogReg | 0.8419 | 0.0069 | 0.7232 |
| Random Forest | 0.8360 | 0.0031 | 0.7099 |
| HistGB | 0.8270 | 0.0056 | 0.7045 |
| Ensemble | 0.8421 | 0.0039 | 0.7190 |

### Model Selection

<badge color="success">Selected model: Logistic Regression</badge>

Although Ensemble achieved the highest ROC-AUC (0.8421), Logistic Regression achieved a nearly identical result (0.8419).

A tolerance of `AUC_TOL = 0.005` was introduced to avoid selecting a more complex model for a negligible improvement.

The selection rule chooses the first model in the candidate list whose ROC-AUC is within 0.005 of the maximum.

Logistic Regression therefore provides a simpler and more interpretable solution without a meaningful loss in predictive performance.

## 🎯 Classification Threshold Optimization

The default classification threshold of 0.5 is not necessarily optimal for medical screening.

A lower threshold increases Recall, allowing the model to identify more positive cases, but also increases the number of false positive predictions.

To find a suitable operating point, out-of-fold predictions from repeated cross-validation were pooled on the training set.

The threshold was selected using the ROC curve.

The objective was to find the **highest possible threshold that achieves the target Recall**.

### Threshold Comparison

The target Recall was varied to demonstrate the trade-off between sensitivity and specificity.

| Target Recall | Threshold | Actual Recall | Specificity | Precision |
|---:|---:|---:|---:|---:|
| 0.95 | 0.137 | 0.950 | 0.472 | 0.491 |
| 0.90 | 0.188 | 0.900 | 0.562 | 0.523 |
| 0.85 | 0.225 | 0.850 | 0.631 | 0.552 |
| 0.80 | 0.276 | 0.800 | 0.707 | 0.594 |
| 0.75 | 0.326 | 0.750 | 0.760 | 0.626 |
| 0.70 | 0.368 | 0.700 | 0.794 | 0.646 |
| 0.60 | 0.492 | 0.602 | 0.881 | 0.730 |

### Selected Operating Point

- Target Recall: **0.85**
- Selected threshold: **0.225**

This operating point was chosen to prioritize detecting diabetes cases while maintaining a reasonable level of specificity.

The threshold was determined exclusively from training data and was not optimized using test-set results.

## 🧪 Final Test Evaluation

The selected Logistic Regression pipeline was fitted on the complete training set and evaluated on the previously unseen test set.

### Test Set Composition

| Class | Observations |
|---|---:|
| No Diabetes | 100 |
| Diabetes | 54 |
| **Total** | **154** |

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
| Macro Average | 0.739 | 0.759 | 0.718 | 154 |
| Weighted Average | 0.791 | 0.721 | 0.726 | 154 |

### Confusion Matrix

| Actual / Predicted | No Diabetes | Diabetes |
|---|---:|---:|
| No Diabetes | 63 | 37 |
| Diabetes | 6 | 48 |

### False Positives and False Negatives

At the selected threshold of 0.225:

- **True Positives (TP):** 48
- **True Negatives (TN):** 63
- **False Positives (FP):** 37
- **False Negatives (FN):** 6

Compared with the previous default-threshold evaluation:

| Metric | Previous threshold | Optimized threshold |
|---|---:|---:|
| False Negatives | 9 | 6 |
| False Positives | 35 | 37 |

The optimized threshold reduced missed diabetes cases from 9 to 6, while increasing false positive predictions by 2.

### Confidence Intervals

Wilson 95% confidence intervals were calculated for Recall and Specificity.

| Metric | Value | 95% Confidence Interval |
|---|---:|---:|
| Recall | 0.889 | 0.78–0.95 |
| Specificity | 0.630 | 0.53–0.72 |

These intervals demonstrate the uncertainty associated with evaluating a model on a relatively small test sample.

## 📈 Visual Evaluation

The final evaluation includes three visualizations:

1. **Confusion Matrix** — shows correct and incorrect classifications.
2. **ROC Curve** — illustrates the relationship between the True Positive Rate and False Positive Rate.
3. **Precision-Recall Curve** — demonstrates the trade-off between Precision and Recall.

The selected classification threshold is also marked on the ROC and Precision-Recall plots.

## 💾 Saved Artifacts

The project saves the following outputs:

| File | Description |
|---|---|
| `diabetes_logreg_pipeline.joblib` | Trained Logistic Regression pipeline |
| `cv_metrics.csv` | Cross-validation model comparison |
| `test_metrics.json` | Final test metrics, selected threshold, model name and feature names |

The trained pipeline includes preprocessing and classification steps, allowing the same transformations to be applied when making future predictions.

## 🔑 Key Findings

1. Logistic Regression achieved competitive performance compared with more complex models.
2. Ensemble produced a marginally higher ROC-AUC, but the improvement was too small to justify additional complexity.
3. Median imputation with missing-value indicators provided a practical preprocessing strategy.
4. Threshold optimization increased Recall compared with the default threshold.
5. The final model identified 48 out of 54 positive cases in the test set.
6. The optimized operating point reduced False Negatives at the cost of additional False Positives.
7. The final ROC-AUC of 0.820 indicates reasonable discrimination, although further validation and improvement are necessary.

## 🚀 Future Improvements

- Hyperparameter optimization.
- Feature importance analysis and model interpretability using SHAP.
- Probability calibration.
- Further investigation of missing-data strategies.
- Evaluation on larger and more diverse datasets.
- Additional threshold optimization based on different screening requirements.
- External validation to assess generalization.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.

The model is not a medical diagnostic tool and must not be used as a substitute for professional medical evaluation.

Its predictions require further clinical validation before any real-world application.
