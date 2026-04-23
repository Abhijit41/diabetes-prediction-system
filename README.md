# 🩺 Diabetes Prediction System

A machine learning project that predicts the likelihood of diabetes in patients based on clinical health attributes. Built using Python and scikit-learn, this project covers the complete data science pipeline — from raw data cleaning to model evaluation and comparison.

---

## 📌 Project Overview

Diabetes is one of the most prevalent chronic conditions globally, yet many cases go undetected until significant damage has already occurred. Early and accurate prediction using routine clinical data can enable timely intervention and better patient outcomes.

This project builds and compares multiple machine learning classifiers on a real-world clinical dataset to identify the best-performing model for diabetes prediction. A key focus is on **clinically meaningful evaluation** — not just accuracy, but understanding where each model fails and what type of error is most costly in a health context.

---

## 📂 Dataset

The dataset contains patient health records with the following attributes:

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Gender | Male / Female |
| BMI | Body Mass Index |
| HbA1c Level | Glycated haemoglobin — key diabetes indicator |
| Blood Glucose Level | Fasting blood glucose measurement |
| Smoking History | Never / Former / Current smoker |
| Hypertension | Whether patient has hypertension (0/1) |
| Heart Disease | Whether patient has heart disease (0/1) |
| Diabetes | Target variable — 1 (diabetic) / 0 (non-diabetic) |

---

## 🛠️ Tech Stack

- **Language:** Python 3.13.5
- **Libraries:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## 🔄 Project Pipeline

### 1. Data Cleaning & Preprocessing
- Removed duplicate records
- Handled missing values using appropriate strategies
- Encoded categorical variables (gender, smoking history) using one-hot encoding
- Applied feature scaling (StandardScaler) to normalise numerical attributes such as BMI, HbA1c, and blood glucose
- Checked and resolved class imbalance in the target variable

### 2. Exploratory Data Analysis (EDA)
- Visualised feature distributions across diabetic and non-diabetic groups
- Analysed correlation between features using heatmaps
- Identified the most discriminative features for prediction

### 3. Model Training & Comparison

Four classifiers were implemented and compared:

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of decision trees |
| K-Nearest Neighbours (KNN) | Distance-based classifier |
| Gradient Boosting | Sequential ensemble, typically strongest |

Each model was trained on the preprocessed training set and evaluated on a held-out test set.

### 4. Evaluation Metrics

Models were evaluated using:
- **Accuracy** — overall correctness
- **ROC-AUC Score** — ability to distinguish between classes
- **Confusion Matrix** — breakdown of true/false positives and negatives
- **Classification Report** — precision, recall, and F1-score per class
- **ROC Curves** — plotted for all models for visual comparison

> **Key Insight:** Accuracy alone was misleading in this dataset due to class imbalance. False negatives — missed diabetic cases — are the clinically costliest error. Recall for the positive class was therefore the most important metric, not overall accuracy. This shaped how the best model was selected.

### 5. Results

All trained models were saved for reproducibility. The best-performing model was identified based on ROC-AUC and recall for the diabetic class.

---

## 📊 Sample Outputs

```
Model               Accuracy    ROC-AUC
Logistic Regression   95.5%      0.95
Random Forest         96.91%     0.96
KNN                   95.86%     0.95
Gradient Boosting     94.86%     0.94
```
---

## 💡 Key Learnings

- In health data, **metric selection is a domain reasoning problem**, not a technical default. Optimising for accuracy in an imbalanced clinical dataset gives a false sense of model performance.
- **False negatives are more dangerous than false positives** in disease prediction — a missed diabetic patient has far worse consequences than a false alarm.
- Feature scaling is critical when combining features like age (small range) with blood glucose levels (large range) — without it, distance-based models like KNN perform poorly.
- Preprocessing decisions upstream of the model significantly impact final performance.

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/rajeshkumarswain1976/diabetes_prediction_system.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook



## 👤 Author

**Rajesh Kumar Swain**
B.Tech Computer Science and Engineering
Biju Patnaik Institute of Technology, Rourkela, Odisha

---

## 📄 License

This project is open for academic and educational use.
