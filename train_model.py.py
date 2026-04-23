import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# Load the dataset
try:
    df = pd.read_csv('diabetes_prediction_dataset.csv')
except FileNotFoundError:
    print("Error: 'diabetes_prediction_dataset.csv' not found.")
    print("Please download the dataset and place it in the same directory as the script.")
    exit()

# ==============================================================================
# DATASET OVERVIEW & PREPARATION
# ==============================================================================
print("="*50)
print("DATASET OVERVIEW & PREPARATION")
print("="*50)

# Remove duplicates and 'Other' gender category
df = df.drop_duplicates()
df = df[df['gender'] != 'Other']
print(f"Dataset shape after cleaning: {df.shape}")

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

df_processed = df.copy()
df_processed = pd.get_dummies(df_processed, columns=['gender', 'smoking_history'], drop_first=True)
print("Categorical encoding completed.")

# ==============================================================================
# TRAIN-TEST SPLIT AND SCALING
# ==============================================================================
print("\n" + "="*50)
print("TRAIN-TEST SPLIT AND SCALING")
print("="*50)

X = df_processed.drop('diabetes', axis=1)
y = df_processed['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

joblib.dump(scaler, 'scaler.joblib')
print("\nFeature scaling completed and scaler saved as 'scaler.joblib'.")

# ==============================================================================
# MODEL TRAINING, EVALUATION, AND SAVING
# ==============================================================================
print("\n" + "="*50)
print("MODEL TRAINING, EVALUATION, AND SAVING")
print("="*50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    model_filename = f'{name.replace(" ", "_").lower()}_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    results[name] = {'accuracy': accuracy, 'auc': auc_score}

    print(f"{name} -> Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    print(f"\nClassification Report for {name}:\n{classification_report(y_test, y_pred)}")

    # =============================
    # Confusion Matrix for Each Model
    # =============================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results).T.sort_values(by='auc', ascending=False)
print("\nModel Performance Summary:")
print(results_df)

# =======================================
# 5. MODEL COMPARISON VISUALIZATION (FIXED)
# =======================================
from sklearn.metrics import roc_curve

# Convert evaluation results to DataFrame
result_df = pd.DataFrame(results).T.sort_values(by="accuracy", ascending=False)
print("\nModel Performance Summary:")
print(result_df)

# Bar plot for Accuracy and ROC-AUC comparison
plt.figure(figsize=(8, 5))
result_df[["accuracy", "auc"]].plot(kind="bar", color=["skyblue", "lightgreen"])
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(["Accuracy", "AUC"], loc="lower right")
plt.tight_layout()
plt.show()

# =======================================
# 6. CONFUSION MATRIX & ROC CURVE (Best Model)
# =======================================
best_model_name = result_df.index[0]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# Confusion Matrix for Best Model
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title(f"Confusion Matrix - {best_model_name} (Best Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve for Best Model
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"{best_model_name} (AUC = {results[best_model_name]['auc']:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.title(f"ROC Curve - {best_model_name}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
