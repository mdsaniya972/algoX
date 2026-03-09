# Import Libraries
# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# -----------------------------
# Step 1: Create Sample Dataset
# -----------------------------

np.random.seed(42)

data_size = 500

data = pd.DataFrame({
    "age": np.random.randint(20, 70, data_size),
    "bmi": np.random.uniform(18, 35, data_size),
    "systolic_bp": np.random.randint(100, 180, data_size),
    "diastolic_bp": np.random.randint(60, 110, data_size),
    "smoking": np.random.randint(0, 2, data_size),
    "alcohol": np.random.randint(0, 2, data_size),
    "physical_activity": np.random.randint(1, 5, data_size),
    "salt_intake": np.random.randint(1, 5, data_size)
})

# Hypertension Stage Labels
conditions = [
    (data["systolic_bp"] < 120) & (data["diastolic_bp"] < 80),
    (data["systolic_bp"] < 130) & (data["diastolic_bp"] < 80),
    (data["systolic_bp"] < 140) | (data["diastolic_bp"] < 90),
    (data["systolic_bp"] >= 140) | (data["diastolic_bp"] >= 90)
]

values = [0, 1, 2, 3]  # Hypertension stages

data["hypertension_stage"] = np.select(conditions, values)

# -----------------------------
# Step 2: Prepare Data
# -----------------------------

X = data.drop("hypertension_stage", axis=1)
y = data["hypertension_stage"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 3: Logistic Regression
# -----------------------------

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# -----------------------------
# Step 4: Random Forest Model
# -----------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# -----------------------------
# Step 5: XGBoost Model
# -----------------------------

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("XGBoost Accuracy:")
print(accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# -----------------------------
# Step 6: Model Comparison
# -----------------------------

print("\nModel Comparison")

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

print("XGBoost Accuracy:",
      accuracy_score(y_test, xgb_pred))

# -----------------------------
# Step 7: Predict New Patient
# -----------------------------

new_patient = np.array([[45, 27.5, 135, 88, 1, 0, 3, 2]])

new_patient_scaled = scaler.transform(new_patient)

prediction = rf_model.predict(new_patient_scaled)

stages = {
    0: "Normal",
    1: "Elevated",
    2: "Hypertension Stage 1",
    3: "Hypertension Stage 2"
}

print("\nPredicted Hypertension Stage:",
      stages[int(prediction[0])])