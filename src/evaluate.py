import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# Create images folder
# -----------------------------

os.makedirs("../images", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("../data/cleaned_impact_ai.csv")

# -----------------------------
# Remove Student_ID
# -----------------------------

df = df.drop(columns=["Student_ID"])

# -----------------------------
# Load encoder and scaler
# -----------------------------

encoders = joblib.load("../models/label_encoders.pkl")
scaler = joblib.load("../models/scaler.pkl")

# -----------------------------
# Encode categorical columns
# -----------------------------

for column in encoders:

    df[column] = encoders[column].transform(df[column])

# -----------------------------
# Features and Target
# -----------------------------

X = df.drop(columns=["Burnout_Risk_Level"])
y = df["Burnout_Risk_Level"]

# -----------------------------
# Scale features
# -----------------------------

X = scaler.transform(X)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("../models/burnout_model.pkl")

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    "../images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion Matrix saved in images folder.")