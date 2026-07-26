import os
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# -------------------------------------
# Create models folder
# -------------------------------------

os.makedirs("../models", exist_ok=True)

# -------------------------------------
# Load Dataset
# -------------------------------------

df = pd.read_csv("../data/cleaned_impact_ai.csv")

print("=" * 60)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 60)

# -------------------------------------
# Remove Student_ID
# -------------------------------------

df = df.drop(columns=["Student_ID"])

print("\nStudent_ID removed.")

# -------------------------------------
# Encode categorical columns
# -------------------------------------

label_encoders = {}

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

print("\nCategorical columns encoded.")

# -------------------------------------
# Features and Target
# -------------------------------------

X = df.drop(columns=["Burnout_Risk_Level"])

y = df["Burnout_Risk_Level"]

# -------------------------------------
# Feature Scaling
# -------------------------------------

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

print("\nFeatures scaled successfully.")

# -------------------------------------
# Train Test Split
# -------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -------------------------------------
# Models
# -------------------------------------

models = {

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        )

}

accuracies = {}

best_model = None

best_accuracy = 0

# -------------------------------------
# Training
# -------------------------------------

for name, model in models.items():

    print("\nTraining", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    accuracies[name] = accuracy

    print(f"Accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

# -------------------------------------
# Results
# -------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model_name, accuracy in accuracies.items():

    print(f"{model_name:<25} {accuracy:.4f}")

print("\nBest Accuracy :", best_accuracy)

print("Best Model    :", best_model)

# -------------------------------------
# Save Model
# -------------------------------------

joblib.dump(best_model, "../models/burnout_model.pkl")

joblib.dump(label_encoders, "../models/label_encoders.pkl")

joblib.dump(scaler, "../models/scaler.pkl")

print("\nModels saved successfully!")