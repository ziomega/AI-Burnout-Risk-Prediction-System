import pandas as pd

print("=" * 60)
print(" IMPACT OF AI ON STUDENTS - DATA PREPROCESSING ")
print("=" * 60)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/impact_ai.csv")

print("\nDataset Loaded Successfully!\n")

# -----------------------------
# Display Dataset
# -----------------------------
print("First Five Rows:\n")
print(df.head())

print("\n" + "=" * 60)

# -----------------------------
# Dataset Information
# -----------------------------
print("\nDataset Information:\n")
df.info()

print("\n" + "=" * 60)

# -----------------------------
# Dataset Shape
# -----------------------------
print("\nRows and Columns")
print(df.shape)

print("\n" + "=" * 60)

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values:\n")
print(df.isnull().sum())

print("\n" + "=" * 60)

# -----------------------------
# Duplicate Rows
# -----------------------------
duplicates = df.duplicated().sum()

print("\nDuplicate Rows =", duplicates)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("Duplicate rows removed.")

print("\n" + "=" * 60)

# -----------------------------
# Fill Missing Numerical Values
# -----------------------------
numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# -----------------------------
# Fill Missing Categorical Values
# -----------------------------
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing values handled successfully.")

print("\n" + "=" * 60)

# -----------------------------
# Column Names
# -----------------------------
print("\nColumns:\n")

for column in df.columns:
    print(column)

print("\n" + "=" * 60)

# -----------------------------
# Save Cleaned Dataset
# -----------------------------
df.to_csv("../data/cleaned_impact_ai.csv", index=False)

print("\nCleaned dataset saved as:")
print("data/cleaned_impact_ai.csv")

print("\nPreprocessing Completed Successfully!")