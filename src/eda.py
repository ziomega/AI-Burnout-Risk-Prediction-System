import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Create images folder if needed
# -----------------------------
os.makedirs("../images", exist_ok=True)

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv("../data/cleaned_impact_ai.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe(include="all"))

# ===================================================
# HISTOGRAMS
# ===================================================

print("\nGenerating Histograms...")

df.hist(figsize=(16,12))

plt.tight_layout()

plt.savefig(
    "../images/histograms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# CORRELATION HEATMAP
# ===================================================

print("Generating Heatmap...")

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.savefig(
    "../images/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# BURNOUT DISTRIBUTION
# ===================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Burnout_Risk_Level"
)

plt.title("Burnout Risk Distribution")

plt.savefig(
    "../images/burnout_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# MAJOR CATEGORY
# ===================================================

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x="Major_Category"
)

plt.xticks(rotation=45)

plt.title("Major Category")

plt.savefig(
    "../images/major_category.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# PRIMARY USE CASE
# ===================================================

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x="Primary_Use_Case"
)

plt.xticks(rotation=45)

plt.title("Primary Use Case")

plt.savefig(
    "../images/primary_use_case.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# PAID SUBSCRIPTION
# ===================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Paid_Subscription"
)

plt.title("Paid Subscription")

plt.savefig(
    "../images/paid_subscription.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# PROMPT ENGINEERING
# ===================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Prompt_Engineering_Skill"
)

plt.title("Prompt Engineering Skill")

plt.savefig(
    "../images/prompt_engineering_skill.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# AI HOURS VS BURNOUT
# ===================================================

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="Burnout_Risk_Level",
    y="Weekly_GenAI_Hours"
)

plt.title("Weekly AI Hours vs Burnout")

plt.savefig(
    "../images/weekly_ai_vs_burnout.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# STUDY HOURS
# ===================================================

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="Burnout_Risk_Level",
    y="Traditional_Study_Hours"
)

plt.title("Traditional Study Hours")

plt.savefig(
    "../images/traditional_study_hours.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# GPA SCATTER
# ===================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Pre_Semester_GPA"],
    df["Post_Semester_GPA"],
    alpha=0.7
)

plt.xlabel("Pre Semester GPA")
plt.ylabel("Post Semester GPA")

plt.title("Pre vs Post Semester GPA")

plt.savefig(
    "../images/gpa_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# SKILL RETENTION
# ===================================================

plt.figure(figsize=(8,6))

sns.histplot(
    data=df,
    x="Skill_Retention_Score",
    kde=True
)

plt.title("Skill Retention Score")

plt.savefig(
    "../images/skill_retention.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================================
# PAIRPLOT
# ===================================================

pair_columns = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Post_Semester_GPA",
    "Skill_Retention_Score"
]

pair_plot = sns.pairplot(df[pair_columns])

pair_plot.savefig(
    "../images/pairplot.png",
    dpi=300
)

plt.show()

# ===================================================
# BOXPLOTS
# ===================================================

numerical_columns = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Traditional_Study_Hours",
    "Post_Semester_GPA",
    "Skill_Retention_Score"
]

for column in numerical_columns:

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x=df[column]
    )

    plt.title(column)

    plt.savefig(
        f"../images/{column}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

print("\nEDA Completed Successfully!")
print("Graphs saved inside the images folder.")