import joblib
import pandas as pd

# -------------------------------------
# Load saved objects
# -------------------------------------

model = joblib.load("../models/burnout_model.pkl")
encoders = joblib.load("../models/label_encoders.pkl")
scaler = joblib.load("../models/scaler.pkl")

print("=" * 60)
print("AI Burnout Risk Prediction")
print("=" * 60)

# -------------------------------------
# User Input
# -------------------------------------

major = input("Major Category: ")
year = input("Year of Study: ")
pre_gpa = float(input("Pre Semester GPA: "))
weekly_ai = float(input("Weekly GenAI Hours: "))
primary_use = input("Primary Use Case: ")
prompt_skill = input("Prompt Engineering Skill: ")
tool_diversity = float(input("Tool Diversity: "))
paid = input("Paid Subscription (True/False): ").strip().lower() == "true"
study_hours = float(input("Traditional Study Hours: "))
dependency = input("Perceived AI Dependency: ")
policy = input("Institutional Policy: ")
anxiety = input("Anxiety Level During Exams: ")
post_gpa = float(input("Post Semester GPA: "))
retention = float(input("Skill Retention Score: "))

# -------------------------------------
# Create DataFrame
# -------------------------------------

sample = pd.DataFrame({
    "Major_Category": [major],
    "Year_of_Study": [year],
    "Pre_Semester_GPA": [pre_gpa],
    "Weekly_GenAI_Hours": [weekly_ai],
    "Primary_Use_Case": [primary_use],
    "Prompt_Engineering_Skill": [prompt_skill],
    "Tool_Diversity": [tool_diversity],
    "Paid_Subscription": [paid],
    "Traditional_Study_Hours": [study_hours],
    "Perceived_AI_Dependency": [dependency],
    "Institutional_Policy": [policy],
    "Anxiety_Level_During_Exams": [anxiety],
    "Post_Semester_GPA": [post_gpa],
    "Skill_Retention_Score": [retention]
})

# -------------------------------------
# Encode categorical columns
# -------------------------------------

for column, encoder in encoders.items():
    if column in sample.columns:
        sample[column] = encoder.transform(sample[column])

# -------------------------------------
# Scale
# -------------------------------------

sample = scaler.transform(sample)

# -------------------------------------
# Predict
# -------------------------------------

prediction = model.predict(sample)

burnout_encoder = encoders["Burnout_Risk_Level"]

result = burnout_encoder.inverse_transform(prediction)

print("\nPredicted Burnout Risk Level:", result[0])