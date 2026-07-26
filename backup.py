import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Burnout Risk Predictor",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------

model = joblib.load("models/burnout_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")
scaler = joblib.load("models/scaler.pkl")

df = pd.read_csv("data/cleaned_impact_ai.csv")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📘 Project Information")

st.sidebar.markdown("""
### AI Burnout Risk Predictor

This application predicts the burnout risk of students based on:

- Academic Performance
- AI Usage
- Study Habits
- Prompt Engineering Skills
- Institutional Policies

Dataset: Kaggle
Model: Logistic Regression
""")

st.sidebar.divider()

st.sidebar.subheader("Dataset")

st.sidebar.write(f"Rows : {df.shape[0]}")
st.sidebar.write(f"Columns : {df.shape[1]}")
st.sidebar.write("Target : Burnout_Risk_Level")

st.sidebar.divider()

st.sidebar.subheader("Model")

st.sidebar.success("Logistic Regression")

st.sidebar.metric(
    label="Accuracy",
    value="52.21%"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🤖 AI Burnout Risk Prediction System")

st.write(
    "Predict the burnout risk level of students based on AI usage and academic information."
)

st.divider()

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

left, right = st.columns(2)

with left:

    major = st.selectbox(
        "Major Category",
        sorted(df["Major_Category"].unique())
    )

    year = st.selectbox(
        "Year of Study",
        sorted(df["Year_of_Study"].unique())
    )

    pre_gpa = st.slider(
        "Pre Semester GPA",
        float(df["Pre_Semester_GPA"].min()),
        float(df["Pre_Semester_GPA"].max()),
        float(df["Pre_Semester_GPA"].mean())
    )

    weekly_ai = st.slider(
        "Weekly GenAI Hours",
        int(df["Weekly_GenAI_Hours"].min()),
        int(df["Weekly_GenAI_Hours"].max()),
        int(df["Weekly_GenAI_Hours"].mean())
    )

    primary_use = st.selectbox(
        "Primary Use Case",
        sorted(df["Primary_Use_Case"].unique())
    )

    prompt_skill = st.selectbox(
        "Prompt Engineering Skill",
        sorted(df["Prompt_Engineering_Skill"].unique())
    )

with right:

    tool_diversity = st.slider(
        "Tool Diversity",
        int(df["Tool_Diversity"].min()),
        int(df["Tool_Diversity"].max()),
        int(df["Tool_Diversity"].mean())
    )

    paid = st.selectbox(
        "Paid Subscription",
        [True, False]
    )

    study_hours = st.slider(
        "Traditional Study Hours",
        float(df["Traditional_Study_Hours"].min()),
        float(df["Traditional_Study_Hours"].max()),
        float(df["Traditional_Study_Hours"].mean())
    )

    dependency = st.slider(
        "Perceived AI Dependency",
        int(df["Perceived_AI_Dependency"].min()),
        int(df["Perceived_AI_Dependency"].max()),
        int(df["Perceived_AI_Dependency"].mean())
    )

    policy = st.selectbox(
        "Institutional Policy",
        sorted(df["Institutional_Policy"].unique())
    )

    anxiety = st.slider(
        "Anxiety During Exams",
        int(df["Anxiety_Level_During_Exams"].min()),
        int(df["Anxiety_Level_During_Exams"].max()),
        int(df["Anxiety_Level_During_Exams"].mean())
    )

    post_gpa = st.slider(
        "Post Semester GPA",
        float(df["Post_Semester_GPA"].min()),
        float(df["Post_Semester_GPA"].max()),
        float(df["Post_Semester_GPA"].mean())
    )

    retention = st.slider(
        "Skill Retention Score",
        float(df["Skill_Retention_Score"].min()),
        float(df["Skill_Retention_Score"].max()),
        float(df["Skill_Retention_Score"].mean())
    )

st.divider()

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("Predict Burnout Risk", use_container_width=True):

    sample = pd.DataFrame({

        "Major_Category":[major],
        "Year_of_Study":[year],
        "Pre_Semester_GPA":[pre_gpa],
        "Weekly_GenAI_Hours":[weekly_ai],
        "Primary_Use_Case":[primary_use],
        "Prompt_Engineering_Skill":[prompt_skill],
        "Tool_Diversity":[tool_diversity],
        "Paid_Subscription":[paid],
        "Traditional_Study_Hours":[study_hours],
        "Perceived_AI_Dependency":[dependency],
        "Institutional_Policy":[policy],
        "Anxiety_Level_During_Exams":[anxiety],
        "Post_Semester_GPA":[post_gpa],
        "Skill_Retention_Score":[retention]

    })

    for column, encoder in encoders.items():

        if column in sample.columns:

            sample[column] = encoder.transform(sample[column])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    result = encoders["Burnout_Risk_Level"].inverse_transform(prediction)[0]

    st.header("Prediction Result")

    if result == "High":

        st.error("🔴 HIGH Burnout Risk")

    elif result == "Medium":

        st.warning("🟡 MEDIUM Burnout Risk")

    else:

        st.success("🟢 LOW Burnout Risk")

    st.info(
        "This prediction is generated using a trained Machine Learning model and is intended for educational purposes."
    )

# ---------------------------------------------------
# DATASET INSIGHTS
# ---------------------------------------------------

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Insights",
    "📈 Model Evaluation",
    "ℹ About"
])

with tab1:

    with st.expander("Correlation Heatmap"):

        st.image(
            "images/correlation_heatmap.png",
            use_container_width=True
        )

    with st.expander("Burnout Distribution"):

        st.image(
            "images/burnout_distribution.png",
            use_container_width=True
        )

    with st.expander("Histograms"):

        st.image(
            "images/histograms.png",
            use_container_width=True
        )

with tab2:

    st.image(
        "images/confusion_matrix.png",
        use_container_width=True
    )

with tab3:

    st.markdown("""
# About this Project

This project predicts student burnout risk using Machine Learning.

### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

### Machine Learning Workflow

- Data Cleaning
- Exploratory Data Analysis
- Feature Encoding
- Feature Scaling
- Model Training
- Model Evaluation
- Prediction

Developed as part of a Machine Learning Workshop.
""")