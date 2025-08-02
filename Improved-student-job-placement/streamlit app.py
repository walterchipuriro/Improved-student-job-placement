import streamlit as st
import pickle
import pandas as pd
import joblib
import shap
import numpy as np

# Load model and encoders
model = joblib.load('random_forest_model.pkl')

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("🎓 Student Placement Predictor")
st.markdown("Enter student information below:")

# Numerical Inputs
gpa = st.number_input("GPA", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("IQ", min_value=70, max_value=200)
math_score = st.number_input("Math Score", min_value=0.0, max_value=100.0)
english_score = st.number_input("English Score", min_value=0.0, max_value=100.0)
backlogs_count = st.number_input("Number of Backlogs", min_value=0)
certification_count = st.number_input("Certifications Completed", min_value=0)
internship_count = st.number_input("Internships Completed", min_value=0)
project_count = st.number_input("Projects Completed", min_value=0)
hours_studied_per_week = st.number_input("Hours Studied Per Week", min_value=0.0)
technical_skills_score = st.number_input("Technical Skills Score", min_value=0.0, max_value=100.0)
soft_skills_score = st.number_input("Soft Skills Score", min_value=0.0, max_value=100.0)
number_of_job_applications = st.number_input("Number of Job Applications", min_value=0)

# Categorical Inputs
degree_program = st.selectbox("Degree Program", options=label_encoders["degree_program"].classes_)
attendance_rate = st.selectbox("Attendance Rate", options=label_encoders["attendance_rate"].classes_)
household_income_bracket = st.selectbox("Household Income", options=label_encoders["household_income_bracket"].classes_)
urban_or_rural_background = st.selectbox("Background", options=label_encoders["urban_or_rural_background"].classes_)
previous_employment = st.selectbox("Previous Employment", options=label_encoders["previous_employment"].classes_)

# Prepare input dictionary
input_dict = {
    "gpa": gpa,
    "iq": iq,
    "degree_program": degree_program,
    "math_score": math_score,
    "english_score": english_score,
    "attendance_rate": attendance_rate,
    "backlogs_count": backlogs_count,
    "certification_count": certification_count,
    "internship_count": internship_count,
    "project_count": project_count,
    "hours_studied_per_week": hours_studied_per_week,
    "technical_skills_score": technical_skills_score,
    "soft_skills_score": soft_skills_score,
    "number_of_job_applications": number_of_job_applications,
    "household_income_bracket": household_income_bracket,
    "urban_or_rural_background": urban_or_rural_background,
    "previous_employment": previous_employment
}

# Encode categorical variables
for col in label_encoders:
    if col in input_dict:
        input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

# Prediction and SHAP explanation
if st.button("Predict Placement"):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    # Display prediction
    st.success("🎯 Prediction: **Placed ✅**" if prediction == 1 else "❌ Prediction: **Not Placed**")

    # SHAP explanation
    st.subheader("📊 Why This Prediction?")
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    # Check and flatten SHAP values
    raw_contributions = shap_values.values[0]

    if raw_contributions.ndim == 2 and raw_contributions.shape[1] > 1:
        contributions = raw_contributions[:, 1]  # Class 1 (Placed)
    elif raw_contributions.ndim == 2:
        contributions = raw_contributions.flatten()
    else:
        contributions = raw_contributions

    # Verify lengths match
    if len(contributions) != len(input_df.columns):
        st.error("Mismatch between SHAP values and feature columns. Check model output.")
    else:
        impact = pd.DataFrame({
            "feature": input_df.columns,
            "value": input_df.iloc[0].values,
            "contribution": contributions
        }).sort_values(by="contribution", key=lambda x: abs(x), ascending=False)

        # Display top 3 contributors
        for i, row in impact.head(3).iterrows():
            symbol = "↑" if row["contribution"] > 0 else "↓"
            st.markdown(f"- **{row['feature']}**: {row['value']} → Impact: `{row['contribution']:.2f}` {symbol}")

    # Recommendations
    st.subheader("💡 Recommendations to Improve Placement Chances")
    recommendations = []

    if input_dict["gpa"] < 6:
        recommendations.append("📘 Improve GPA through focused studying or tutoring.")
    if input_dict["soft_skills_score"] < 50:
        recommendations.append("🗣️ Enhance your soft skills by practicing interviews or presentations.")
    if input_dict["internship_count"] < 1:
        recommendations.append("💼 Gain internship experience to boost your employability.")
    if input_dict["project_count"] < 2:
        recommendations.append("🔧 Complete more practical projects to showcase your skills.")
    if input_dict["certification_count"] < 1:
        recommendations.append("📚 Consider earning relevant certifications to strengthen your resume.")

    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("✅ Excellent profile — keep applying confidently!")

    # Optional: SHAP debugging info
    # st.write("SHAP values shape:", raw_contributions.shape)
