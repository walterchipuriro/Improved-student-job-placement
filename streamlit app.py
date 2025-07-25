import streamlit as st
import pickle
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('random_forest_model.pkl')

# Load encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("Student Placement Predictor")
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
degree_program = st.selectbox(
    "Degree Program", 
    options=label_encoders["degree_program"].classes_, 
    key="degree_program_key"
)

attendance_rate = st.selectbox(
    "Attendance Rate", 
    options=label_encoders["attendance_rate"].classes_, 
    key="attendance_rate_key"
)

household_income_bracket = st.selectbox(
    "Household Income", 
    options=label_encoders["household_income_bracket"].classes_, 
    key="household_income_bracket_key"
)

urban_or_rural_background = st.selectbox(
    "Background", 
    options=label_encoders["urban_or_rural_background"].classes_, 
    key="urban_or_rural_background_key"
)

previous_employment = st.selectbox(
    "Previous Employment", 
    options=label_encoders["previous_employment"].classes_, 
    key="previous_employment_key"
)

# Text Inputs
# extracurricular_activities_description = st.text_area("Extracurricular Activities")
# leadership_experience_text = st.text_area("Leadership Experience")
# volunteering_experience_text = st.text_area("Volunteering Experience")
# career_objective_text = st.text_area("Career Objective")

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
    "previous_employment": previous_employment,

    # Comment out the text features if not encoded:
    # "extracurricular_activities_description": extracurricular_activities_description,
    # "leadership_experience_text": leadership_experience_text,
    # "volunteering_experience_text": volunteering_experience_text,
    # "career_objective_text": career_objective_text,
}


# Encode categorical variables
for col in label_encoders:
    if col in input_dict:
        input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Make prediction
if st.button("Predict Placement"):
    if any(value == '' for key, value in input_dict.items()):
        st.error("Please fill in all required fields.")
    else:
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        st.success("Prediction: Placed ✅" if prediction == 1 else "Predicted: Not Placed ❌")

