import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Load saved model
model = joblib.load("https://raw.githubusercontent.com/1sleep231/edu-app/refs/heads/main/data.csv")  # Sesuaikan dengan lokasi model Anda

# Load data
data = pd.read_csv(r"C:\Career\Dicoding\submission\data.csv", sep=';')

# Feature importance data for predictors
feature_importance = pd.DataFrame({
    "Feature": [
        "Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade", "Admission_grade",
        "Curricular_units_1st_sem_approved", "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_evaluations", "Curricular_units_1st_sem_evaluations",
        "Curricular_units_2nd_sem_enrolled", "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_credited"
    ],
    "Importance": [
        0.217599, 0.190263, 0.144747, 0.105375, 0.103560,
        0.075156, 0.060971, 0.040403, 0.033991, 0.015196
    ]
})

# Layout of the Streamlit app
st.title("Dropout Prediction Dashboard")

# Feature Importance Graph
st.header("Feature Importance")
fig = px.bar(feature_importance, y="Feature", x="Importance", orientation="h",
             title="Top Predictors for Dropout Risk").update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig)

# Distribution of Student Status
st.header("Distribution of Student Status")
fig2 = px.histogram(data, x="Status", title="Student Status Distribution").update_xaxes(categoryorder="total descending")
st.plotly_chart(fig2)

# Prediction Section
st.header("Predict Dropout Risk")

# Input fields for user to input predictor values
with st.form(key='predict_form'):
    approved_2nd_sem = st.number_input("Curricular Units 2nd Semester Approved", min_value=0, value=0)
    grade_2nd_sem = st.number_input("Curricular Units 2nd Semester Grade", min_value=0.0, max_value=20.0, step=0.1, value=0.0)
    admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=20.0, step=0.1, value=0.0)
    approved_1st_sem = st.number_input("Curricular Units 1st Semester Approved", min_value=0, value=0)
    grade_1st_sem = st.number_input("Curricular Units 1st Semester Grade", min_value=0.0, max_value=20.0, step=0.1, value=0.0)
    evaluations_2nd_sem = st.number_input("Curricular Units 2nd Semester Evaluations", min_value=0, value=0)
    evaluations_1st_sem = st.number_input("Curricular Units 1st Semester Evaluations", min_value=0, value=0)
    enrolled_2nd_sem = st.number_input("Curricular Units 2nd Semester Enrolled", min_value=0, value=0)
    enrolled_1st_sem = st.number_input("Curricular Units 1st Semester Enrolled", min_value=0, value=0)
    credited_1st_sem = st.number_input("Curricular Units 1st Semester Credited", min_value=0, value=0)

    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Prediction logic after button is clicked
if submit_button:
    # Create input array
    input_data = np.array([[approved_2nd_sem, grade_2nd_sem, admission_grade, approved_1st_sem,
                            grade_1st_sem, evaluations_2nd_sem, evaluations_1st_sem,
                            enrolled_2nd_sem, enrolled_1st_sem, credited_1st_sem]])

    # Make prediction
    prediction = model.predict(input_data)

    # Interpret prediction
    status_map = {0: "Graduate", 1: "Dropout", 2: "Enrolled"}
    predicted_status = status_map[prediction[0]]

    # Display result
    st.write(f"Predicted Status: {predicted_status}")
