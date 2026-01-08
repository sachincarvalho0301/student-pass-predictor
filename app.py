import streamlit as st
import pandas as pd
import joblib

st.title('Student Pass Prediction')

pipeline = joblib.load('student_pipeline.pkl')

study_hours = st.number_input('Study hours per day', min_value = 0)
attendance = st.number_input('Attendance %', min_value = 0, max_value = 100)
prev_score = st.number_input('Previous Exam Score', min_value = 0, max_value = 100)

if st.button("Predict Result"):
    input_df = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "prev_score": [prev_score]
    })

    prediction = pipeline.predict(input_df)

    if prediction[0] == 1:
        st.success('Pass')
    else:
        st.error('Fail')