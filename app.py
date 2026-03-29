import streamlit as st
from predict import predict_processing_time

st.title("Visa Processing Time Estimator")

st.write("Enter application details:")

# Inputs
country = st.selectbox("Country", ["India", "USA", "UK"])
visa_type = st.selectbox("Visa Type", ["Student", "Tourist", "Work"])
processing_office = st.selectbox("Processing Office", ["Delhi", "New York", "London"])
application_date = st.date_input("Application Date")

# Button
if st.button("Predict"):

    input_data = {
        "country": country,
        "visa_type": visa_type,
        "processing_office": processing_office,
        "application_date": str(application_date)
    }

    result = predict_processing_time(input_data)

    st.success(f"Estimated Processing Time: {int(result)} days")