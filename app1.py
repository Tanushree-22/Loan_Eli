# app.py

import streamlit as st
import pandas as pd
import joblib

# Load saved model pipeline
model = joblib.load('loan_model_pipeline.pkl')

st.title("🏦 Loan Eligibility Predictor")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=480)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Predict button
if st.button("Predict Loan Eligibility"):
    total_income = applicant_income + coapplicant_income
    emi = loan_amount * 0.08 * (1 + 0.08)**loan_term / ((1 + 0.08)**loan_term - 1)
    income_to_emi = total_income / emi if emi != 0 else 0

    input_dict = {
        'Gender': gender,
        'Married': married,
        'Dependents': int(dependents),
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'Total_Income': total_income,
        'EMI': emi,
        'Income_to_EMI': income_to_emi
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")
