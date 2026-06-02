import streamlit as st
import pandas as pd
import pickle
import gdown
import os

MODEL_URL = "https://drive.google.com/file/d/1F__J9MGuDafywkrQ8UtVYAEaITF1vHSg/view?usp=sharing"
MODEL_PATH = "best_bank_marketing_model.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

st.title("Bank Marketing Prediction App")

st.write("Aplikasi ini memprediksi apakah nasabah akan subscribe deposito.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.selectbox("Marital", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Default", ["no", "yes"])
balance = st.number_input("Balance", value=1000)
housing = st.selectbox("Housing", ["no", "yes"])
loan = st.selectbox("Loan", ["no", "yes"])
contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
day = st.number_input("Day", min_value=1, max_value=31, value=15)
month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
campaign = st.number_input("Campaign", min_value=1, value=1)
pdays = st.number_input("Pdays", value=-1)
previous = st.number_input("Previous", min_value=0, value=0)
poutcome = st.selectbox("Poutcome", ["failure", "other", "success", "unknown"])

data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "day": [day],
    "month": [month],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome]
})

st.write("Data input:")
st.dataframe(data)

if st.button("Predict"):
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.success("Nasabah diprediksi subscribe deposito")
    else:
        st.error("Nasabah diprediksi tidak subscribe deposito")

    st.write(f"Probabilitas subscribe: {probability:.2%}")