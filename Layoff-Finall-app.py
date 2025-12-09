import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/rf_model.pkl")

st.title("Prediksi Skala Layoff Perusahaan ")
st.markdown("Masukkan data perusahaan di bawah ini untuk memprediksi apakah PHK berskala kecil, sedang, atau besar.")

# Input user
industry = st.selectbox("Industry", [
    "Technology", "Finance", "Healthcare", "Consumer", "HR", "Logistics", "Unknown"
])

stage = st.selectbox("Stage", [
    "Seed", "Series A", "Series B", "Series C", "Post-IPO", "Acquired", "Unknown"
])

country = st.text_input("Country", "United States")
location = st.text_input("Location", "SF Bay Area")
source = st.text_input("Source", "Internal Memo")

funds_raised = st.number_input("Funds Raised (juta USD)", min_value=0.0, step=10.0)
total_laid_off = st.number_input("Total Karyawan di-PHK", min_value=0, step=1)

# Tombol Prediksi
if st.button("Prediksi Skala Layoff"):
    input_data = pd.DataFrame({
        "industry": [industry],
        "stage": [stage],
        "country": [country],
        "location": [location],
        "source": [source],
        "funds_raised": [funds_raised],
        "total_laid_off": [total_laid_off]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Hasil Prediksi: **{prediction}**")
