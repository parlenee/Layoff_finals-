import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_model_compressed.pkl")

st.set_page_config(page_title="Prediksi Layoff Startup", page_icon="💼", layout="centered")

st.title("💼 Prediksi Skala Layoff Startup")
st.markdown("Masukkan data perusahaan di bawah ini untuk memprediksi tingkat **keparahan PHK (Small, Medium, Large)**.")

# --- Input form ---
industry = st.selectbox("Industry", [
    "Technology", "Finance", "Healthcare", "Consumer", "HR", "Logistics", "Unknown"
])

stage = st.selectbox("Stage", [
    "Seed", "Series A", "Series B", "Series C", "Post-IPO", "Acquired", "Unknown"
])

country = st.text_input("Country", "United States")
location = st.text_input("Location", "SF Bay Area")
source = st.text_input("Source", "Internal Memo")
funds_raised = st.number_input("Funds Raised (USD)", min_value=0.0, step=10.0)
year = st.number_input("Year", min_value=2020, max_value=2024, step=1)

if st.button("Prediksi Skala Layoff"):
    input_data = pd.DataFrame({
        "industry": [industry],
        "country": [country],
        "stage": [stage],
        "location": [location],
        "source": [source],
        "funds_raised": [funds_raised],
        "year": [year]
    })

    input_data["year"] = input_data["year"].astype(int)
    prediction = model.predict(input_data)[0]

    mapping = {0: "Small Layoff", 1: "Medium Layoff", 2: "Large Layoff"}
    result_text = mapping.get(prediction, "Unknown")

    color_map = {
        "Small Layoff": "#4CAF50",  
        "Medium Layoff": "#FFC107",
        "Large Layoff": "#F44336"
    }

    color = color_map.get(result_text, "#2196F3")

    st.markdown(
        f"""
        <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center">
            <h3 style="color:white;">Hasil Prediksi: {result_text}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    if result_text == "Small Layoff":
        st.info("Risiko PHK relatif kecil. Perusahaan berada pada kondisi stabil.")
    elif result_text == "Medium Layoff":
        st.warning("Risiko PHK sedang. Perlu perhatian pada faktor keuangan dan pendanaan.")
    else:
        st.error("Risiko PHK tinggi! Perusahaan kemungkinan melakukan PHK besar-besaran.")
