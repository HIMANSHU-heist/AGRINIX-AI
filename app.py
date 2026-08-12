import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="AGRINEX AI - Crop Recommender", page_icon="🌾", layout="centered")

# Load model + labels
model = joblib.load('crop_model.pkl')
with open('crop_labels.json') as f:
    crop_labels = json.load(f)

# ---- HEADER ----
st.markdown("<h1 style='text-align:center; color:green;'>🌾 AGRINEX AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Smart Crop Recommendation System</h4>", unsafe_allow_html=True)
st.write("---")

st.write("Tumcha mati (soil) ani weather cha data takal ki AI tumhala best crop suggest karel.")

# ---- INPUT FORM ----
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=205, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

st.write("")

# ---- PREDICT BUTTON ----
if st.button("🔍 Recommend Crop", use_container_width=True):
    sample = pd.DataFrame([{
        'N': N, 'P': P, 'K': K,
        'temperature': temperature, 'humidity': humidity,
        'ph': ph, 'rainfall': rainfall
    }])

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    confidence = max(probabilities) * 100

    st.write("---")
    st.markdown(
        f"""
        <div style='background-color:#e8f5e9; padding:25px; border-radius:12px; text-align:center;'>
            <h2 style='color:#2e7d32;'>✅ Recommended Crop: {prediction.upper()}</h2>
            <h4 style='color:#555;'>Confidence: {confidence:.2f}%</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top 3 predictions
    st.write("")
    st.subheader("📊 Top 3 Possible Crops")
    top3_idx = np.argsort(probabilities)[-3:][::-1]
    for idx in top3_idx:
        st.write(f"**{model.classes_[idx]}** — {probabilities[idx]*100:.2f}%")

st.write("---")
st.caption("AGRINEX AI Prototype | Built with Machine Learning 🌱")
