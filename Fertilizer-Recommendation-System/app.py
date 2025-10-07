# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load model and encoders
# -------------------------
model = joblib.load("fertilizer_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Fertilizer Recommendation System", page_icon="🌱", layout="centered")

# -------------------------
# Header
# -------------------------
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color:#2E7D32;">🌾 Fertilizer Recommendation System 🌾</h1>
        <p style="font-size:18px;">Simple and easy-to-use tool for farmers to know fertilizer needs</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("📝 Enter Soil, Crop & Nutrients")

soil_type = st.sidebar.selectbox("🌍 Select Soil Type:", soil_encoder.classes_)
crop_type = st.sidebar.selectbox("🌿 Select Crop Type:", crop_encoder.classes_)

N = st.sidebar.number_input("Nitrogen (N)", 0, 200, 50, help="Helps in leaf growth")
P = st.sidebar.number_input("Phosphorus (P)", 0, 200, 50, help="Important for root development")
K = st.sidebar.number_input("Potassium (K)", 0, 200, 50, help="Improves plant immunity and fruit quality")

# Optional extra inputs (future-ready)
# temperature = st.sidebar.number_input("Temperature (°C)", 0, 50, 25)
# humidity = st.sidebar.number_input("Humidity (%)", 0, 100, 60)

# -------------------------
# Helper for nutrient status
# -------------------------
def nutrient_status(value):
    if value < 30:
        return "❌ LOW", "red"
    elif value <= 70:
        return "✅ OPTIMAL", "green"
    else:
        return "⚠️ HIGH", "orange"

# -------------------------
# Prediction Button
# -------------------------
if st.sidebar.button("🔍 Get Recommendation"):
    # Encode categorical inputs
    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]

    # Arrange features
    features = np.array([[soil_encoded, crop_encoded, N, P, K]])

    # Prediction
    prediction = model.predict(features)[0]

    # Decode fertilizer
    fertilizer = fertilizer_encoder.inverse_transform([prediction])[0]

    # -------------------------
    # Display Results
    # -------------------------
    st.success(f"✅ Recommended Fertilizer: **{fertilizer}**")
    st.balloons()

    # 📋 Input summary
    st.subheader("📋 Your Input Summary")
    st.table(pd.DataFrame([[soil_type, crop_type, N, P, K]],
             columns=["Soil Type", "Crop Type", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]))

    # 📊 Nutrient Levels Chart
    st.subheader("📊 Soil Nutrient Distribution")
    nutrients = {"Nitrogen (N)": N, "Phosphorus (P)": P, "Potassium (K)": K}
    fig, ax = plt.subplots()
    ax.bar(nutrients.keys(), nutrients.values(), color=["#4CAF50", "#FF9800", "#2196F3"])
    ax.set_ylabel("Value")
    ax.set_title("N-P-K Nutrient Levels")
    st.pyplot(fig)

    # 🌱 Farmer-friendly Nutrient Status
    st.subheader("⚡ Nutrient Health Indicators")
    st.write("👉 Shows if nutrient level is LOW, OPTIMAL, or HIGH")

    for nutrient, value in nutrients.items():
        status, color = nutrient_status(value)
        st.markdown(f"**{nutrient}: {value} → <span style='color:{color}; font-weight:bold'>{status}</span>**", unsafe_allow_html=True)
        st.progress(min(value, 100) / 100)

    # 💡 Fertilizer usage tip
    st.info("💡 Tip: Apply the recommended fertilizer in correct quantity and at the right time for best results.")

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <hr>
    <div style="text-align: center; color: grey;">
        Developed by <b>Aditya Dilip Kadam</b> | Made with ❤️ using <b>Streamlit</b> | Easy for Farmers 🌾
    </div>
    """,
    unsafe_allow_html=True
)
