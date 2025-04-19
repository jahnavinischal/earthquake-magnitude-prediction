# import os
# import gdown

# # Model download setup
# models = {
#     "classifier.pkl": "https://drive.google.com/uc?id=1iPMxGEkuZMMQQVrakL1VcZxNQqh9s8lg",
#     "regression_model_rf0.pkl": "https://drive.google.com/uc?id=1IweAur1Plbzcy1qbnJXp91JUndx4Ahs8",
#     "regression_model_rf1.pkl": "https://drive.google.com/uc?id=18jvpleKHqScNpBlypap2ziK428RQRumK"
# }

# # Download models if not already present
# for model_filename, drive_url in models.items():
#     if not os.path.exists(model_filename):
#         print(f"Downloading {model_filename}...")
#         gdown.download(drive_url, model_filename, quiet=False)
#     else:
#         print(f"{model_filename} already exists.")



import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
scaler_reg = joblib.load('scaler_reg.pkl')
classifier = joblib.load('classifier.pkl')
regression_model_rf0 = joblib.load('regression_model_rf0.pkl')
regression_model_rf1 = joblib.load('regression_model_rf1.pkl')
regression_model_rf2 = joblib.load('regression_model_rf2.pkl')

# Page config
st.set_page_config(page_title="Earthquake Magnitude Predictor", page_icon="🌍", layout="centered")

# Header
st.markdown(
    """
    <div style="text-align:center">
        <h1>🌍 Earthquake Magnitude Predictor</h1>
        <p style='color:gray;'>Enter location & event details to predict earthquake magnitude</p>
    </div>
    """, unsafe_allow_html=True
)

# Input UI in columns
col1, col2 = st.columns(2)
with col1:
    significance = st.number_input("📊 Significance", min_value=0.0, value=0.0)
    latitude = st.number_input("🧭 Latitude", value=0.0, format="%.4f")

with col2:
    longitude = st.number_input("🧭 Longitude", value=0.0, format="%.4f")

# Optional map preview
if st.checkbox("Show location on map"):
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))

# Predict button
if st.button("🔮 Predict Magnitude"):
    input_df = pd.DataFrame({
        'significance': [significance],
        'longitude': [longitude],
        'latitude': [latitude]
    })

    input_features = scaler_reg.transform(input_df)
    input_class = classifier.predict(input_features)[0]

    # Predict based on class
    if input_class == 0:
        predicted_magnitude = regression_model_rf0.predict(input_features)
    elif input_class == 1:
        predicted_magnitude = regression_model_rf1.predict(input_features)
    else:
        predicted_magnitude = regression_model_rf2.predict(input_features)

    # st.success(f"📌 Magnitude Class: `{input_class}`")
    # st.metric("Predicted Magnitude", f"{predicted_magnitude[0]:.2f}")
    st.success(f"Predicted Magnitude: {predicted_magnitude[0]:.2f}")

    

