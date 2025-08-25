# import os
# import gdown
# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # Import model URLs from separate file
# from model_links import models


# # Download models if not already present

# for model_filename, drive_url in models.items():
#     if not os.path.exists(model_filename):
#         st.info(f"Downloading {model_filename} from Google Drive...")
#         gdown.download(drive_url, model_filename, quiet=False)


# # Load models

# # for running locally

# # scaler_reg = joblib.load('scaler_reg.pkl')
# # classifier = joblib.load('classifier.pkl')
# # regression_model_rf0 = joblib.load('regression_model_rf0.pkl')
# # regression_model_rf1 = joblib.load('regression_model_rf1.pkl')
# # regression_model_rf2 = joblib.load('regression_model_rf2.pkl')

# # for deployment
# models = {
#     "scaler_reg.pkl": st.secrets["SCALER_URL"],
#     "classifier.pkl": st.secrets["CLASSIFIER_URL"],
#     "regression_model_rf0.pkl": st.secrets["RF0_URL"],
#     "regression_model_rf1.pkl": st.secrets["RF1_URL"],
#     "regression_model_rf2.pkl": st.secrets["RF2_URL"],
# }

# # Streamlit page config & header
# st.set_page_config(page_title="Earthquake Magnitude Predictor", page_icon="🌍", layout="centered")

# st.markdown(
#     """
#     <div style="text-align:center">
#         <h1>🌍 Earthquake Magnitude Predictor</h1>
#         <p style='color:gray;'>Enter location & event details to predict earthquake magnitude</p>
#     </div>
#     """, unsafe_allow_html=True
# )


# col1, col2 = st.columns(2)
# with col1:
#     significance = st.number_input("📊 Significance", min_value=0.0, value=0.0)
#     latitude = st.number_input("🧭 Latitude", value=0.0, format="%.4f")
# with col2:
#     longitude = st.number_input("🧭 Longitude", value=0.0, format="%.4f")

# if st.checkbox("Show location on map"):
#     st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))


# if st.button("🔮 Predict Magnitude"):
#     input_df = pd.DataFrame({
#         'significance': [significance],
#         'longitude': [longitude],
#         'latitude': [latitude]
#     })

#     input_features = scaler_reg.transform(input_df)
#     input_class = classifier.predict(input_features)[0]

#     # Predict based on class
#     if input_class == 0:
#         predicted_magnitude = regression_model_rf0.predict(input_features)
#     elif input_class == 1:
#         predicted_magnitude = regression_model_rf1.predict(input_features)
#     else:
#         predicted_magnitude = regression_model_rf2.predict(input_features)

#     st.success(f"Predicted Magnitude: {predicted_magnitude[0]:.2f}")

import os
import gdown
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Earthquake Magnitude Predictor", page_icon="🌍", layout="centered")

st.markdown("""
<div style="text-align:center">
    <h1>🌍 Earthquake Magnitude Predictor</h1>
    <p style='color:gray;'>Enter location & event details to predict earthquake magnitude</p>
</div>
""", unsafe_allow_html=True)

# Model URLs from secrets
models = {
    "scaler_reg.pkl": st.secrets["SCALER_URL"],
    "classifier.pkl": st.secrets["CLASSIFIER_URL"],
    "regression_model_rf0.pkl": st.secrets["RF0_URL"],
    "regression_model_rf1.pkl": st.secrets["RF1_URL"],
    "regression_model_rf2.pkl": st.secrets["RF2_URL"],
}

# Download models if missing
for filename, url in models.items():
    if not os.path.exists(filename):
        st.info(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)

# Load models
scaler_reg = joblib.load("scaler_reg.pkl")
classifier = joblib.load("classifier.pkl")
regression_model_rf0 = joblib.load("regression_model_rf0.pkl")
regression_model_rf1 = joblib.load("regression_model_rf1.pkl")
regression_model_rf2 = joblib.load("regression_model_rf2.pkl")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    significance = st.number_input("📊 Significance", min_value=0.0, value=0.0)
    latitude = st.number_input("🧭 Latitude", value=0.0, format="%.4f")
with col2:
    longitude = st.number_input("🧭 Longitude", value=0.0, format="%.4f")

if st.checkbox("Show location on map"):
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))

# Prediction
if st.button("🔮 Predict Magnitude"):
    input_df = pd.DataFrame({
        'significance': [significance],
        'longitude': [longitude],
        'latitude': [latitude]
    })
    input_features = scaler_reg.transform(input_df)
    input_class = classifier.predict(input_features)[0]

    if input_class == 0:
        predicted_magnitude = regression_model_rf0.predict(input_features)
    elif input_class == 1:
        predicted_magnitude = regression_model_rf1.predict(input_features)
    else:
        predicted_magnitude = regression_model_rf2.predict(input_features)

    st.success(f"Predicted Magnitude: {predicted_magnitude[0]:.2f}")
