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

# scaler_reg = joblib.load('scaler_reg.pkl')
# classifier = joblib.load('classifier.pkl')
# regression_model_rf0 = joblib.load('regression_model_rf0.pkl')
# regression_model_rf1 = joblib.load('regression_model_rf1.pkl')
# regression_model_rf2 = joblib.load('regression_model_rf2.pkl')

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

# ------------------- Streamlit page config -------------------
st.set_page_config(
    page_title="Earthquake Magnitude Predictor",
    page_icon="🌍",
    layout="centered"
)

st.markdown("""
<div style="text-align:center">
    <h1>🌍 Earthquake Magnitude Predictor</h1>
    <p style='color:gray;'>Enter location & event details to predict earthquake magnitude</p>
</div>
""", unsafe_allow_html=True)

# ------------------- Model URLs -------------------
try:
    # Streamlit Cloud secrets
    models = {
        "scaler_reg.pkl": st.secrets["SCALER_URL"],
        "classifier.pkl": st.secrets["CLASSIFIER_URL"],
        "regression_model_rf0.pkl": st.secrets["RF0_URL"],
        "regression_model_rf1.pkl": st.secrets["RF1_URL"],
        "regression_model_rf2.pkl": st.secrets["RF2_URL"],
    }
except:
    # Local fallback URLs
    models = {
        "scaler_reg.pkl": "https://drive.google.com/uc?id=1goSGXQgIQJwS6BJ_c_P5LZLVUTks5r0K",
        "classifier.pkl": "https://drive.google.com/uc?id=10FTd9arh_eR5hR1WlH7YLf9wowlPJ9bq",
        "regression_model_rf0.pkl": "https://drive.google.com/uc?id=1966nL0gd0ipQtkp3cge07Dv2l7a20ey",
        "regression_model_rf1.pkl": "https://drive.google.com/uc?id=1TSsANKbPhwD-bUaBRALOmF1eDzuwd0FO",
        "regression_model_rf2.pkl": "https://drive.google.com/uc?id=1Wmr58BXVBW-IMhRQoPCCeVVmHoadpipT",
    }

# ------------------- Function to download a file if missing -------------------
def download_file(filename, url):
    if not os.path.exists(filename):
        st.info(f"Downloading {filename} from Google Drive...")
        gdown.download(url, filename, quiet=False, fuzzy=True)
        if not os.path.exists(filename) or os.path.getsize(filename) < 1000:
            st.error(f"Failed to download {filename}. Check Google Drive link or permissions.")
            st.stop()

# ------------------- Cached model loader -------------------
@st.cache_resource
def load_models():
    # Download all models first
    for filename, url in models.items():
        download_file(filename, url)

    # Load models into memory
    scaler = joblib.load("scaler_reg.pkl")
    clf = joblib.load("classifier.pkl")
    rf0 = joblib.load("regression_model_rf0.pkl")
    rf1 = joblib.load("regression_model_rf1.pkl")
    rf2 = joblib.load("regression_model_rf2.pkl")
    return scaler, clf, rf0, rf1, rf2

# Load models (cached)
scaler_reg, classifier, regression_model_rf0, regression_model_rf1, regression_model_rf2 = load_models()
# st.success("Models loaded successfully!")

# ------------------- User Inputs -------------------
col1, col2 = st.columns(2)
with col1:
    significance = st.number_input("📊 Significance", min_value=0.0, value=0.0)
    latitude = st.number_input("🧭 Latitude", value=0.0, format="%.4f")
with col2:
    longitude = st.number_input("🧭 Longitude", value=0.0, format="%.4f")

if st.checkbox("Show location on map"):
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))

# ------------------- Prediction -------------------
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

    st.success(f"Predicted Magnitude: {predicted_magnitude[0]:.2f}")
