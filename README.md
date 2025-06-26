# 🌍 Earthquake Magnitude Prediction Web App

A machine learning-powered web application that predicts the likelihood of earthquakes based on a 30-year historical dataset. Built using **Streamlit**, this app leverages data-driven models to offer quick predictions and insights.

![image](https://github.com/user-attachments/assets/e39f33d8-c7dc-49ac-9c57-a59613623fba)

*Screenshot of the Streamlit Web App Interface*

**You may access the deployed app through the link given below:**
https://earthquake-magnitude-predictor1.streamlit.app/

## 📊 Dataset

The dataset used spans **30 years** of global earthquake data, sourced from [Kaggle](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023)

## 🧠 Technologies Used

- **Python**
- **Pandas, NumPy, Scikit-learn** for data processing and modeling
- **Streamlit** for the web interface
- **Pickle (.pkl)** files for model storage

## 🚀 Features

- Predict earthquake likelihood based on input parameters
- Load and use pre-trained models from `.pkl` files
- User-friendly interface powered by Streamlit
- Based on long-term historical seismic activity data

## 🔧 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jahnavinischal/earthquake-prediction.git
   cd earthquake-prediction

2. Install the required packages:

   ```bash
     pip install -r requirements.txt

4. Run the Streamlit web app:

   ```bash
   streamlit run index.py
   
If .pkl model files are not already present, they will be generated automatically on the first run.

## 🧪 Usage
- Launch the app using the command above.
- Input the necessary parameters via the interface.
- View the predicted earthquake outcome and other related insights.


## 📎 Notes
- Make sure the dataset is placed in the correct directory if required.
- The model might take a few moments to generate if .pkl files don’t exist.
