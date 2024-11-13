# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import joblib
import pandas as pd  # Import pandas for DataFrame conversion

# Load the trained model
model = joblib.load("crop_recommendation.pkl")

# Load the Label Encoder for the Recommendations
label_encoder = joblib.load("label_encoder.pkl")

# Title and description
st.title("Crop Recommendation System")
st.write("Enter the soil and climate parameters to get a crop recommendation.")

# User input for the features
nitrogen_ratio = st.number_input("Nitrogen Ratio (N)", min_value=0.0, max_value=100.0, step=0.1)
phosphorous_ratio = st.number_input("Phosphorous Ratio (P)", min_value=0.0, max_value=100.0, step=0.1)
potassium_ratio = st.number_input("Potassium Ratio (K)", min_value=0.0, max_value=100.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

# Make prediction button
if st.button("Recommend Crop"):
    # Prepare the input data as a DataFrame for prediction
    input_data = pd.DataFrame([[nitrogen_ratio, phosphorous_ratio, potassium_ratio, temperature, humidity, ph, rainfall]],
                              columns=['Nitrogen_Ratio', 'Phosphorous_Ratio', 'Potassium_Ratio', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    # Make a prediction
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    
    # Display the recommended crop
    st.write(f"The recommended crop is: **{prediction[0]}**")
