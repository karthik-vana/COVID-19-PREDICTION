
# COVID-19 Prediction Web App
# Save this as: app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('covid_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('🦠 COVID-19 Cases Predictor')
st.write('Predict future COVID-19 confirmed cases using Machine Learning')

# Sidebar inputs
st.sidebar.header('Input Features')
day_number = st.sidebar.number_input('Day Number', min_value=0, value=100)
ma_7 = st.sidebar.number_input('7-Day Moving Average', min_value=0, value=50000)
confirmed_lag1 = st.sidebar.number_input('Yesterday Cases', min_value=0, value=48000)
growth_rate = st.sidebar.slider('Growth Rate', -0.5, 0.5, 0.01)

# Predict button
if st.sidebar.button('🔮 Predict'):
    # Prepare input
    features = np.array([[day_number, ma_7, confirmed_lag1, growth_rate]])
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Display result
    st.success(f'Predicted Cases: {prediction:,.0f}')
    
    # Visualization
    fig, ax = plt.subplots()
    ax.bar(['Predicted Cases'], [prediction], color='steelblue')
    ax.set_ylabel('Cases')
    st.pyplot(fig)

st.sidebar.info('Adjust the parameters and click Predict!')
