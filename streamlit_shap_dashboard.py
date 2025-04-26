# streamlit_shap_dashboard.py

import streamlit as st
import pandas as pd
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load model and encoders (assume already saved)
model = joblib.load('flight_delay_model.pkl')
encoders = joblib.load('encoders.pkl')

# UI: User inputs
st.title('Flight Delay Prediction & Explanation')

month = st.selectbox('Month', list(range(1, 13)))
part_of_day = st.selectbox('Part of Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
carrier = st.selectbox('Carrier', ['Other', 'UA', 'AA', 'DL', 'B6', 'WN'])
origin = st.selectbox('Origin Airport', ['Other', 'JFK', 'EWR', 'LGA'])
dest = st.selectbox('Destination Airport', ['Other', 'ORD', 'ATL', 'LAX'])
distance = st.slider('Flight Distance (miles)', 100, 3000, 500)

# Prepare input
input_data = pd.DataFrame({
    'month': [month],
    'day_of_week': [day_of_week],
    'part_of_day': [part_of_day],
    'carrier_simplified': [carrier],
    'origin_simplified': [origin],
    'dest_simplified': [dest],
    'distance': [distance]
})

# Encode categorical fields
for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
    le = encoders[col]
    input_data[col] = le.transform(input_data[col])

# Predict
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.error(f'⚠️ Flight is likely to be Delayed! (Confidence: {prediction_proba:.2f})')
else:
    st.success(f'✅ Flight is likely On-Time! (Confidence: {1 - prediction_proba:.2f})')

# Explain prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)

st.subheader('SHAP Explanation')
shap.force_plot(
    explainer.expected_value[1], 
    shap_values[1], 
    input_data,
    matplotlib=True
)
