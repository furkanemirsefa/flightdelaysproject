# full_airline_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load model and encoders
model = joblib.load('flight_delay_model.pkl')
encoders = joblib.load('encoders.pkl')

# Load small sample or clean dataset for analytics
flights_cleaned = pd.read_csv('your_cleaned_flight_data.csv')

# Sidebar selection
st.sidebar.title('✈️ Airline Dashboard')
section = st.sidebar.selectbox('Go to:', ["Overview Analytics", "Predict Flight Delay", "Explain Prediction"])

st.title('Airline Delay Insights Dashboard')

if section == "Overview Analytics":
    st.header('📊 Overview of Flight Delays')

    avg_delays = flights_cleaned.groupby('carrier_simplified')['dep_delay'].mean().sort_values()
    fig1 = px.bar(avg_delays, orientation='h', title='Average Departure Delay by Airline')
    st.plotly_chart(fig1)

    monthly_delays = flights_cleaned.groupby('month')['dep_delay'].mean()
    fig2 = px.line(monthly_delays, title='Monthly Average Departure Delay')
    st.plotly_chart(fig2)

    top_origins = flights_cleaned.groupby('origin_simplified')['dep_delay'].mean().sort_values(ascending=False).head(5)
    st.subheader('Worst Origin Airports for Delays')
    st.write(top_origins)

elif section == "Predict Flight Delay":
    st.header('🎯 Predict Flight Delay')

    month = st.selectbox('Month', list(range(1, 13)))
    part_of_day = st.selectbox('Part of Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    carrier = st.selectbox('Carrier', ['Other', 'UA', 'AA', 'DL', 'B6', 'WN'])
    origin = st.selectbox('Origin Airport', ['Other', 'JFK', 'EWR', 'LGA'])
    dest = st.selectbox('Destination Airport', ['Other', 'ORD', 'ATL', 'LAX'])
    distance = st.slider('Flight Distance (miles)', 100, 3000, 500)

    input_data = pd.DataFrame({
        'month': [month],
        'day_of_week': [day_of_week],
        'part_of_day': [part_of_day],
        'carrier_simplified': [carrier],
        'origin_simplified': [origin],
        'dest_simplified': [dest],
        'distance': [distance]
    })

    for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
        le = encoders[col]
        input_data[col] = le.transform(input_data[col])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f'⚠️ Flight likely Delayed (Confidence: {prediction_proba:.2f})')
    else:
        st.success(f'✅ Flight likely On-Time (Confidence: {1 - prediction_proba:.2f})')

elif section == "Explain Prediction":
    st.header('🔍 Explain the Flight Delay Prediction')

    shap.initjs()

    input_sample = flights_cleaned.sample(1, random_state=42)
    input_X = input_sample[['month', 'day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified', 'distance']]

    for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
        le = encoders[col]
        input_X[col] = le.transform(input_X[col])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_X)

    st.subheader('Random Sample Prediction Explanation')
    st.write('Sample Flight Details:', input_sample)

    st.pyplot(shap.force_plot(explainer.expected_value[1], shap_values[1], input_X, matplotlib=True))

    st.info('Reload page to see another random flight explanation!')
