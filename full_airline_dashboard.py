import streamlit as st
import pandas as pd
import shap
import joblib
import plotly.express as px

@st.cache_resource
def load_model():
    return joblib.load('flight_delay_model.pkl')

@st.cache_resource
def load_encoders():
    return joblib.load('encoders.pkl')

@st.cache_resource
def load_cleaned_data():
    return pd.read_csv('your_cleaned_flight_data.csv')

# Load them
model = load_model()
encoders = load_encoders()
flights_cleaned = load_cleaned_data()
