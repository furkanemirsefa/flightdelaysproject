import pandas as pd
import shap
import joblib
import streamlit as st
import plotly.express as px
from streamlit_shap import st_shap


@st.cache_resource
def load_model():
    return joblib.load('flight_delay_model.pkl')

@st.cache_resource
def load_encoders():
    return joblib.load('encoders.pkl')

@st.cache_resource
def load_cleaned_data():
    return pd.read_csv('your_cleaned_flight_data.csv')

# Load resources lazily
model = load_model()
encoders = load_encoders()
flights_cleaned = load_cleaned_data()

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

    # Sample 1 random flight
    input_sample = flights_cleaned.sample(1, random_state=42)
    input_X = input_sample[['month', 'day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified', 'distance']]

    # Encode categoricals
    for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
        le = encoders[col]
        input_X[col] = le.transform(input_X[col])

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_X)

    st.subheader('Random Sample Prediction Explanation')
    st.write('Sample Flight Details:', input_sample)

    sample_idx = 0

    st_shap(shap.plots.force(explainer.expected_value[0],shap_values[sample_idx, :]),height=300)
    st.markdown("""
    ### ℹ️ How to Interpret This Force Plot

    - **Center point:** represents the model's base prediction (average across all flights).
    - **Red arrows:** features that push the flight **toward being delayed**.
    - **Blue arrows:** features that push the flight **toward being on-time**.
    - **Arrow size:** bigger arrow means a stronger influence on the prediction.

    ---
    ✅ If most of the large arrows are **red**, the model predicts the flight will likely be **delayed**.  
    ✅ If most of the large arrows are **blue**, the model predicts the flight will likely be **on-time**.

    ---
    ### 📈 Quick Example:
    - A **Late Night Departure** or **High Traffic Carrier** pushes toward **delay** (red).
    - A **Morning Departure** or **Short Distance Flight** pushes toward **on-time** (blue).
    ---
    """)

    # Create a Matplotlib figure for feature importance (downloadable version)

    import matplotlib.pyplot as plt
    
    # Get feature names and SHAP values
    feature_importances = pd.DataFrame({
        'feature': input_X.columns,
        'shap_value': shap_values[sample_idx, :]
    }).sort_values('shap_value', key=abs, ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    feature_importances.plot.barh(
        x='feature',
        y='shap_value',
        ax=ax,
        color="skyblue",
        legend=False
    )
    ax.set_title("Feature Contributions to Prediction")
    ax.set_xlabel("SHAP Value")
    plt.tight_layout()
    
    # Save to buffer
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.download_button(
        label="📥 Download Explanation as PNG",
        data=buf.getvalue(),
        file_name="flight_delay_explanation.png",
        mime="image/png"
    )

    

    st.info('Reload page to see another random flight explanation!')
