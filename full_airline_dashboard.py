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
    st.header('🔍 Analyze Flight Delay Performance')

    # --- Step 1: User selects options ---
    with st.form(key='selection_form'):
        carrier_list = flights_cleaned['carrier'].unique()
        selected_carrier = st.selectbox("✈️ Select an Airline Carrier:", carrier_list)

        direction = st.radio("🛫🛬 Choose Flight Type:", ("Departing Flights", "Arriving Flights"))

        submit_button = st.form_submit_button(label='Analyze Selection')

    if submit_button:
        # --- Step 2: Filter flights based on choices ---
        if direction == "Departing Flights":
            filtered_flights = flights_cleaned[flights_cleaned['carrier'] == selected_carrier]
        else:  # Arrival Flights (you can later customize more if needed)
            filtered_flights = flights_cleaned[flights_cleaned['carrier'] == selected_carrier]

        # --- Step 3: Check if flights exist ---
        if filtered_flights.empty:
            st.warning("⚠️ No flights found for this selection. Please try another airline or flight type.")
        else:
            st.success(f"Found {len(filtered_flights)} flights for your selection!")

            # --- Step 4: Prepare data ---
            input_X = filtered_flights[['month', 'day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified', 'distance']]

            for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
                le = encoders[col]
                input_X[col] = le.transform(input_X[col])

            # --- Step 5: Predict delays ---
            y_pred = model.predict(input_X)
            filtered_flights['predicted_delay'] = y_pred

            # --- Step 6: Calculate metrics ---
            delay_rate = (filtered_flights['predicted_delay'].sum() / len(filtered_flights)) * 100
            on_time_rate = 100 - delay_rate
            avg_dep_delay = filtered_flights['dep_delay'].mean()
            avg_arr_delay = filtered_flights['arr_delay'].mean()

            # --- Step 7: Create 2 Rows - 3 Columns Layout ---
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted % Delayed", f"{delay_rate:.2f}%")
                st.metric("Average Departure Delay", f"{avg_dep_delay:.1f} min")

            with col2:
                # --- Pie chart in the middle ---
                import plotly.express as px

                fig = px.pie(
                    names=["Delayed", "On-Time"],
                    values=[delay_rate, on_time_rate],
                    title="Delay Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.metric("Predicted % On-Time", f"{on_time_rate:.2f}%")
                st.metric("Average Arrival Delay", f"{avg_arr_delay:.1f} min")

            # --- Step 8: Second Row ---
            st.subheader("🛫 Top 5 Most Delayed Flights")
            top_delays = filtered_flights.sort_values('dep_delay', ascending=False).head(5)
            st.dataframe(top_delays[['flight', 'origin', 'dest', 'dep_delay', 'arr_delay']])
