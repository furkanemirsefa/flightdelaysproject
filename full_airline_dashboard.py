import streamlit as st
import pandas as pd
import shap
import joblib
import plotly.express as px

# Load model, encoders, cleaned data
model = joblib.load('flight_delay_model.pkl')
encoders = joblib.load('encoders.pkl')
flights_cleaned = pd.read_csv('your_cleaned_flight_data.csv')

# Setup Streamlit page
st.set_page_config(layout="wide")
st.title("✈️ Flight Delay Prediction and Analytics Dashboard")

# ---------------------------------------------
# 📊 SECTION 1: Overview Analytics
# ---------------------------------------------
st.header("📊 Overview of Airlines and Airports")

overview_col1, overview_col2 = st.columns(2)

with overview_col1:
    st.subheader("Average Departure Delay by Airline")
    dep_delay_by_carrier = flights_cleaned.groupby('carrier')['dep_delay'].mean().sort_values()
    fig1 = px.bar(dep_delay_by_carrier, title="Avg Departure Delay by Airline (minutes)")
    st.plotly_chart(fig1, use_container_width=True)

with overview_col2:
    st.subheader("Top 10 Busiest Airports (Origins)")
    top_origins = flights_cleaned['origin'].value_counts().nlargest(10)
    fig2 = px.bar(top_origins, title="Top 10 Departure Airports")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------
# 🔮 SECTION 2: Predict a Specific Flight Delay
# ---------------------------------------------
st.header("🔮 Predict Delay for a Specific Flight")

with st.form("predict_form"):
    month = st.selectbox("Month", sorted(flights_cleaned['month'].unique()))
    day_of_week = st.selectbox("Day of Week", sorted(flights_cleaned['day_of_week'].unique()))
    part_of_day = st.selectbox("Part of Day", flights_cleaned['part_of_day'].unique())
    carrier = st.selectbox("Carrier", flights_cleaned['carrier_simplified'].unique())
    origin = st.selectbox("Origin Airport", flights_cleaned['origin_simplified'].unique())
    dest = st.selectbox("Destination Airport", flights_cleaned['dest_simplified'].unique())
    distance = st.number_input("Flight Distance (miles)", min_value=1, value=300)

    predict_button = st.form_submit_button("Predict Delay")

if predict_button:
    input_dict = {
        'month': month,
        'day_of_week': encoders['day_of_week'].transform([day_of_week])[0],
        'part_of_day': encoders['part_of_day'].transform([part_of_day])[0],
        'carrier_simplified': encoders['carrier_simplified'].transform([carrier])[0],
        'origin_simplified': encoders['origin_simplified'].transform([origin])[0],
        'dest_simplified': encoders['dest_simplified'].transform([dest])[0],
        'distance': distance
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Prediction: Flight will likely be Delayed! (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Prediction: Flight will likely be On-Time! (Confidence: {1-probability:.2f})")

# ---------------------------------------------
# 🔍 SECTION 3: Airline + Airport Performance Analytics
# ---------------------------------------------
st.header("🔍 Analyze Airline and Airport Delay Performance")

with st.form("group_form"):
    carrier_list = flights_cleaned['carrier'].unique()
    selected_carrier = st.selectbox("✈️ Select Airline", carrier_list)

    direction = st.radio("🛫🛬 Departing or Arriving?", ("Departing Flights", "Arriving Flights"))

    if direction == "Departing Flights":
        airport_list = flights_cleaned[flights_cleaned['carrier'] == selected_carrier]['origin'].unique()
    else:
        airport_list = flights_cleaned[flights_cleaned['carrier'] == selected_carrier]['dest'].unique()

    selected_airport = st.selectbox("🏢 Select Airport", airport_list)
    analyze_button = st.form_submit_button("Analyze Selection")

if analyze_button:
    if direction == "Departing Flights":
        filtered_flights = flights_cleaned[
            (flights_cleaned['carrier'] == selected_carrier) &
            (flights_cleaned['origin'] == selected_airport)
        ]
    else:
        filtered_flights = flights_cleaned[
            (flights_cleaned['carrier'] == selected_carrier) &
            (flights_cleaned['dest'] == selected_airport)
        ]

    if filtered_flights.empty:
        st.warning("⚠️ No flights found for your selection. Please try again.")
    else:
        st.success(f"Found {len(filtered_flights)} flights for your selection.")

        input_X = filtered_flights[['month', 'day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified', 'distance']]
        for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
            le = encoders[col]
            input_X[col] = le.transform(input_X[col])

        y_pred = model.predict(input_X)
        filtered_flights['predicted_delay'] = y_pred

        delay_rate = (filtered_flights['predicted_delay'].sum() / len(filtered_flights)) * 100
        on_time_rate = 100 - delay_rate
        avg_dep_delay = filtered_flights['dep_delay'].mean()
        avg_arr_delay = filtered_flights['arr_delay'].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted % Delayed", f"{delay_rate:.2f}%")
            st.metric("Average Departure Delay", f"{avg_dep_delay:.1f} min")

        with col2:
            fig = px.pie(
                names=["Delayed", "On-Time"],
                values=[delay_rate, on_time_rate],
                title="Delay Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.metric("Predicted % On-Time", f"{on_time_rate:.2f}%")
            st.metric("Average Arrival Delay", f"{avg_arr_delay:.1f} min")

        st.subheader("🛫 Top 5 Most Delayed Flights")
        top_delays = filtered_flights.sort_values('dep_delay', ascending=False).head(5)
        st.dataframe(top_delays[['flight', 'origin', 'dest', 'dep_delay', 'arr_delay']])
