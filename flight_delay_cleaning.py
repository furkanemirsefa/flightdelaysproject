# flight_delay_cleaning.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load data
flights = pd.read_csv('flights.csv')
airlines = pd.read_csv('airlines_carrier_codes.csv')

# Merge airline names
airlines.columns = ['carrier', 'airline_name']
flights = flights.merge(airlines, on='carrier', how='left')

# Simplify dep_time and arr_time
flights['dep_hour'] = (flights['dep_time'] // 100).fillna(0).astype(int)
flights['dep_minute'] = (flights['dep_time'] % 100).fillna(0).astype(int)
flights['arr_hour'] = (flights['arr_time'] // 100).fillna(0).astype(int)
flights['arr_minute'] = (flights['arr_time'] % 100).fillna(0).astype(int)

# Drop cancelled flights
flights_cleaned = flights.dropna(subset=['dep_delay', 'arr_delay'])

# Create date and extract day of week
flights_cleaned['flight_date'] = pd.to_datetime(flights_cleaned[['year', 'month', 'day']])
flights_cleaned['day_of_week'] = flights_cleaned['flight_date'].dt.day_name()

# Season feature
season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
flights_cleaned['season'] = flights_cleaned['month'].map(season_map)

# Part of day
part_of_day = lambda h: 'Morning' if 5 <= h < 12 else ('Afternoon' if 12 <= h < 17 else ('Evening' if 17 <= h < 21 else 'Night'))
flights_cleaned['part_of_day'] = flights_cleaned['dep_hour'].apply(part_of_day)

# Label for delay
flights_cleaned['is_delayed'] = (flights_cleaned['dep_delay'] > 15).astype(int)

# Simplify carrier, origin, dest
for col, top_n in [('carrier', 5), ('origin', 5), ('dest', 5)]:
    top_items = flights_cleaned[col].value_counts().nlargest(top_n).index
    flights_cleaned[col + '_simplified'] = flights_cleaned[col].apply(lambda x: x if x in top_items else 'Other')

# Features
features = ['month', 'day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified', 'distance']
X = flights_cleaned[features]
y = flights_cleaned['is_delayed']

# Encode categoricals
label_encoders = {}
for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
    le = LabelEncoder()
    unique_vals = flights_cleaned[col].unique().tolist()
    if 'Other' not in unique_vals:
        unique_vals.append('Other')
    le.fit(unique_vals)
    X[col] = le.transform(flights_cleaned[col])
    label_encoders[col] = le


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Optional: Save the model and encoders with joblib
import joblib
joblib.dump(model, 'flight_delay_model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
# Save the cleaned dataset for dashboard
flights_cleaned.to_csv('your_cleaned_flight_data.csv', index=False)
