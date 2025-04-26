# shap_explanation_notebook.py

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your cleaned dataset
flights = pd.read_csv('your_cleaned_flight_data.csv')

# Define features and target
target = 'is_delayed'
features = ['month', 'day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified', 'distance']
X = flights[features]
y = flights[target]

# Encode categorical features
label_encoders = {}
for col in ['day_of_week', 'part_of_day', 'carrier_simplified', 'origin_simplified', 'dest_simplified']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test)

# Force plot for a single prediction
sample_idx = 0  # Change index to check different flights
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][sample_idx], X_test.iloc[sample_idx])

# Optional: Dependence plot
# shap.dependence_plot('dep_hour', shap_values[1], X_test)
