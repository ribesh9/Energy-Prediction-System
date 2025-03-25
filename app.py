import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your trained model and any necessary data
model = joblib.load('Model.pkl')
df = pd.read_csv(r"D:\DS\Datasets\Aep_hourly\AEP_hourly.csv")

# Sidebar for user input
st.sidebar.header("User  Input")
year = st.sidebar.number_input("Year", min_value=2000, max_value=2018, value=2018)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1)

# Prepare input data for prediction
# input_data = pd.DataFrame({
#     'year': [year],
#     'month': [month],
#     'day': [day],
#     'week': [week],
#     'time': [time]
# })

input_data = pd.DataFrame({
    'year': [year],
    'month': [month],
})

# Display the input data
st.write("Input Data for Prediction:")
st.write(input_data)

prediction = model.predict(input_data)
st.write(f"Predicted AEP_MW: {prediction[0]}")

# Define X as the feature set
X = df[['year', 'month','week']]
y = df['AEP_MW']

rf = RandomForestRegressor(n_estimators=100, random_state=42)

# After splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Display model performance metrics
st.header("Model Performance Metrics")
# Uncomment and replace with actual metrics
accuracy = 0.85  # Example accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R^2 Score: {r2:.2f}")

# Confusion Matrix (for classification tasks)
# Uncomment and replace with actual confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
st.pyplot()

# Feature Importance
st.header("Feature Importance")
# Uncomment and replace with actual feature importances
importances = model.feature_importances_
features = df.columns[:-1]  # Assuming last column is the target
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Sample feature importance for demonstration
feature_importance_df = pd.DataFrame({
    'Feature': ['year', 'month','week'],
    'Importance': [0.2, 0.3, 0.1, 0.25, 0.15]
})

st.bar_chart(feature_importance_df.set_index('Feature'))

# Run the app
if __name__ == "__main__":
    st.title("Random Forest Model Dashboard")
    st.write("This dashboard displays the performance of a Random Forest model.")