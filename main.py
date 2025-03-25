import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

st.set_page_config(page_title="LSTM Time Series Predictor", layout="wide")

st.title("LSTM Time Series Predictor")

# Load pre-trained model from the backend
MODEL_PATH = "PredictionModel.pkl"  # Path to the model stored in the backend

# @st.cache_data(allow_output_mutation=True)
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

# Load model once at startup
model = load_model()
st.success("Model loaded successfully from the backend!")

# Input Year for Prediction
st.sidebar.header("Enter Year for Prediction")
selected_year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, step=1, value=datetime.now().year)

# Generate Predictions
if st.sidebar.button("Generate Predictions"):
    with st.spinner("Predicting..."):
        # Generate input data based on the selected year
        future_dates = pd.date_range(start=f"{selected_year}-01-01", end=f"{selected_year}-12-31", freq="D")
        predictions = model.predict(np.array(range(len(future_dates))).reshape(-1, 1))
        
        pred_df = pd.DataFrame({"date": future_dates, "predicted": predictions.flatten()})
        
        # Visualization
        st.subheader("Predictions for the Year")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pred_df["date"], pred_df["predicted"], label="Predictions", linestyle="dashed", color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)
        
        # Display Data
        st.subheader("Prediction Results")
        st.dataframe(pred_df)