import streamlit as st
import cv2
import numpy as np
from src.models.stress_predictor import StressPredictor
from src.models.fringe_estimator import FringeEstimator
from src.prediction.predict import make_predictions
from src.preprocessing.data_loader import DataLoader

# Initialize models
stress_model = StressPredictor()
fringe_model = FringeEstimator()

# Load models (assuming models are saved and loaded here)
stress_model.load_model('path/to/stress_model')
fringe_model.load_model('path/to/fringe_model')

# Streamlit interface for predictions
st.title("Stress Prediction and Fringe Order Estimation")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    data_loader = DataLoader()
    image = data_loader.load_image(uploaded_file)
    
    # Make predictions
    stress_prediction, fringe_prediction = make_predictions(image, stress_model, fringe_model)
    
    # Display results
    st.subheader("Predictions")
    st.write(f"Stress Level: {stress_prediction}")
    st.write(f"Fringe Order: {fringe_prediction}")