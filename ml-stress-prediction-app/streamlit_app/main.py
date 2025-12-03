import streamlit as st
from pages import analysis, prediction, training

st.set_page_config(page_title="Photoelastic ML Workspace", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analysis", "ML Prediction", "Model Training"])

if page == "Analysis":
    analysis.render()
elif page == "ML Prediction":
    prediction.render()
elif page == "Model Training":
    training.render()