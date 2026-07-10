import streamlit as st

from src.models.predict import predict

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("Heart Disease Risk Predictor")

st.write("Enter 13 clinical features to receive a prediction.")

features = []
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal"
]

for name in feature_names:
    value = st.number_input(name, value=0.0, step=0.1)
    features.append(value)

if st.button("Predict"):
    result = predict(features)
    st.success(f"Prediction: {result['prediction']}")
    st.info(f"Probability: {result['probability']:.2f}")
