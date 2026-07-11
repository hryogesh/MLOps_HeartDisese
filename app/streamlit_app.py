import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import predict

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.markdown(
    """
    <style>
    .main {padding-top: 1rem;}
    div[data-testid="stMetric"] {background-color: #f5f7ff; border: 1px solid #dce6ff; padding: 0.8rem; border-radius: 0.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    # ❤️ Heart Disease Risk Predictor
    """,
    unsafe_allow_html=True,
)
st.caption("Enter the clinical details below to estimate the likelihood of heart disease.")

st.markdown(
    """
    <div style="background-color:#f7f9ff; padding: 1rem 1.2rem; border-radius: 0.8rem; border: 1px solid #dce6ff; margin-bottom: 1rem;">
    This demo uses the trained model artifact to provide a quick risk assessment.<br>
    Please enter values for the 13 clinical features in a realistic range.
    </div>
    """,
    unsafe_allow_html=True,
)

feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal"
]

feature_config = {
    "age": {"label": "Age (years)", "min_value": 18.0, "max_value": 100.0, "step": 1.0, "value": 50.0},
    "sex": {"label": "Sex", "min_value": 0.0, "max_value": 1.0, "step": 1.0, "value": 1.0},
    "cp": {"label": "Chest pain type", "min_value": 0.0, "max_value": 3.0, "step": 1.0, "value": 1.0},
    "trestbps": {"label": "Resting blood pressure", "min_value": 80.0, "max_value": 220.0, "step": 1.0, "value": 120.0},
    "chol": {"label": "Cholesterol", "min_value": 100.0, "max_value": 600.0, "step": 1.0, "value": 200.0},
    "fbs": {"label": "Fasting blood sugar", "min_value": 0.0, "max_value": 1.0, "step": 1.0, "value": 0.0},
    "restecg": {"label": "Resting ECG", "min_value": 0.0, "max_value": 2.0, "step": 1.0, "value": 0.0},
    "thalach": {"label": "Max heart rate", "min_value": 60.0, "max_value": 220.0, "step": 1.0, "value": 150.0},
    "exang": {"label": "Exercise-induced angina", "min_value": 0.0, "max_value": 1.0, "step": 1.0, "value": 0.0},
    "oldpeak": {"label": "ST depression", "min_value": 0.0, "max_value": 6.0, "step": 0.1, "value": 1.0},
    "slope": {"label": "ST slope", "min_value": 0.0, "max_value": 2.0, "step": 1.0, "value": 1.0},
    "ca": {"label": "Major vessels", "min_value": 0.0, "max_value": 4.0, "step": 1.0, "value": 0.0},
    "thal": {"label": "Thalassemia", "min_value": 0.0, "max_value": 3.0, "step": 1.0, "value": 2.0},
}

with st.container():
    cols = st.columns(2)
    features = []
    for index, name in enumerate(feature_names):
        config = feature_config[name]
        with cols[index % 2]:
            value = st.number_input(
                config["label"],
                min_value=config["min_value"],
                max_value=config["max_value"],
                value=config["value"],
                step=config["step"],
                key=name,
            )
            features.append(float(value))

st.divider()

if st.button("Predict Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing the provided clinical profile..."):
        result = predict(features)

    prediction = "High risk" if int(result["prediction"]) == 1 else "Low risk"
    probability = float(result["probability"]) * 100

    if prediction == "High risk":
        st.error(f"Estimated risk: {prediction}")
    else:
        st.success(f"Estimated risk: {prediction}")

    st.metric("Model confidence", f"{probability:.1f}%")
    st.caption("The model output is based on the latest trained artifact in the repository.")

with st.expander("How this app works"):
    st.write(
        "The predictor uses the trained heart disease model and the preprocessing pipeline "
        "to transform your inputs into a meaningful risk estimate."
    )
    st.write("Tip: use realistic values for the clinical features for the most meaningful prediction.")
