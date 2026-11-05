"""
Step 9: Streamlit Frontend Web App for Customer Lifetime Value (CLV) Prediction
Run locally via: streamlit run app.py
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Customer Lifetime Value (CLV) Predictor",
    page_icon="💸",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    """Load trained sklearn artifacts if present, or train/load from JSON fallback."""
    if os.path.exists('linear_clv_model.pkl') and os.path.exists('clv_scaler.pkl'):
        import joblib
        model = joblib.load('linear_clv_model.pkl')
        scaler = joblib.load('clv_scaler.pkl')
        return {"mode": "sklearn", "model": model, "scaler": scaler}
    elif os.path.exists('model_artifacts.json'):
        with open('model_artifacts.json', 'r') as f:
            meta = json.load(f)
        return {"mode": "json", "meta": meta}
    else:
        raise FileNotFoundError("Run `python train_pipeline.py` first to generate model artifacts.")

artifacts = load_artifacts()

st.title("💸 Customer Lifetime Value (CLV) Predictor")
st.write("Input consumer engagement metrics below to calculate predicted yearly revenue contributions and simulate ROI scenarios.")

col_form, col_result = st.columns([1.15, 1])

with col_form:
    st.subheader("Customer Engagement Controls")
    session_length = st.slider("Average Session Length (Minutes)", 20.0, 50.0, 34.0, 0.5)
    time_on_app = st.slider("Time spent on Mobile App (Minutes)", 5.0, 20.0, 12.0, 0.25)
    time_on_web = st.slider("Time spent on Desktop Website (Minutes)", 10.0, 50.0, 36.0, 0.5)
    membership_length = st.slider("Length of Membership (Years)", 0.0, 10.0, 3.5, 0.1)

    calculate_clicked = st.button("Calculate Expected Value", type="primary", use_container_width=True)

# Compute prediction continuously and highlight on button press
input_data = np.array([[session_length, time_on_app, time_on_web, membership_length]])

if artifacts["mode"] == "sklearn":
    scaled_input = artifacts["scaler"].transform(input_data)
    prediction = float(artifacts["model"].predict(scaled_input)[0])
else:
    meta = artifacts["meta"]
    mean = np.array(meta["scaler_mean"])
    scale = np.array(meta["scaler_scale"])
    coefs = np.array([meta["scaled_coefficients"][c] for c in meta["feature_cols"]])
    scaled_input = (input_data[0] - mean) / scale
    prediction = float(meta["intercept"] + np.dot(scaled_input, coefs))

app_web_ratio = time_on_app / max(time_on_web, 0.1)
engagement_score = session_length * membership_length

with col_result:
    st.subheader("Real-Time CLV Forecast")
    if calculate_clicked:
        st.success(f"### Predicted Yearly Spending: ${prediction:,.2f}")
    else:
        st.info(f"### Predicted Yearly Spending: ${prediction:,.2f}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Monthly Equivalent", f"${prediction / 12.0:,.2f}")
    m2.metric("App-to-Web Ratio", f"{app_web_ratio:.2f}x")
    m3.metric("Engagement Score", f"{engagement_score:.1f}")

    st.markdown("#### Key Spending Drivers (Standardized Model Weights)")
    drivers_df = pd.DataFrame({
        "Feature": ["Length of Membership", "Time on App", "Avg. Session Length", "Time on Website"],
        "Scaled Coefficient ($ / SD)": [55.13, 37.34, 25.30, -0.44],
        "Unscaled Impact ($ / unit)": ["+$61.20 / yr", "+$38.33 / min", "+$25.56 / min", "-$0.43 / min"]
    })
    st.dataframe(drivers_df, hide_index=True, use_container_width=True)
