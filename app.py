import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Load trained model and threshold
# --------------------------------------------------

model = joblib.load("credit_card_fraud_pipeline.pkl")
threshold = joblib.load("fraud_threshold.pkl")

# Feature names used during training
feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Enter the transaction features below and the machine learning "
    "model will estimate the probability of fraud."
)

st.info(f"Decision threshold: {threshold}")

# --------------------------------------------------
# Input section
# --------------------------------------------------

st.subheader("Transaction Details")

values = {}

for feature in feature_names:
    values[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.6f"
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Check Transaction"):

    transaction = pd.DataFrame(
        [values],
        columns=feature_names
    )

    fraud_probability = model.predict_proba(
        transaction
    )[0][1]

    if fraud_probability >= threshold:
        prediction = "Fraudulent Transaction"
    else:
        prediction = "Legitimate Transaction"

    st.subheader("Prediction")

    if prediction == "Fraudulent Transaction":
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric(
        "Model-estimated Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    st.write(
        f"The model uses a threshold of **{threshold}** "
        f"for the final decision."
    )