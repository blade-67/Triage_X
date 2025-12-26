import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load Saved Objects
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model_voting.pkl")
        imputer = joblib.load("imputer.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, imputer, feature_columns
    except FileNotFoundError:
        st.error("⚠️ Models not found! Please run 'python model.py' first to train models.")
        st.stop()

model, imputer, feature_columns = load_models()

# Triage Level Mapping
triage_levels = {
    0: ("🟢 Non-Urgent", "Low priority"),
    1: ("🟡 Urgent", "Moderate priority"),
    2: ("🔴 Very Urgent", "High priority"),
    3: ("🚨 Critical", "Life-threatening")
}

# Create the Streamlit UI
st.set_page_config(page_title="Emergency Triage", layout="wide")
st.title("🧠 AI-Assisted Emergency Triage System")
st.write("Enter patient details to get triage priority prediction")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    age = st.slider("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    chronic_condition = st.selectbox("Chronic Condition", ["No", "Yes"])

with col2:
    st.subheader("💉 Vital Signs")
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 80)
    systolic_bp = st.number_input("Systolic BP (mmHg)", 70, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)
    spo2 = st.number_input("SpO₂ (%)", 75, 100, 96)
    respiratory_rate = st.number_input("Respiratory Rate", 8, 45, 18)
    temperature = st.number_input("Temperature (°F)", 95.0, 104.0, 98.4)

with col3:
    st.subheader("📋 Clinical")
    pain_score = st.slider("Pain Score", 0, 10, 3)
    consciousness = st.selectbox("Consciousness", ["Unconscious", "Conscious"])
    arrival_mode = st.selectbox("Arrival Mode", ["Walk-in", "Ambulance"])

# Convert Inputs to Model Format
input_data = pd.DataFrame([{
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "heart_rate": heart_rate,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "spo2": spo2,
    "respiratory_rate": respiratory_rate,
    "temperature": temperature,
    "pain_score": pain_score,
    "consciousness": 1 if consciousness == "Conscious" else 0,
    "arrival_mode": 1 if arrival_mode == "Ambulance" else 0,
    "chronic_condition": 1 if chronic_condition == "Yes" else 0
}])

# Ensure correct column order
input_data = input_data[feature_columns]

# Make prediction
st.markdown("---")
if st.button("🔍 Analyze Patient", use_container_width=True):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    confidence = max(prediction_proba) * 100
    
    level_text, level_desc = triage_levels[prediction]
    
    # Display results
    col1, col2 = st.columns([2, 1])
    with col1:
        if prediction == 3:
            st.error(f"### {level_text}")
            st.error(f"**Priority:** {level_desc}")
            st.error("⚠️ **ACTION:** Immediate physician evaluation, emergency protocols")
        elif prediction == 2:
            st.warning(f"### {level_text}")
            st.warning(f"**Priority:** {level_desc}")
            st.warning("⚠️ **ACTION:** Urgent evaluation within 30 minutes")
        elif prediction == 1:
            st.info(f"### {level_text}")
            st.info(f"**Priority:** {level_desc}")
            st.info("ℹ️ **ACTION:** Standard triage assessment")
        else:
            st.success(f"### {level_text}")
            st.success(f"**Priority:** {level_desc}")
            st.success("✓ **ACTION:** Routine waiting room assessment")
    
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
