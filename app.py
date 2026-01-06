import streamlit as st
import pandas as pd
import joblib

# 1. Load the "Saved Brain"
model = joblib.load('best_model.joblib')

# 2. Design the Website Interface
st.title("🏥 Patient No-Show Predictor")
st.markdown("Predict the likelihood of a patient missing their appointment to optimize clinic scheduling.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    with col2:
        distance = st.number_input("Distance to Clinic (miles)", min_value=0.0, value=5.0)
        
    notes = st.text_area("Doctor's Health Notes", placeholder="e.g., Patient reports chronic pain...")
    
    submit = st.form_submit_button("Analyze Risk")

# 3. Handle the Prediction
if submit:
    # Create a dataframe exactly like the one used in training
    input_data = pd.DataFrame({
        'age': [age],
        'distance': [distance],
        'health_notes': [notes]
    })
    
    # Predict!
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Chance of no-show
    
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: {probability:.2%} chance of No-Show.")
        st.write("Recommendation: Call patient 24 hours in advance.")
    else:
        st.success(f"✅ LOW RISK: {probability:.2%} chance of No-Show.")