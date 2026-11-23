
import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
model = joblib.load("ckd_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🏥",
    layout="centered"
)


st.title("🩺 Chronic Kidney Disease Prediction")
st.write("Enter the patient's details below to predict the likelihood of CKD.")

st.divider()

with st.form("ckd_form"):

    st.subheader("Patient Information")

    age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0.0, value=80.0)
    al = st.number_input("Albumin (0–5)", min_value=0, max_value=5, value=0)
    su = st.number_input("Sugar (0–5)", min_value=0, max_value=5, value=0)

    bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=0.0, value=120.0)
    bu = st.number_input("Blood Urea (mg/dL)", min_value=0.0, value=30.0)
    sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.0)
    hemo = st.number_input("Haemoglobin (g/dL)", min_value=0.0, value=14.0)
    sod = st.number_input("Sodium", min_value=0.0)
    pcv = st.number_input("Packed Cell Volume (%)", min_value=0.0, value=42.0)
    wc = st.number_input("White Blood Cell Count", min_value=0.0)
    
    htn = st.selectbox("Hypertension", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])


    submitted = st.form_submit_button("Predict CKD")

if submitted:

    patient_data = {
        "age": age,
        "bp": bp,
        "al": al,
        "bgr": bgr,
        "bu": bu,
        "sc": sc,
        "sod": sod,

        "hemo": hemo,
        "pcv": pcv,
        "wc": wc,
        "htn": htn,
        "dm": dm,

    }

    patient = pd.DataFrame([patient_data])

    # Apply the same preprocessing used during training
    patient = pd.get_dummies(patient, drop_first=True)
    patient = patient.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(patient)[0]

    st.subheader("Prediction Result")

    if prediction == "ckd":
        st.error("Model prediction: Chronic kidney disease suspected")
    else:
        st.success("Model prediction: normal kidney")

    st.warning(
        "Educational machine-learning output only. "
        "This is not a clinical diagnosis."
    )