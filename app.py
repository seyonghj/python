import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Duration Predictor",
    page_icon="🌙",
    layout="centered"
)

# ── Load model, scaler, columns ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "sleep_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "sleep_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "sleep_columns.pkl"), "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns

try:
    model, scaler, columns = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🌙 Sleep Duration Predictor")
st.markdown("Fill in your details below to predict how many hours of sleep you need.")
st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    age = st.slider("Age", min_value=18, max_value=70, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox("Occupation", [
        "Accountant", "Doctor", "Engineer", "Lawyer", "Manager",
        "Nurse", "Sales Representative", "Salesperson",
        "Scientist", "Software Engineer", "Teacher"
    ])
    bmi = st.selectbox("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"])
    sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])

with col2:
    st.subheader("❤️ Health Metrics")
    quality_of_sleep = st.slider("Quality of Sleep (1–10)", 1, 10, 7)
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
    physical_activity = st.slider("Physical Activity Level (min/day)", 0, 90, 30)
    heart_rate = st.slider("Heart Rate (bpm)", 50, 100, 70)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=20000, value=7000, step=500)
    bp_systolic = st.slider("Blood Pressure — Systolic", 90, 180, 120)
    bp_diastolic = st.slider("Blood Pressure — Diastolic", 60, 120, 80)

st.divider()

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Sleep Duration", use_container_width=True, type="primary"):

    if not model_loaded:
        st.error(
            "⚠️ Model files not found! Make sure `sleep_model.pkl`, "
            "`sleep_scaler.pkl`, and `sleep_columns.pkl` are in the same folder as this app."
        )
    else:
        # Build input row with all zeros
        input_data = pd.DataFrame([{col: 0 for col in columns}])

        # Fill numeric features
        input_data["Age"] = age
        input_data["Quality of Sleep"] = quality_of_sleep
        input_data["Physical Activity Level"] = physical_activity
        input_data["Stress Level"] = stress_level
        input_data["Heart Rate"] = heart_rate
        input_data["Daily Steps"] = daily_steps
        input_data["BP_Systolic"] = bp_systolic
        input_data["BP_Diastolic"] = bp_diastolic

        # Fill one-hot encoded features
        if gender == "Male" and "Gender_Male" in columns:
            input_data["Gender_Male"] = 1

        occ_col = f"Occupation_{occupation}"
        if occ_col in columns:
            input_data[occ_col] = 1

        bmi_col = f"BMI Category_{bmi}"
        if bmi_col in columns:
            input_data[bmi_col] = 1

        if sleep_disorder == "None" and "Sleep Disorder_None" in columns:
            input_data["Sleep Disorder_None"] = 1
        elif sleep_disorder == "Sleep Apnea" and "Sleep Disorder_Sleep Apnea" in columns:
            input_data["Sleep Disorder_Sleep Apnea"] = 1
        # Insomnia is the baseline (drop_first=True), so no column needed

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = round(float(prediction), 2)

        hours = int(prediction)
        minutes = int((prediction - hours) * 60)

        # ── Result display ───────────────────────────────────────────────────
        st.success(f"### 🛏️ Predicted Sleep Duration: **{prediction} hours** ({hours}h {minutes}m)")

        # Gauge bar
        pct = min(prediction / 10, 1.0)
        if prediction < 6:
            color = "#E74C3C"
            label = "⚠️ Below recommended (< 6 hrs)"
        elif prediction <= 9:
            color = "#2ECC71"
            label = "✅ Within healthy range (6–9 hrs)"
        else:
            color = "#F39C12"
            label = "⚠️ Above average (> 9 hrs)"

        st.markdown(f"""
        <div style='background:#eee;border-radius:10px;height:20px;width:100%;margin-bottom:6px'>
            <div style='background:{color};width:{pct*100:.0f}%;height:20px;border-radius:10px'></div>
        </div>
        <p style='color:{color};font-weight:bold'>{label}</p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Based on a Linear Regression model trained on the Sleep Health and Lifestyle Dataset.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center><small>Lab 5.0 — Artificial Intelligence | Sleep Duration Predictor</small></center>",
    unsafe_allow_html=True
)
