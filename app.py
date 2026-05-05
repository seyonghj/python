import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import random

st.set_page_config(
    page_title="Sleep Duration Predictor",
    page_icon="🌙",
    layout="wide"  # changed to wide to accommodate sidebar
)

# ── Load model ────────────────────────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌙 Sleep Tools")
    st.divider()

    # --- Sleep Tips ---
    st.subheader("💡 Sleep Tips")
    all_tips = [
        "Keep your bedroom between 60–67°F (15–19°C) for optimal sleep.",
        "Avoid screens at least 30 minutes before bed — blue light delays melatonin production.",
        "Try the 4-7-8 breathing method: inhale 4s, hold 7s, exhale 8s.",
        "Go to bed and wake up at the same time every day — even on weekends.",
        "Avoid caffeine after 2 PM; it has a half-life of about 6 hours.",
        "Exercise regularly, but avoid intense workouts within 2 hours of bedtime.",
        "Limit alcohol — it suppresses REM sleep and reduces sleep quality.",
        "Keep naps under 20 minutes to avoid grogginess.",
        "Write a to-do list before bed to offload anxious thoughts.",
        "Dim your lights 1–2 hours before bedtime to signal your brain it's time to sleep.",
    ]
    # Rotate tips daily using the day of year as seed
    random.seed(pd.Timestamp.now().day_of_year)
    daily_tips = random.sample(all_tips, 3)
    for tip in daily_tips:
        st.markdown(f"- {tip}")

    st.button("🔄 New Tips", key="refresh_tips",
              help="Show different tips")

    st.divider()

    # --- Sleep Score Calculator ---
    st.subheader("📊 Sleep Health Score")
    score_hours = st.slider("Your usual sleep (hrs)", 4.0, 12.0, 7.0, 0.5,
                            key="score_hrs")
    score_quality = st.slider("Perceived quality (1–10)", 1, 10, 7,
                              key="score_q")
    score_consistent = st.checkbox("Consistent schedule?", value=True)
    score_caffeine = st.checkbox("No caffeine after 2 PM?", value=True)
    score_screens = st.checkbox("Screen-free 30 min before bed?", value=False)

    # Compute score (0–100)
    hour_score = max(0, 100 - abs(score_hours - 7.5) * 15)
    quality_score = score_quality * 10
    bonus = (10 if score_consistent else 0) + \
            (5 if score_caffeine else 0) + \
            (5 if score_screens else 0)
    raw_score = (hour_score * 0.4 + quality_score * 0.4 + bonus * 1.0)
    sleep_score = min(100, int(raw_score))

    if sleep_score >= 80:
        score_label, score_color = "Excellent 🌟", "green"
    elif sleep_score >= 60:
        score_label, score_color = "Good 👍", "blue"
    elif sleep_score >= 40:
        score_label, score_color = "Fair ⚠️", "orange"
    else:
        score_label, score_color = "Poor 😴", "red"

    st.metric("Your Sleep Score", f"{sleep_score}/100", score_label)
    st.progress(sleep_score / 100)

    st.divider()

    # --- Sleep Cycle Calculator ---
    st.subheader("⏰ Sleep Cycle Planner")
    st.caption("A full sleep cycle is ~90 minutes. Plan your wake time "
               "to complete full cycles and feel refreshed.")
    bedtime = st.time_input("Planned bedtime", value=pd.Timestamp("22:00").time(),
                            key="bedtime")
    cycles = st.radio("Number of cycles", [4, 5, 6],
                      horizontal=True, index=1, key="cycles")
    fall_asleep_mins = st.slider("Minutes to fall asleep", 5, 30, 14, key="fam")

    total_mins = cycles * 90 + fall_asleep_mins
    wake_time = (pd.Timestamp.combine(pd.Timestamp.today(), bedtime)
                 + pd.Timedelta(minutes=total_mins)).time()
    st.success(f"⏰ Wake up at **{wake_time.strftime('%I:%M %p')}** "
               f"({cycles} cycles = {cycles * 1.5:.1f} hrs of sleep)")

    st.divider()

    # --- Sleep Stages Info ---
    with st.expander("🧠 Sleep Stages Explained"):
        st.markdown("""
| Stage | % of night | Role |
|-------|-----------|------|
| Awake | ~5% | Brief arousals |
| Light (N1/N2) | ~45% | Transition & memory |
| Deep (N3) | ~25% | Physical recovery |
| REM | ~25% | Dreams & learning |

Deep sleep peaks in the first half of the night; REM in the second half.
        """)

    # --- Myth Busters ---
    with st.expander("🚫 Sleep Myth Busters"):
        myths = {
            "You can catch up on sleep on weekends.":
                "Chronic sleep debt isn't fully reversible by sleeping in.",
            "Alcohol helps you sleep better.":
                "It reduces REM sleep and worsens overall sleep quality.",
            "Older adults need less sleep.":
                "Adults of all ages need 7–9 hours; sleep just becomes lighter.",
            "Snoring is harmless.":
                "Loud snoring can signal sleep apnea, a serious condition.",
        }
        for myth, fact in myths.items():
            st.markdown(f"**Myth:** {myth}  \n✅ **Fact:** {fact}")
            st.markdown("---")

    # --- Resources ---
    with st.expander("📚 Resources"):
        st.markdown("""
- [CDC — Sleep & Sleep Disorders](https://www.cdc.gov/sleep)
- [National Sleep Foundation](https://www.thensf.org)
- [Sleep Health Journal](https://www.sleephealthjournal.org)
- [Why We Sleep — Matthew Walker](https://www.amazon.com/dp/1501144316)
- [Kaggle Dataset](https://www.kaggle.com/datasets/adilshamim8/sleep-cycle-and-productivity/discussion?sort=hotness)
        """)

# ── Main App ──────────────────────────────────────────────────────────────────
st.title("🌙 Sleep Duration Predictor")
st.markdown("Fill in your details below to predict how many hours of sleep you need.")
st.divider()

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

if st.button("🔮 Predict Sleep Duration", use_container_width=True, type="primary"):
    if not model_loaded:
        st.error(
            "⚠️ Model files not found! Make sure `sleep_model.pkl`, "
            "`sleep_scaler.pkl`, and `sleep_columns.pkl` are in the same folder."
        )
    else:
        input_data = pd.DataFrame([{col: 0 for col in columns}])
        input_data["Age"] = age
        input_data["Quality of Sleep"] = quality_of_sleep
        input_data["Physical Activity Level"] = physical_activity
        input_data["Stress Level"] = stress_level
        input_data["Heart Rate"] = heart_rate
        input_data["Daily Steps"] = daily_steps
        input_data["BP_Systolic"] = bp_systolic
        input_data["BP_Diastolic"] = bp_diastolic

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

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = round(float(prediction), 2)
        hours = int(prediction)
        minutes = int((prediction - hours) * 60)

        st.success(f"### 🛏️ Predicted Sleep Duration: **{prediction} hours** ({hours}h {minutes}m)")

        pct = min(prediction / 10, 1.0)
        if prediction < 6:
            color, label = "#E74C3C", "⚠️ Below recommended (< 6 hrs)"
        elif prediction <= 9:
            color, label = "#2ECC71", "✅ Within healthy range (6–9 hrs)"
        else:
            color, label = "#F39C12", "⚠️ Above average (> 9 hrs)"

        st.markdown(f"""
        <div style='background:#eee;border-radius:10px;height:20px;width:100%;margin-bottom:6px'>
            <div style='background:{color};width:{pct*100:.0f}%;height:20px;border-radius:10px'></div>
        </div>
        <p style='color:{color};font-weight:bold'>{label}</p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Based on a Linear Regression model trained on the Sleep Health and Lifestyle Dataset.")

st.markdown(
    "<br><center><small>Lab 5.0 — Artificial Intelligence | Sleep Duration Predictor</small></center>",
    unsafe_allow_html=True
)
