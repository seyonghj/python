import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import random

st.set_page_config(
    page_title="Sleep Duration Predictor",
    page_icon="🌙",
    layout="wide"
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

# ── Main Inputs ───────────────────────────────────────────────────────────────
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

# ── Heart Rate Insight ────────────────────────────────────────────────────────
st.divider()
st.subheader("❤️ Heart Rate Insight")

hr_col1, hr_col2, hr_col3 = st.columns(3)

with hr_col1:
    st.metric("Your Heart Rate", f"{heart_rate} bpm")

with hr_col2:
    avg_resting = 72
    diff = heart_rate - avg_resting
    st.metric("vs. Average (72 bpm)", f"{heart_rate} bpm", delta=f"{diff:+d} bpm")

with hr_col3:
    if heart_rate < 60:
        zone, zone_icon = "Low (Bradycardia)", "🔵"
    elif heart_rate <= 72:
        zone, zone_icon = "Normal", "🟢"
    elif heart_rate <= 85:
        zone, zone_icon = "Elevated", "🟡"
    else:
        zone, zone_icon = "High", "🔴"
    st.metric("Zone", f"{zone_icon} {zone}")

pct_hr = min(heart_rate / 120, 1.0)
if heart_rate < 60:
    bar_color_hr = "#3498DB"
elif heart_rate <= 72:
    bar_color_hr = "#2ECC71"
elif heart_rate <= 85:
    bar_color_hr = "#F39C12"
else:
    bar_color_hr = "#E74C3C"

st.markdown(f"""
<div style='background:#eee;border-radius:10px;height:16px;width:100%;margin-bottom:4px'>
    <div style='background:{bar_color_hr};width:{pct_hr*100:.0f}%;height:16px;border-radius:10px'></div>
</div>
<small style='color:gray'>60 bpm — Normal &nbsp;|&nbsp; 72 bpm — Average &nbsp;|&nbsp; 90+ bpm — High</small>
""", unsafe_allow_html=True)

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Sleep Duration", use_container_width=True, type="primary"):
    if not model_loaded:
        st.error(
            "⚠️ Model files not found! Make sure `sleep_model.pkl`, "
            "`sleep_scaler.pkl`, and `sleep_columns.pkl` are in the same folder as this app."
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

        # ── Result Header ─────────────────────────────────────────────────────
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

        st.divider()

        # ── Summary Metrics ───────────────────────────────────────────────────
        st.subheader("📊 Your Health Snapshot")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Sleep", f"{prediction} hrs")
        m2.metric("Stress Level", f"{stress_level}/10",
                  delta="High" if stress_level >= 7 else ("Moderate" if stress_level >= 4 else "Low"),
                  delta_color="inverse")
        m3.metric("Physical Activity", f"{physical_activity} min/day",
                  delta="Good" if physical_activity >= 30 else "Low",
                  delta_color="normal" if physical_activity >= 30 else "inverse")
        m4.metric("Heart Rate", f"{heart_rate} bpm",
                  delta="Elevated" if heart_rate > 80 else "Normal",
                  delta_color="inverse" if heart_rate > 80 else "normal")

        st.divider()

        # ── Personalized Suggestions ──────────────────────────────────────────
        st.subheader("💡 Personalized Suggestions")

        suggestions = []

        if prediction < 6:
            suggestions.append(("🔴 Critical Sleep", "Your predicted sleep is critically low. Prioritize getting at least 7–9 hours to avoid serious health risks like heart disease and impaired cognition."))
        elif prediction < 7:
            suggestions.append(("🟠 Sleep More", "You're slightly under the recommended 7–9 hours. Try going to bed 30–60 minutes earlier each night."))
        elif prediction > 9:
            suggestions.append(("🟡 Oversleeping", "Sleeping over 9 hours regularly may indicate an underlying issue. Consider speaking to a doctor if this is consistent."))
        else:
            suggestions.append(("🟢 Great Sleep!", "Your predicted sleep duration is within the healthy 7–9 hour range. Keep up your current habits!"))

        if stress_level >= 8:
            suggestions.append(("😰 High Stress", "Your stress level is very high. Try mindfulness meditation, journaling, or speaking to a counselor. High stress significantly reduces sleep quality."))
        elif stress_level >= 5:
            suggestions.append(("😐 Moderate Stress", "Consider adding a relaxing wind-down routine before bed — light stretching, reading, or deep breathing can help lower stress."))

        if physical_activity < 20:
            suggestions.append(("🏃 Move More", "You're getting very little physical activity. Even a 20–30 min walk daily can significantly improve sleep quality and duration."))
        elif physical_activity > 60:
            suggestions.append(("💪 Stay Consistent", "Great activity level! Just avoid intense exercise within 2 hours of bedtime as it can delay sleep onset."))

        if heart_rate > 85:
            suggestions.append(("❤️ High Heart Rate", "Your resting heart rate is elevated. Regular aerobic exercise, hydration, and stress reduction can help bring it down over time."))
        elif heart_rate < 60:
            suggestions.append(("💙 Low Heart Rate", "A low resting heart rate can be normal for athletes, but if you feel dizzy or fatigued, consult a doctor."))

        if bmi in ["Overweight", "Obese"]:
            suggestions.append(("⚖️ Weight & Sleep", "Higher BMI is linked to sleep apnea and poor sleep quality. A balanced diet and regular activity can improve both weight and sleep."))

        if sleep_disorder == "Insomnia":
            suggestions.append(("😶 Insomnia Tips", "Stick to a strict sleep schedule, avoid naps after 3 PM, limit caffeine after noon, and consider Cognitive Behavioral Therapy for Insomnia (CBT-I)."))
        elif sleep_disorder == "Sleep Apnea":
            suggestions.append(("😮‍💨 Sleep Apnea", "Make sure you're using your CPAP device if prescribed. Sleep on your side, avoid alcohol before bed, and maintain a healthy weight."))

        if daily_steps < 5000:
            suggestions.append(("👟 Step It Up", "You're averaging fewer than 5,000 steps. Aim for 7,000–10,000 daily steps to boost energy, mood, and sleep quality."))

        if bp_systolic >= 130 or bp_diastolic >= 85:
            suggestions.append(("🩺 Blood Pressure", "Your blood pressure readings are on the higher side. Reduce sodium intake, manage stress, and consult your doctor if this persists."))

        if quality_of_sleep <= 4:
            suggestions.append(("😴 Poor Sleep Quality", "Your sleep quality is low. Try blackout curtains, white noise, and keeping your room cool (60–67°F / 15–19°C) for better rest."))

        if age >= 50:
            suggestions.append(("🧓 Age & Sleep", "As we age, sleep becomes lighter and more fragmented. Avoid long naps, stay active, and maintain a consistent bedtime routine."))

        for badge, text in suggestions:
            st.markdown(f"""
            <div style='background:#f8f9fa;border-left:4px solid #4A90D9;border-radius:8px;
                        padding:12px 16px;margin-bottom:10px;'>
                <strong>{badge}</strong><br>
                <span style='font-size:14px;color:#444;'>{text}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Recommended Bedtime ───────────────────────────────────────────────
        st.subheader("⏰ Recommended Bedtime")
        st.markdown("Based on your predicted sleep duration, here are ideal bedtimes to wake up feeling refreshed:")

        wake_options = ["6:00 AM", "6:30 AM", "7:00 AM", "7:30 AM", "8:00 AM"]
        bt_cols = st.columns(len(wake_options))
        for i, wake in enumerate(wake_options):
            wake_ts = pd.Timestamp(f"2024-01-01 {wake}")
            sleep_ts = wake_ts - pd.Timedelta(hours=prediction)
            bedtime_str = sleep_ts.strftime("%I:%M %p")
            bt_cols[i].markdown(f"""
            <div style='text-align:center;background:#eaf4fb;border-radius:10px;padding:10px;'>
                <div style='font-size:12px;color:#666;'>Wake at</div>
                <div style='font-weight:bold;font-size:15px;'>{wake}</div>
                <div style='font-size:12px;color:#666;'>Sleep by</div>
                <div style='font-weight:bold;color:#1a6fa8;font-size:15px;'>{bedtime_str}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.caption("Based on a Linear Regression model trained on the Sleep Health and Lifestyle Dataset.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌙 Sleep Tools")
    st.divider()

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
    random.seed(pd.Timestamp.now().day_of_year)
    daily_tips = random.sample(all_tips, 3)
    for tip in daily_tips:
        st.markdown(f"- {tip}")

    st.button("🔄 New Tips", key="refresh_tips")

    st.divider()

    st.subheader("📊 Sleep Health Score")
    score_hours = st.slider("Your usual sleep (hrs)", 4.0, 12.0, 7.0, 0.5, key="score_hrs")
    score_quality = st.slider("Perceived quality (1–10)", 1, 10, 7, key="score_q")
    score_consistent = st.checkbox("Consistent schedule?", value=True)
    score_caffeine = st.checkbox("No caffeine after 2 PM?", value=True)
    score_screens = st.checkbox("Screen-free 30 min before bed?", value=False)

    hour_score = max(0, 100 - abs(score_hours - 7.5) * 15)
    quality_score = score_quality * 10
    bonus = (10 if score_consistent else 0) + \
            (5 if score_caffeine else 0) + \
            (5 if score_screens else 0)
    raw_score = (hour_score * 0.4 + quality_score * 0.4 + bonus * 1.0)
    sleep_score = min(100, int(raw_score))

    if sleep_score >= 80:
        score_label = "Excellent 🌟"
    elif sleep_score >= 60:
        score_label = "Good 👍"
    elif sleep_score >= 40:
        score_label = "Fair ⚠️"
    else:
        score_label = "Poor 😴"

    st.metric("Your Sleep Score", f"{sleep_score}/100", score_label)
    st.progress(sleep_score / 100)

    st.divider()

    st.subheader("⏰ Sleep Cycle Planner")
    st.caption("A full sleep cycle is ~90 minutes. Plan your wake time to complete full cycles.")
    bedtime = st.time_input("Planned bedtime", value=pd.Timestamp("22:00").time(), key="bedtime")
    cycles = st.radio("Number of cycles", [4, 5, 6], horizontal=True, index=1, key="cycles")
    fall_asleep_mins = st.slider("Minutes to fall asleep", 5, 30, 14, key="fam")

    total_mins = cycles * 90 + fall_asleep_mins
    wake_time = (pd.Timestamp.combine(pd.Timestamp.today(), bedtime)
                 + pd.Timedelta(minutes=total_mins)).time()
    st.success(f"⏰ Wake up at **{wake_time.strftime('%I:%M %p')}** ({cycles} cycles = {cycles * 1.5:.1f} hrs)")

    st.divider()

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

    with st.expander("📚 Resources"):
        st.markdown("""
- [CDC — Sleep & Sleep Disorders](https://www.cdc.gov/sleep)
- [National Sleep Foundation](https://www.thensf.org)
- [Sleep Health Journal](https://www.sleephealthjournal.org)
- [Why We Sleep — Matthew Walker](https://www.amazon.com/dp/1501144316)
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center><small>Lab 5.0 — Artificial Intelligence | Sleep Duration Predictor</small></center>",
    unsafe_allow_html=True
)
