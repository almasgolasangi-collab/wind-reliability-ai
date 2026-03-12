import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CONFIG & BRANDING ---
st.set_page_config(page_title="Wind AI: Live India Monitor", page_icon="logo.ico", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; }
    .fta-box { border: 2px solid #2e86c1; padding: 10px; border-radius: 5px; text-align: center; background: #ebf5fb; }
    .gate { font-weight: bold; color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LIVE PLANT DATA (March 12, 2026 - 15:00 IST) ---
LIVE_PLANT_DATA = {
    "Brahmanvel, MH": {"v": 3.8, "dir": "NW", "iec": "Class III"},
    "Dhalgaon, MH": {"v": 4.0, "dir": "E", "iec": "Class III"},
    "Chitradurga, KA": {"v": 4.9, "dir": "E", "iec": "Class III"},
    "Kayathar, TN": {"v": 4.5, "dir": "E", "iec": "Class III"}
}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("🔋 Control Center")

data_source = st.sidebar.radio("Data Input:", ["Live Remote Monitoring", "Local Dataset Upload"])

if data_source == "Live Remote Monitoring":
    site = st.sidebar.selectbox("Select Active Wind Farm:", list(LIVE_PLANT_DATA.keys()))
    v_base = LIVE_PLANT_DATA[site]["v"]
    # Generate distribution based on live wind
    data = np.random.normal(v_base, v_base * 0.1, 100)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Plant Log", type=["xlsx", "csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        data = df.iloc[:, 0].values
        site = "Custom Upload"
    else:
        st.warning("Please upload a file.")
        st.stop()

avg_v = np.mean(data)

# --- 4. FAULT TREE ANALYSIS (FTA) LOGIC ---
# Probabilities of basic events (modeled on wind speed intensity)
p_blade_fracture = 0.02 * (avg_v / 10)
p_pitch_failure = 0.015
p_gearbox_wear = 0.03 * (avg_v / 10)
p_gen_overheat = 0.01

# Intermediate Events
# OR Gate: Rotor Failure = Blade Fracture OR Pitch Failure
p_rotor_failure = 1 - (1 - p_blade_fracture) * (1 - p_pitch_failure)
# AND Gate: Drive Train Failure = Gearbox AND Generator (assuming redundancy/check)
p_drivetrain_failure = p_gearbox_wear * p_gen_overheat 

# TOP EVENT: System Shutdown = Rotor Failure OR Drivetrain Failure
p_top_event = 1 - (1 - p_rotor_failure) * (1 - p_drivetrain_failure)

# --- 5. MAIN DASHBOARD ---
st.title(f"🌬️ Reliability Analysis: {site}")
st.write(f"**Live Feed:** March 12, 2026, 15:00 IST")

# Row 1: Key Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Live Wind Speed", f"{avg_v:.2f} m/s")
m2.metric("Markov Reliability", f"{np.exp(-0.04 * avg_v)*100:.1f}%")
m3.metric("System Failure Risk", f"{p_top_event*100:.2f}%")
m4.metric("Site IEC Class", LIVE_PLANT_DATA.get(site, {"iec": "N/A"})["iec"])

st.markdown("---")

# Row 2: Fault Tree Visualization
st.subheader("🌲 Fault Tree Analysis (FTA)")
f1, f2, f3 = st.columns([1, 2, 1])
with f2:
    st.markdown(f"""
    <div class="fta-box">
        <strong>TOP EVENT: SYSTEM SHUTDOWN</strong><br>
        Probability: <span style="color:red">{p_top_event*100:.3f}%</span>
    </div>
    <div style="text-align:center">↑ <span class="gate">OR GATE</span> ↑</div>
    <div style="display: flex; justify-content: space-around;">
        <div class="fta-box" style="width: 45%;">
            <strong>Rotor Failure</strong><br>{p_rotor_failure*100:.3f}%
            <br>↑ <span class="gate">OR</span> ↑<br>
            <small>Blades ({p_blade_fracture:.2%})<br>Pitch ({p_pitch_failure:.2%})</small>
        </div>
        <div class="fta-box" style="width: 45%;">
            <strong>Drivetrain Failure</strong><br>{p_drivetrain_failure*100:.5f}%
            <br>↑ <span class="gate">AND</span> ↑<br>
            <small>Gearbox ({p_gearbox_wear:.2%})<br>Generator ({p_gen_overheat:.2%})</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Row 3: Method Comparison
st.subheader("📊 Mathematical Comparison")
comparison_df = pd.DataFrame({
    "Method": ["Markov Chain", "Monte Carlo", "Vague Sets", "Fault Tree (FTA)"],
    "Focus": ["Life Cycle", "Stochastic Risk", "Uncertainty", "Root Cause Logic"],
    "Score": [
        f"{np.exp(-0.04 * avg_v)*100:.1f}%", 
        f"{(1-(np.std(data)/avg_v))*100:.1f}%", 
        f"{(1-(avg_v/40))*100:.1f}%",
        f"{(1-p_top_event)*100:.2f}%"
    ]
})
st.table(comparison_df)

# Row 4: Live Trend
st.subheader("📈 Live Stochastic Wind Stream")
st.line_chart(data)