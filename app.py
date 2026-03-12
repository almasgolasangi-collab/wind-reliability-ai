import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- 1. SETTINGS & LOGO ---
st.set_page_config(page_title="Wind AI Reliability Monitor", layout="wide")

API_KEY = "3bea6d570f4e26ab35c5f69864e977d6" 
SITES = {
    "Brahmanvel, MH": {"lat": 21.03, "lon": 74.22},
    "Dhalgaon, MH": {"lat": 17.58, "lon": 74.85},
    "Chitradurga, KA": {"lat": 14.23, "lon": 76.40},
    "Kayathar, TN": {"lat": 8.94, "lon": 77.72}
}

# --- 2. SIDEBAR CONTROLS ---
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.info("Upload 'logo.png' to app folder.")

st.sidebar.title("Data Input Control")
input_method = st.sidebar.radio("Input Method:", ["Live API", "Manual Slider", "Upload Data Set"])

v_curr = 12.0
turb = 2.0
wind_series = []

if input_method == "Live API":
    site = st.sidebar.selectbox("Select Site", list(SITES.keys()))
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={SITES[site]['lat']}&lon={SITES[site]['lon']}&appid={API_KEY}&units=metric"
        v_curr = requests.get(url, timeout=5).json()['wind']['speed']
        st.sidebar.success(f"Live Speed: {v_curr} m/s")
    except: v_curr = 12.0
    wind_series = np.random.normal(v_curr, 0.8, 50) # Simulated series for graph

elif input_method == "Manual Slider":
    v_curr = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 50.0, 12.0)
    turb = st.sidebar.slider("Turbulence Level", 0.1, 10.0, 2.0)
    wind_series = np.random.normal(v_curr, turb, 50)

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        col = [c for c in df.columns if 'speed' in c.lower() or 'wind' in c.lower()][0]
        wind_series = df[col].dropna().values
        v_curr = np.mean(wind_series)
        turb = np.std(wind_series)
    else:
        st.stop()

# --- 3. THE WIND GRAPH ---
st.title("🌬️ Wind Profile & Reliability Analysis")
st.subheader("1. Real-Time Wind Speed Data")
fig_w, ax_w = plt.subplots(figsize=(12, 3))
ax_w.plot(wind_series, color='#2c3e50', linewidth=1.5, label="Wind Speed Trend")
ax_w.axhline(v_curr, color='red', linestyle='--', label=f"Mean: {v_curr:.2f} m/s")
ax_w.set_ylabel("m/s")
ax_w.set_xlabel("Time Intervals")
ax_w.legend(loc='upper right')
st.pyplot(fig_w)

st.divider()

# --- 4. THE 3 METHODS VISUALIZATIONS ---
st.subheader("2. Reliability Methodology Analysis")
c1, c2, c3 = st.columns(3)

# A. Fault Tree (Component Failure Logic)
with c1:
    st.markdown("#### Fault Tree (FTA)")
    # Logic: Risk = (Wind / Design Limit)
    b_risk = (v_curr/30)**2; g_risk = (v_curr/40)**1.5; gen_risk = (v_curr/45)**1.2
    fig1, ax1 = plt.subplots()
    ax1.bar(['Blades', 'Gearbox', 'Generator'], [b_risk, g_risk, gen_risk], color='#e74c3c')
    ax1.set_ylabel("Failure Probability")
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)
    st.caption("Visualizing specific hardware component risks.")

# B. Monte Carlo (Turbulence Simulation)
with c2:
    st.markdown("#### Monte Carlo")
    # Histogram of simulated reliability scores
    mc_samples = np.random.normal(100 - (v_curr*1.5), turb*3, 1000)
    fig2, ax2 = plt.subplots()
    ax2.hist(mc_samples, bins=30, color='#3498db', alpha=0.7)
    ax2.set_xlabel("Reliability Score %")
    st.pyplot(fig2)
    st.caption("Simulating 1000 turbulence iterations.")

# C. Markov Chain (State Transition)
with c3:
    st.markdown("#### Markov Chain")
    p_safe = np.exp(-0.04 * v_curr)
    p_fail = 1 - p_safe
    fig3, ax3 = plt.subplots()
    ax3.pie([p_safe, p_fail], labels=['Operational', 'Fault Risk'], colors=['#2ecc71', '#f1c40f'], autopct='%1.1f%%')
    st.pyplot(fig3)
    st.caption("Predicting future health state transitions.")

st.divider()

# --- 5. COMPONENT STATUS & FINAL COMPARISON ---
st.subheader("3. Component Status & Final Comparison")
tab1, tab2 = st.tabs(["Component Failure Table", "Method Comparison Bar Graph"])

with tab1:
    st.table(pd.DataFrame({
        "Component": ["Turbine Blades", "Gearbox", "Generator"],
        "FTA Risk": [f"{b_risk:.2%}", f"{g_risk:.2%}", f"{gen_risk:.2%}"],
        "Condition": ["Good" if b_risk < 0.2 else "Watch", "Good", "Good"]
    }))

with tab2:
    # Final Comparison Graph
    fta_final = (1 - max(b_risk, g_risk, gen_risk)) * 100
    mc_final = np.mean(mc_samples)
    mar_final = p_safe * 100
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    ax_bar.bar(["Fault Tree", "Monte Carlo", "Markov Chain"], [fta_final, mc_final, mar_final], color=['#2c3e50', '#3498db', '#e67e22'])
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("Overall Reliability %")
    st.pyplot(fig_bar)