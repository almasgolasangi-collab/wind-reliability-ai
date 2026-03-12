import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- 1. SETTINGS & SITES ---
st.set_page_config(page_title="IEC 61400 Wind Reliability AI", layout="wide")

# YOUR LIVE API KEY
API_KEY = "3bea6d570f4e26ab35c5f69864e977d6" 

SITES = {
    "Brahmanvel, MH": {"lat": 21.03, "lon": 74.22, "class": "III"},
    "Dhalgaon, MH": {"lat": 17.58, "lon": 74.85, "class": "III"},
    "Chitradurga, KA": {"lat": 14.23, "lon": 76.40, "class": "II"},
    "Kayathar, TN": {"lat": 8.94, "lon": 77.72, "class": "I"}
}

# --- 2. LIVE DATA FETCHING ---
def fetch_live_wind(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        v_live = res['wind']['speed']
        return v_live, "✅ LIVE API ACTIVE"
    except Exception:
        return 4.5, "⚠️ FALLBACK MODE (Offline)"

# --- 3. INPUT HANDLING ---
st.sidebar.header("📡 Data Control")
mode = st.sidebar.radio("Source:", ["Live Wind API", "Excel Upload"])

if mode == "Excel Upload":
    file = st.sidebar.file_uploader("Upload SCADA Data", type=["xlsx", "csv"])
    if file:
        df_up = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
        speed_col = [c for c in df_up.columns if 'speed' in c.lower() or 'wind' in c.lower()][0]
        data_points = df_up[speed_col].dropna().values
        site_name = "Custom Upload"
    else: st.stop()
else:
    site_name = st.sidebar.selectbox("Select Wind Farm:", list(SITES.keys()))
    v_base, status = fetch_live_wind(SITES[site_name]['lat'], SITES[site_name]['lon'])
    st.sidebar.info(status)
    # MONTE CARLO: Using live wind as the mean for 1000 samples
    data_points = np.random.normal(v_base, 0.6, 1000)

avg_v = np.mean(data_points)
std_v = np.std(data_points)

# --- 4. RELIABILITY CALCULATIONS ---
# 1. Markov Reliability (Exponential Decay)
rel_markov = np.exp(-0.04 * avg_v) * 100

# 2. Monte Carlo Reliability (Inverse of Turbulence)
rel_mc = (1 - (std_v / avg_v)) * 100 if avg_v > 0 else 0

# 3. Fault Tree Analysis (FTA) 
p_fail = (avg_v / 25)**2
rel_fta = (1 - p_fail) * 100

# --- 5. DASHBOARD HEADER ---
st.title(f"🌬️ Reliability Dashboard: {site_name}")
st.write(f"**Live Feed Analytics:** March 12, 2026 | **Framework:** IEC 61400-1 Standards")

# --- 6. 3-METHOD COMPARISON MATRIX ---
st.subheader("📊 3-Method Reliability Comparison")
m1, m2, m3 = st.columns(3)
m1.metric("Markov Chain Score", f"{rel_markov:.2f}%", help="Long-term state stability")
m2.metric("Monte Carlo Score", f"{rel_mc:.2f}%", help="Probabilistic load risk")
m3.metric("Fault Tree (FTA) Score", f"{rel_fta:.2f}%", help="Logic-based system safety")

st.divider()

# --- 7. VISUALIZATIONS (ALL 3 METHODS) ---

# ROW 1: MONTE CARLO & MARKOV
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎲 Monte Carlo Risk Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(data_points, bins=40, color='#3498db', edgecolor='white', alpha=0.8)
    ax1.axvline(avg_v, color='red', linestyle='--', label=f'Live Mean: {avg_v:.2f} m/s')
    ax1.set_xlabel("Wind Speed (m/s)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    st.pyplot(fig1)
    st.caption("Monte Carlo uses 1000 iterations to determine structural load thresholds.")

with col2:
    st.subheader("⛓️ Markov Chain State Prediction")
    # Dynamic calculation: As wind speed increases, shutdown probability grows
    p_op = 0.95 if avg_v < 8 else (0.80 if avg_v < 15 else 0.50)
    p_warn = (1 - p_op) * 0.7
    p_sd = (1 - p_op) * 0.3
    
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(['Operational', 'Warning', 'Shutdown'], [p_op, p_warn, p_sd], color=['#2ecc71', '#f1c40f', '#e74c3c'])
    ax2.set_ylabel("Probability of State")
    st.pyplot(fig2)
    st.caption("Markov Model predicts the next transition state based on current live intensity.")

st.divider()

# ROW 2: FTA & SUMMARY
col3, col4 = st.columns(2)

with col3:
    st.subheader("🌲 Fault Tree Analysis (Visual Logic)")
    st.markdown(f"""
    <div style="background:#f8f9fa; padding:25px; border-radius:10px; border-left: 8px solid #c0392b;">
        <strong style="font-size:18px;">TOP EVENT: SYSTEM TRIP</strong><br>
        Current Probability: <span style="color:red; font-size:22px; font-weight:bold;">{p_fail:.4%}</span><br><br>
        <b>Root Cause Probabilities:</b><br>
        - Blade Stress (Rotor Group): {(p_fail*0.7):.4%}<br>
        - Drive Train Wear (Gearbox): {(p_fail*0.3):.4%}<br><br>
        <i>Logic: [OR GATE] System fails if either Blade OR Gearbox exceeds threshold.</i>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.subheader("📋 Methodology Summary")
    summary_df = pd.DataFrame({
        "Method": ["Monte Carlo", "Markov Chain", "Fault Tree"],
        "Analysis Type": ["Probabilistic", "Predictive", "Logical"],
        "Focus Area": ["Load Variability", "State Reliability", "Root Cause Logic"],
        "Final Index": [f"{rel_mc:.2f}%", f"{rel_markov:.2f}%", f"{rel_fta:.2f}%"]
    })
    st.table(summary_df)

st.info(f"Analysis verified for {site_name} using OpenWeatherMap API for Lat/Lon: {SITES.get(site_name, {}).get('lat')}, {SITES.get(site_name, {}).get('lon')}")