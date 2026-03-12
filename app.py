import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- 1. CONFIG ---
st.set_page_config(page_title="IEC 61400 Reliability AI", layout="wide")

API_KEY = "3bea6d570f4e26ab35c5f69864e977d6" 

SITES = {
    "Brahmanvel, MH": {"lat": 21.03, "lon": 74.22},
    "Dhalgaon, MH": {"lat": 17.58, "lon": 74.85},
    "Chitradurga, KA": {"lat": 14.23, "lon": 76.40},
    "Kayathar, TN": {"lat": 8.94, "lon": 77.72}
}

# --- 2. LIVE FETCH ---
def fetch_live_wind(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        return res['wind']['speed'], "✅ LIVE API ACTIVE"
    except:
        return 4.5, "⚠️ FALLBACK MODE"

# --- 3. INPUT ---
st.sidebar.header("📡 Data Control")
mode = st.sidebar.radio("Source:", ["Live API", "Excel Upload"])

if mode == "Excel Upload":
    file = st.sidebar.file_uploader("Upload SCADA Data", type=["xlsx", "csv"])
    if not file: st.stop()
    df_up = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
    col = [c for c in df_up.columns if 'speed' in c.lower() or 'wind' in c.lower()][0]
    data_points = df_up[col].dropna().values
    site_name = "Uploaded Dataset"
else:
    site_name = st.sidebar.selectbox("Select Plant:", list(SITES.keys()))
    v_base, status = fetch_live_wind(SITES[site_name]['lat'], SITES[site_name]['lon'])
    st.sidebar.info(status)
    # Monte Carlo simulation (Line data)
    data_points = np.random.normal(v_base, 0.5, 50)

# --- 4. RELIABILITY MATH ---
avg_v = np.mean(data_points)
rel_markov = np.exp(-0.04 * avg_v) * 100
rel_mc = (1 - (np.std(data_points)/avg_v)) * 100
p_fail = (avg_v / 25)**2
rel_fta = (1 - p_fail) * 100

# --- 5. DASHBOARD HEADER & COMPARISON ---
st.title(f"🌬️ Reliability Analysis: {site_name}")
st.subheader("📊 3-Method Comparison Matrix")
m1, m2, m3 = st.columns(3)
m1.metric("Markov Chain", f"{rel_markov:.2f}%", help="Predictive state-based reliability")
m2.metric("Monte Carlo", f"{rel_mc:.2f}%", help="Probabilistic load-based reliability")
m3.metric("Fault Tree (FTA)", f"{rel_fta:.2f}%", help="Logic-based system safety")

st.divider()

# --- 6. VISUALIZATIONS ---

# ROW 1: MONTE CARLO (LINE GRAPH)
st.subheader("🎲 Monte Carlo: Stochastic Load Simulation")
fig_mc, ax_mc = plt.subplots(figsize=(12, 4))
ax_mc.plot(data_points, marker='o', linestyle='-', color='#3498db', label="Simulated Gusts")
ax_mc.axhline(avg_v, color='red', linestyle='--', label=f"Mean Speed: {avg_v:.2f} m/s")
ax_mc.fill_between(range(len(data_points)), avg_v - np.std(data_points), avg_v + np.std(data_points), color='skyblue', alpha=0.3, label="Turbulence Band")
ax_mc.set_ylabel("Wind Speed (m/s)")
ax_mc.legend()
st.pyplot(fig_mc)

st.divider()

# ROW 2: MARKOV & FTA
col1, col2 = st.columns(2)

with col1:
    st.subheader("⛓️ Markov Chain: Health States")
    # Transition probabilities based on live data
    p_op = 0.90 if avg_v < 12 else 0.60
    p_warn = (1 - p_op) * 0.7
    p_fail_state = (1 - p_op) * 0.3
    
    fig_mar, ax_mar = plt.subplots(figsize=(7, 5))
    ax_mar.bar(['Operational', 'Warning', 'Shutdown'], [p_op, p_warn, p_fail_state], color=['#2ecc71', '#f1c40f', '#e74c3c'])
    ax_mar.set_ylabel("Probability of Transition")
    st.pyplot(fig_mar)

with col2:
    st.subheader("🌲 Fault Tree: Logical Risk")
    st.markdown(f"""
    <div style="background:#f8f9fa; padding:30px; border-radius:15px; border-left: 10px solid #c0392b; height: 350px;">
        <strong style="font-size:20px;">TOP EVENT: CRITICAL SYSTEM TRIP</strong><br>
        <span style="color:red; font-size:28px; font-weight:bold;">{p_fail:.4%} Prob.</span><br><br>
        <b>Failure Pathways (OR GATE Logic):</b><br>
        1. Blade Structural Stress: {(p_fail*0.7):.4%}<br>
        2. Gearbox Wear (Drivetrain): {(p_fail*0.3):.4%}<br><br>
        <i>Note: FTA calculates the logic of failure events, whereas Monte Carlo simulates the physical wind loads.</i>
    </div>
    """, unsafe_allow_html=True)

# --- 7. COMPARISON TABLE ---
st.divider()
st.subheader("📋 Methodology Breakdown")
st.table(pd.DataFrame({
    "Method": ["Monte Carlo", "Markov Chain", "Fault Tree"],
    "Visualization Used": ["Line Trend Graph", "Probability Bar Chart", "Logic Diagram"],
    "Analysis Value": ["Load Variance", "Future State", "Component Failure"],
    "Result Index": [f"{rel_mc:.2f}%", f"{rel_markov:.2f}%", f"{rel_fta:.2f}%"]
}))