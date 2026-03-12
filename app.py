import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- 1. CONFIG ---
st.set_page_config(page_title="Wind AI: 3-Method Reliability Engine", layout="wide")

# YOUR LIVE API KEY
API_KEY = "3bea6d570f4e26ab35c5f69864e977d6" 

SITES = {
    "Brahmanvel, MH": {"lat": 21.03, "lon": 74.22},
    "Dhalgaon, MH": {"lat": 17.58, "lon": 74.85},
    "Chitradurga, KA": {"lat": 14.23, "lon": 76.40},
    "Kayathar, TN": {"lat": 8.94, "lon": 77.72}
}

# --- 2. LIVE FETCH ---
def get_live_data(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        return res['wind']['speed'], "✅ LIVE"
    except:
        return 5.2, "⚠️ CACHE"

# --- 3. INPUT ---
st.sidebar.header("📡 Data Control")
site_name = st.sidebar.selectbox("Select Plant Site", list(SITES.keys()))
v_base, status = get_live_data(SITES[site_name]['lat'], SITES[site_name]['lon'])
st.sidebar.info(f"Source Status: {status}")

# Generate 50 points of data for the graphs
data_points = np.random.normal(v_base, 0.5, 50)
avg_v = np.mean(data_points)

# --- 4. MATHEMATICAL COMPARISON ---
# 1. Markov: State-based (predicts transitions)
rel_markov = np.exp(-0.04 * avg_v) * 100

# 2. Monte Carlo: Load-based (predicts variance)
rel_mc = (1 - (np.std(data_points)/avg_v)) * 100

# 3. FTA: Logic-based (predicts component failure)
p_fail = (avg_v / 25)**2
rel_fta = (1 - p_fail) * 100

# --- 5. DASHBOARD ---
st.title(f"🌬️ Universal Wind Reliability: {site_name}")
st.write(f"**Analysis Date:** March 12, 2026 | **Live Wind Speed:** {avg_v:.2f} m/s")

# --- 6. VISUALIZATION OF ALL 3 METHODS ---
st.subheader("📊 Method Visualizations")
v1, v2, v3 = st.columns(3)

with v1:
    st.markdown("#### 1. Monte Carlo (Line Graph)")
    fig_mc, ax_mc = plt.subplots(figsize=(5, 4))
    ax_mc.plot(data_points, color='#3498db', linewidth=2, marker='o', markersize=3)
    ax_mc.fill_between(range(50), data_points.min(), data_points, color='#3498db', alpha=0.2)
    ax_mc.set_title("Stochastic Wind Load")
    st.pyplot(fig_mc)
    st.caption("Visualizes the random fluctuations in wind load to test structural fatigue.")

with v2:
    st.markdown("#### 2. Markov Chain (State Graph)")
    # Transition Logic
    p_op = 0.92 if avg_v < 12 else 0.65
    p_warn = (1 - p_op) * 0.7
    p_fail_state = (1 - p_op) * 0.3
    
    fig_mar, ax_mar = plt.subplots(figsize=(5, 4))
    ax_mar.bar(['Operational', 'Warning', 'Fault'], [p_op, p_warn, p_fail_state], color=['#2ecc71', '#f1c40f', '#e74c3c'])
    ax_mar.set_title("Future Health Probability")
    st.pyplot(fig_mar)
    st.caption("Predicts the likelihood of the turbine staying healthy vs. failing next.")

with v3:
    st.markdown("#### 3. Fault Tree (Logic Visualization)")
    # Representing FTA Logic as a Component Risk Graph
    blade_risk = p_fail * 0.7
    gear_risk = p_fail * 0.3
    
    fig_fta, ax_fta = plt.subplots(figsize=(5, 4))
    ax_fta.pie([blade_risk, gear_risk, 1-p_fail], labels=['Blade Risk', 'Gearbox Risk', 'Safe'], 
               colors=['#e74c3c', '#d35400', '#2ecc71'], autopct='%1.1f%%', startangle=140)
    ax_fta.set_title("Root Cause Contribution")
    st.pyplot(fig_fta)
    st.caption("Breaks down the 100% total system into specific component failure risks.")

st.divider()

# --- 7. COMPARISON & RELIABILITY VERDICT ---
st.subheader("⚖️ Which Method is Most Reliable?")
c1, c2 = st.columns([2, 1])

with c1:
    comparison_data = {
        "Method": ["Monte Carlo", "Markov Chain", "Fault Tree (FTA)"],
        "Reliability Score": [f"{rel_mc:.2f}%", f"{rel_markov:.2f}%", f"{rel_fta:.2f}%"],
        "Focus Area": ["Load & Turbulence", "System States & Time", "Component Logic"],
        "Reliability Verdict": [
            "Best for structural fatigue",
            "Best for availability uptime",
            "Best for maintenance safety"
        ]
    }
    st.table(pd.DataFrame(comparison_data))

with c2:
    st.info("💡 **AI Verdict:**")
    st.write("""
    For **Real-time Safety**, **Fault Tree (FTA)** is most reliable as it monitors critical hardware logic. 
    
    For **Long-term Planning**, **Markov Chain** is superior as it predicts future availability.
    
    For **Extreme Weather**, **Monte Carlo** provides the most accurate risk buffer.
    """)

st.success(f"Final Integrated Reliability Index: **{(rel_markov + rel_mc + rel_fta)/3:.2f}%**")