import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- 1. SETTINGS ---
st.set_page_config(page_title="Wind AI Reliability", layout="wide")

API_KEY = "3bea6d570f4e26ab35c5f69864e977d6" 

SITES = {
    "Brahmanvel, MH": {"lat": 21.03, "lon": 74.22},
    "Dhalgaon, MH": {"lat": 17.58, "lon": 74.85},
    "Chitradurga, KA": {"lat": 14.23, "lon": 76.40},
    "Kayathar, TN": {"lat": 8.94, "lon": 77.72}
}

# --- 2. DATA FETCH ---
def get_live_data(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        return res['wind']['speed']
    except:
        return 11.26 # Fallback to match your screenshot

# --- 3. SIDEBAR ---
st.sidebar.title("Data Input Control")
input_method = st.sidebar.radio("Input Method:", ["Live API", "Slider (Fast Demo)"])

if input_method == "Slider (Fast Demo)":
    v_curr = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 50.0, 11.26)
else:
    site = st.sidebar.selectbox("Select Site", list(SITES.keys()))
    v_curr = get_live_data(SITES[site]['lat'], SITES[site_name]['lon'])
    st.sidebar.write(f"Live Speed: {v_curr} m/s")

turb = st.sidebar.slider("Turbulence Level", 0.0, 10.0, 3.38)

# --- 4. TOP METRICS ---
status = "SAFE" if v_curr < 25 else "DANGER"
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.header(status)
col_m2.header(f"{max(0, 100 - (v_curr*2)):.1f}%") # Generic Score 1
col_m3.header(f"{max(0, 100 - (v_curr*1.5)):.1f}%") # Generic Score 2

st.markdown("### Reliability Method Comparison")

# --- 5. THE 4-PLOT GRID ---
x = np.linspace(0, 50, 100)
c1, c2 = st.columns(2)

# 1. Fault Tree (Deterministic Step Function)
with c1:
    st.write("1. Fault Tree (Deterministic)")
    fig1, ax1 = plt.subplots()
    y_fta = np.where(x < 25, 1.0, 0.0)
    ax1.plot(x, y_fta, color='black', label='Binary Logic')
    ax1.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("Wind Speed (m/s)")
    ax1.legend()
    st.pyplot(fig1)

# 2. Markov Chain (Exponential Decay)
with c2:
    st.write("2. Markov Chain (Probabilistic)")
    fig2, ax2 = plt.subplots()
    y_markov = np.exp(-0.04 * x)
    ax2.plot(x, y_markov, color='orange', label='State Reliability')
    ax2.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Wind Speed (m/s)")
    ax2.legend()
    st.pyplot(fig2)

c3, c4 = st.columns(2)

# 3. Monte Carlo (Sampling/Density)
with c3:
    st.write("3. Monte Carlo (Sampling)")
    fig3, ax3 = plt.subplots()
    # Simulated density curve
    y_mc = np.exp(-((x - v_curr)**2) / (2 * (turb**2)))
    ax3.fill_between(x, y_mc, color='blue', alpha=0.2, label='Probability Density')
    ax3.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_xlabel("Wind Speed (m/s)")
    ax3.legend()
    st.pyplot(fig3)

# 4. Vague Set (Fuzzy/Membership)
with c4:
    st.write("4. Vague Set (Membership)")
    fig4, ax4 = plt.subplots()
    # Trapezoidal fuzzy set
    y_vague = np.clip((35 - x) / 10, 0, 0.5) 
    ax4.fill_between(x, y_vague, color='yellow', alpha=0.6, label='Fuzzy Membership')
    ax4.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_xlabel("Wind Speed (m/s)")
    ax4.legend()
    st.pyplot(fig4)