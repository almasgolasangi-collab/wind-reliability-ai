import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- 1. SETTINGS & LOGO ---
st.set_page_config(page_title="Wind AI Reliability", layout="wide")

# Sidebar Logo
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.info("Upload 'logo.png' to Sidebar")

# --- 2. DATA INPUT CONTROL ---
st.sidebar.title("Data Control")
input_choice = st.sidebar.radio("Input Source:", ["Set of Data (Upload)", "Live API", "Manual Demo"])

v_mean = 12.0
v_std = 1.5
wind_series = []

if input_choice == "Set of Data (Upload)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        col = [c for c in df.columns if 'speed' in c.lower() or 'wind' in c.lower()][0]
        wind_series = df[col].dropna().values
        v_mean = np.mean(wind_series)
        v_std = np.std(wind_series)
    else:
        st.stop()
elif input_choice == "Live API":
    # Brahmanvel MH Example
    try:
        url = "https://api.openweathermap.org/data/2.5/weather?lat=21.03&lon=74.22&appid=3bea6d570f4e26ab35c5f69864e977d6&units=metric"
        res = requests.get(url).json()
        v_mean = res['wind']['speed']
    except:
        v_mean = 11.5
    wind_series = np.random.normal(v_mean, 0.8, 100)
else:
    v_mean = st.sidebar.slider("Wind Speed (m/s)", 0.0, 45.0, 15.0)
    v_std = st.sidebar.slider("Turbulence", 0.1, 5.0, 1.2)
    wind_series = np.random.normal(v_mean, v_std, 100)

# --- 3. THE WIND PROFILE GRAPH ---
st.header("1. Wind Speed Profile")
fig_w, ax_w = plt.subplots(figsize=(12, 3))
ax_w.plot(wind_series, color='#2c3e50', linewidth=1.5, label="Raw Wind Speed")
ax_w.fill_between(range(len(wind_series)), wind_series, color='#34495e', alpha=0.1)
ax_w.axhline(v_mean, color='red', linestyle='--', label=f"Mean: {v_mean:.2f} m/s")
ax_w.set_ylabel("m/s")
ax_w.legend()
st.pyplot(fig_w)

st.divider()

# --- 4. INDEPENDENT METHOD VISUALIZATIONS ---
st.header("2. Methodology Analysis")
c1, c2, c3 = st.columns(3)

# A. Fault Tree (Binary Step Graph)
with c1:
    st.write("### Fault Tree (FTA)")
    x_fta = np.linspace(0, 50, 100)
    y_fta = np.where(x_fta < 25, 100, 0) # Logic: 100% until 25m/s
    fig1, ax1 = plt.subplots()
    ax1.plot(x_fta, y_fta, color='black', linewidth=2, label='System Limit')
    ax1.axvline(v_mean, color='red', linestyle='--', label='Current Wind')
    ax1.fill_between(x_fta, y_fta, alpha=0.1, color='gray')
    ax1.set_ylabel("Reliability %")
    ax1.legend()
    st.pyplot(fig1)
    rel_fta = 100 if v_mean < 25 else 0

# B. Monte Carlo (Normal Distribution Curve)
with c2:
    st.write("### Monte Carlo")
    # Using a Bell Curve to show Probability Density (No Histogram)
    x_mc = np.linspace(v_mean - 10, v_mean + 10, 100)
    y_mc = (1 / (v_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_mc - v_mean) / v_std)**2)
    fig2, ax2 = plt.subplots()
    ax2.plot(x_mc, y_mc, color='#3498db', linewidth=2, label='Failure Probability')
    ax2.fill_between(x_mc, y_mc, color='#3498db', alpha=0.3)
    ax2.axvline(v_mean, color='red', linestyle='--')
    ax2.set_title("Stochastic Risk Distribution")
    st.pyplot(fig2)
    rel_mc = max(0, 100 - (v_std / v_mean * 100))

# C. Markov Chain (State Decay Curve)
with c3:
    st.write("### Markov Chain")
    x_mar = np.linspace(0, 50, 100)
    y_mar = np.exp(-0.04 * x_mar) * 100 # Exponential Decay
    fig3, ax3 = plt.subplots()
    ax3.plot(x_mar, y_mar, color='orange', linewidth=2, label='State Reliability')
    ax3.axvline(v_mean, color='red', linestyle='--')
    ax3.set_ylabel("Reliability %")
    ax3.legend()
    st.pyplot(fig3)
    rel_mar = np.exp(-0.04 * v_mean) * 100

st.divider()

# --- 5. COMPONENT FAILURE & COMPARISON ---
st.header("3. Component Failure & Comparison Bar Graph")
col_left, col_right = st.columns([1, 2])

with col_left:
    st.write("#### Component Failure Risk")
    # Calculated risk for specific parts
    b_risk = (v_mean/32)**2; g_risk = (v_mean/42)**2; gen_risk = (v_mean/48)**2
    comp_df = pd.DataFrame({
        "Component": ["Blades", "Gearbox", "Generator"],
        "Risk %": [f"{b_risk:.1%}", f"{g_risk:.1%}", f"{gen_risk:.1%}"],
        "Condition": ["SAFE" if b_risk < 0.2 else "WARN", "SAFE", "SAFE"]
    })
    st.table(comp_df)

with col_right:
    # BAR GRAPH TO COMPARE ALL 3
    methods = ["Fault Tree", "Monte Carlo", "Markov Chain"]
    scores = [rel_fta, rel_mc, rel_mar]
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    bars = ax_bar.bar(methods, scores, color=['#2c3e50', '#3498db', '#e67e22'])
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("Final Reliability %")
    
    # Adding values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.1f}%", ha='center', fontweight='bold')
    
    st.pyplot(fig_bar)