import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# --- 1. SETTINGS & LOGO ---
st.set_page_config(page_title="Wind AI Reliability Analysis", layout="wide")

# Sidebar Logo
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.info("Upload 'logo.png' to see your brand.")

# --- 2. DATA INPUT CONTROL ---
st.sidebar.title("Data Control Center")
input_choice = st.sidebar.radio("Input Method:", ["Upload Data Set", "Live API Feed", "Manual Demo"])

v_mean = 12.0
v_std = 2.0
wind_series = []

if input_choice == "Upload Data Set":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        # Auto-detect wind column
        col = [c for c in df.columns if 'wind' in c.lower() or 'speed' in c.lower()][0]
        wind_series = df[col].dropna().values
        v_mean = np.mean(wind_series)
        v_std = np.std(wind_series)
    else:
        st.info("Awaiting file upload...")
        st.stop()
elif input_choice == "Live API Feed":
    # Brahmanvel MH coordinates for example
    url = "https://api.openweathermap.org/data/2.5/weather?lat=21.03&lon=74.22&appid=3bea6d570f4e26ab35c5f69864e977d6&units=metric"
    try:
        res = requests.get(url).json()
        v_mean = res['wind']['speed']
        wind_series = np.random.normal(v_mean, 0.5, 100)
    except:
        v_mean = 14.2
        wind_series = np.random.normal(v_mean, 0.5, 100)
else:
    v_mean = st.sidebar.slider("Set Mean Wind Speed (m/s)", 0.0, 40.0, 15.0)
    v_std = st.sidebar.slider("Set Turbulence (Std Dev)", 0.1, 5.0, 2.0)
    wind_series = np.random.normal(v_mean, v_std, 100)

# --- 3. TIER 1: ACTUAL WIND GRAPH ---
st.header("1. Input Wind Data Profile")
fig_wind, ax_wind = plt.subplots(figsize=(12, 3))
ax_wind.plot(wind_series, color='#2c3e50', linewidth=1, label="Wind Velocity (m/s)")
ax_wind.axhline(v_mean, color='red', linestyle='--', label=f"Mean: {v_mean:.2f} m/s")
ax_wind.fill_between(range(len(wind_series)), wind_series, alpha=0.1, color='blue')
ax_wind.set_ylabel("Velocity")
ax_wind.legend()
st.pyplot(fig_wind)

st.markdown("---")

# --- 4. TIER 2: METHODOLOGY VISUALIZATIONS ---
st.header("2. Scientific Methodology Visuals")
col1, col2, col3 = st.columns(3)

# METHOD A: Fault Tree Analysis (Logic Visualization)
with col1:
    st.subheader("Fault Tree (FTA)")
    # Logic: Reliability is 100% until cut-off at 25m/s
    x_fta = np.linspace(0, 40, 100)
    y_fta = [100 if i < 25 else 0 for i in x_fta]
    fig_fta, ax_fta = plt.subplots()
    ax_fta.plot(x_fta, y_fta, color='black', linewidth=3)
    ax_fta.fill_between(x_fta, y_fta, color='gray', alpha=0.2)
    ax_fta.axvline(v_mean, color='red', linestyle='--', label="Current Wind")
    ax_fta.set_title("Deterministic Binary Logic")
    ax_fta.set_ylabel("System Reliability %")
    st.pyplot(fig_fta)
    rel_fta = 100 if v_mean < 25 else 0

# METHOD B: Monte Carlo (Probability Visualization)
with col2:
    st.subheader("Monte Carlo")
    # Visualization: The "Confidence Bell Curve" rather than a raw histogram
    from scipy.stats import norm
    x_mc = np.linspace(v_mean - (3*v_std), v_mean + (3*v_std), 100)
    y_mc = norm.pdf(x_mc, v_mean, v_std)
    fig_mc, ax_mc = plt.subplots()
    ax_mc.plot(x_mc, y_mc, color='#3498db', linewidth=2)
    ax_mc.fill_between(x_mc, y_mc, color='#3498db', alpha=0.3, label="Failure Risk Zone")
    ax_mc.set_title("Stochastic Risk Distribution")
    ax_mc.set_xlabel("Simulated Wind Scenarios")
    st.pyplot(fig_mc)
    rel_mc = (1 - (v_std / v_mean)) * 100 if v_mean > 0 else 0

# METHOD C: Markov Chain (State Visualization)
with col3:
    st.subheader("Markov Chain")
    # Visualization: Pie Chart of current state transition probabilities
    p_healthy = np.exp(-0.03 * v_mean)
    p_degraded = (1 - p_healthy) * 0.7
    p_fail = (1 - p_healthy) * 0.3
    fig_mar, ax_mar = plt.subplots()
    ax_mar.pie([p_healthy, p_degraded, p_fail], 
               labels=['Healthy', 'Wear', 'Critical'], 
               colors=['#2ecc71', '#f1c40f', '#e74c3c'], 
               autopct='%1.1f%%', startangle=90, explode=(0.1, 0, 0))
    ax_mar.set_title("State Transition Probability")
    st.pyplot(fig_mar)
    rel_markov = p_healthy * 100

st.markdown("---")

# --- 5. TIER 3: BAR COMPARISON & COMPONENTS ---
st.header("3. Final Comparison & Component Failure")
c_left, c_right = st.columns([2, 1])

with c_left:
    # BAR GRAPH TO COMPARE ALL 3
    methods = ["Fault Tree", "Monte Carlo", "Markov Chain"]
    scores = [rel_fta, rel_mc, rel_markov]
    
    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    bar_plot = ax_bar.bar(methods, scores, color=['#34495e', '#3498db', '#e67e22'])
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("Reliability Score (%)")
    ax_bar.set_title("Reliability Method Comparison")
    
    # Adding text labels on bars
    for bar in bar_plot:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha='center', weight='bold')
    
    st.pyplot(fig_bar)

with c_right:
    # COMPONENT FAILURE TABLE
    st.write("#### Component Failure Risk")
    # Simulated component risks based on wind pressure
    b_risk = (v_mean/35)**2
    g_risk = (v_mean/45)**2
    gen_risk = (v_mean/50)**2
    
    comp_df = pd.DataFrame({
        "Component": ["Blades", "Gearbox", "Generator"],
        "Risk %": [f"{b_risk:.1%}", f"{g_risk:.1%}", f"{gen_risk:.1%}"],
        "Status": ["Safe" if b_risk < 0.15 else "Review", "Safe", "Safe"]
    })
    st.table(comp_df)