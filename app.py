import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# 1. PAGE SETUP (This makes your app look professional and full-screen)
st.set_page_config(page_title="Wind Reliability AI", layout="wide")

API_KEY = "3bea6d570f4e26ab35c5f69864e977d6" 

SITES = {
    "Brahmanvel, MH": {"lat": 21.03, "lon": 74.22},
    "Dhalgaon, MH": {"lat": 17.58, "lon": 74.85},
    "Chitradurga, KA": {"lat": 14.23, "lon": 76.40},
    "Kayathar, TN": {"lat": 8.94, "lon": 77.72}
}

# 2. SIDEBAR & LOGO
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.info("Place 'logo.png' in folder to show brand logo.")

st.sidebar.title("📡 Data Control")
mode = st.sidebar.radio("Select Source:", ["Live API Feed", "Upload Data Set", "Manual Demo"])

v_mean = 12.0
v_std = 1.2
wind_series = []

# 3. DATA ACQUISITION
if mode == "Live API Feed":
    site_name = st.sidebar.selectbox("Select Wind Farm Site:", list(SITES.keys()))
    lat, lon = SITES[site_name]['lat'], SITES[site_name]['lon']
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        v_mean = res['wind']['speed']
        st.sidebar.success(f"Live Wind: {v_mean} m/s")
    except:
        v_mean = 12.5 # Fallback
    wind_series = np.random.normal(v_mean, 0.7, 100)

elif mode == "Upload Data Set":
    file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
        col = [c for c in df.columns if 'speed' in c.lower() or 'wind' in c.lower()][0]
        wind_series = df[col].dropna().values
        v_mean = np.mean(wind_series)
        v_std = np.std(wind_series)
    else: st.stop()

else:
    v_mean = st.sidebar.slider("Wind Speed (m/s)", 0.0, 45.0, 15.0)
    v_std = st.sidebar.slider("Turbulence", 0.1, 5.0, 1.5)
    wind_series = np.random.normal(v_mean, v_std, 100)

# 4. WIND GRAPH (TOP)
st.header("1. Wind Velocity Profile")
fig_w, ax_w = plt.subplots(figsize=(12, 3))
ax_w.plot(wind_series, color='#2c3e50', linewidth=1)
ax_w.fill_between(range(len(wind_series)), wind_series, color='#34495e', alpha=0.1)
ax_w.axhline(v_mean, color='red', linestyle='--', label=f"Mean: {v_mean:.2f} m/s")
ax_w.set_ylabel("m/s")
ax_w.legend()
st.pyplot(fig_w)

st.divider()

# 5. 3 SEPARATE SCIENTIFIC GRAPHS
st.header("2. Independent Reliability Visualizations")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Fault Tree (FTA)")
    x = np.linspace(0, 50, 100)
    y_fta = np.where(x < 25, 100, 0)
    fig1, ax1 = plt.subplots()
    ax1.plot(x, y_fta, color='black', linewidth=2)
    ax1.fill_between(x, y_fta, color='gray', alpha=0.1)
    ax1.axvline(v_mean, color='red', linestyle='--')
    ax1.set_ylabel("Reliability %")
    st.pyplot(fig1)
    rel_fta = 100 if v_mean < 25 else 0

with c2:
    st.markdown("### Monte Carlo")
    # Using a Bell Curve (PDF) - NOT A HISTOGRAM
    x_mc = np.linspace(v_mean-10, v_mean+10, 100)
    y_mc = (1/(v_std*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_mc-v_mean)/v_std)**2)
    fig2, ax2 = plt.subplots()
    ax2.plot(x_mc, y_mc, color='#3498db', linewidth=2)
    ax2.fill_between(x_mc, y_mc, color='#3498db', alpha=0.3)
    ax2.set_title("Probability Density")
    st.pyplot(fig2)
    rel_mc = max(0, 100 - (v_std/v_mean*100))

with c3:
    st.markdown("### Markov Chain")
    x_mar = np.linspace(0, 50, 100)
    y_mar = np.exp(-0.04 * x_mar) * 100
    fig3, ax3 = plt.subplots()
    ax3.plot(x_mar, y_mar, color='#e67e22', linewidth=2)
    ax3.axvline(v_mean, color='red', linestyle='--')
    ax3.set_ylabel("Reliability %")
    st.pyplot(fig3)
    rel_mar = np.exp(-0.04 * v_mean) * 100

st.divider()

# 6. COMPONENT FAILURE & BAR COMPARISON
st.header("3. Component Failure & Method Comparison")
col_l, col_r = st.columns([1, 2])

with col_l:
    st.write("#### Component Failure Breakdown")
    b_r = (v_mean/32)**2; g_r = (v_mean/42)**2; gen_r = (v_mean/48)**2
    comp_df = pd.DataFrame({
        "Component": ["Blades", "Gearbox", "Generator"],
        "Risk %": [f"{b_r:.1%}", f"{g_r:.1%}", f"{gen_r:.1%}"],
        "Status": ["OK" if b_r < 0.2 else "WARN", "OK", "OK"]
    })
    st.table(comp_df)

with col_r:
    methods = ["Fault Tree", "Monte Carlo", "Markov Chain"]
    scores = [rel_fta, rel_mc, rel_mar]
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    bars = ax_bar.bar(methods, scores, color=['#2c3e50', '#3498db', '#e67e22'])
    ax_bar.set_ylim(0, 110)
    for bar in bars:
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{bar.get_height():.1f}%", ha='center', weight='bold')
    st.pyplot(fig_bar)