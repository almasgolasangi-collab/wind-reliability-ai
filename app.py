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

# --- 2. SIDEBAR (LOGO & CONTROLS) ---
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.info("💡 Tip: Place 'logo.png' in the same folder to see the app logo.")

st.sidebar.title("Data Input Control")
input_method = st.sidebar.radio("Input Method:", ["Slider (Fast Demo)", "Live API", "Set of Data (CSV/Excel)"])

# Default Values
v_curr = 11.26
turb = 3.38

if input_method == "Slider (Fast Demo)":
    v_curr = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 50.0, 11.26)
    turb = st.sidebar.slider("Turbulence Level", 0.0, 10.0, 3.38)

elif input_method == "Live API":
    site = st.sidebar.selectbox("Select Site", list(SITES.keys()))
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={SITES[site]['lat']}&lon={SITES[site]['lon']}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        v_curr = res['wind']['speed']
        st.sidebar.success(f"Live Speed: {v_curr} m/s")
    except:
        st.sidebar.error("API Error. Using default.")
        v_curr = 11.26
    turb = 3.38

elif input_method == "Set of Data (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("Upload SCADA/Wind Log", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        # Attempt to find wind speed column automatically
        potential_cols = [c for c in df.columns if 'wind' in c.lower() or 'speed' in c.lower()]
        if potential_cols:
            col = potential_cols[0]
            v_curr = df[col].mean()
            turb = df[col].std()
            st.sidebar.write(f"**Processed Mean:** {v_curr:.2f} m/s")
            st.sidebar.write(f"**Processed Turb:** {turb:.2f}")
        else:
            st.sidebar.error("No 'Wind' or 'Speed' column found.")
    else:
        st.sidebar.warning("Awaiting file upload...")

# --- 3. TOP METRICS ---
status = "SAFE" if v_curr < 25 else "DANGER"
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.markdown(f"<h1 style='text-align: center; color: {'green' if status=='SAFE' else 'red'};'>{status}</h1>", unsafe_allow_html=True)
col_m2.markdown(f"<h1 style='text-align: center;'>{max(0, 100 - (v_curr*2.1)):.1f}%</h1>", unsafe_allow_html=True)
col_m3.markdown(f"<h1 style='text-align: center;'>{max(0, 100 - (v_curr*1.8)):.1f}%</h1>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Reliability Method Comparison")

# --- 4. THE 4-PLOT GRID ---
x = np.linspace(0, 50, 100)
c1, c2 = st.columns(2)

with c1:
    st.write("1. Fault Tree (Deterministic)")
    fig1, ax1 = plt.subplots()
    y_fta = np.where(x < 25, 1.0, 0.0)
    ax1.plot(x, y_fta, color='black', label='Binary Logic')
    ax1.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    st.pyplot(fig1)

with c2:
    st.write("2. Markov Chain (Probabilistic)")
    fig2, ax2 = plt.subplots()
    y_markov = np.exp(-0.04 * x)
    ax2.plot(x, y_markov, color='orange', label='State Reliability')
    ax2.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    st.pyplot(fig2)

c3, c4 = st.columns(2)

with c3:
    st.write("3. Monte Carlo (Sampling)")
    fig3, ax3 = plt.subplots()
    # If turb is NaN (single data point), default to 1.0
    safe_turb = turb if not np.isnan(turb) else 1.0
    y_mc = np.exp(-((x - v_curr)**2) / (2 * (max(0.1, safe_turb)**2)))
    ax3.fill_between(x, y_mc, color='blue', alpha=0.2, label='Probability Density')
    ax3.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    st.pyplot(fig3)

with c4:
    st.write("4. Vague Set (Membership)")
    fig4, ax4 = plt.subplots()
    y_vague = np.clip((40 - x) / 15, 0, 0.5) 
    ax4.fill_between(x, y_vague, color='yellow', alpha=0.6, label='Fuzzy Membership')
    ax4.axvline(v_curr, color='red', linestyle='--', label='Current Wind')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend()
    st.pyplot(fig4)