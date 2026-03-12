import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. BRANDED APP CONFIGURATION ---
st.set_page_config(page_title="Wind AI Reliability Engine", page_icon="logo.png", layout="wide")

# --- 2. THE "CLEAN APP" UI ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    .app-title { font-size: 42px; font-weight: 800; color: #1E3A8A; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR & FILE UPLOADER ---
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("<h2 style='text-align: center; color: #1E3A8A;'>WindAI Systems</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.header("📂 Data Input")
input_choice = st.sidebar.radio("Choose Input Type:", ["Upload Excel/CSV", "Live Simulator", "Manual Entry"])

data = [15.0, 16.0, 14.0] # Default fallback

if input_choice == "Upload Excel/CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Wind Data", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Look for a column named 'wind' or 'speed', otherwise take the first column
            col_name = next((c for c in df.columns if 'wind' in c.lower() or 'speed' in c.lower()), df.columns[0])
            data = df[col_name].dropna().values
            st.sidebar.success(f"Loaded {len(data)} readings from '{col_name}'")
        except Exception as e:
            st.sidebar.error("Error reading file. Ensure it has a column of numbers.")

elif input_choice == "Manual Entry":
    data_in = st.sidebar.text_input("Enter Wind Speeds (comma separated):", "14.5, 16.2, 15.1")
    data = [float(x.strip()) for x in data_in.split(",")]
else:
    mean_v = st.sidebar.slider("Mean Speed (m/s)", 0.0, 45.0, 12.5)
    data = np.random.normal(mean_v, 2.5, 15)

# --- 4. RELIABILITY MATH ENGINE ---
avg_v = np.mean(data)
std_v_calc = np.std(data) if len(data) > 1 else 0.5
x_axis = np.linspace(0, 50, 500)

# Calculations
rel_ft = 1.0 if avg_v < 25 else 0.0
rel_mar = np.exp(-0.045 * avg_v)
rel_mc = np.clip(1 - (avg_v / 42.5), 0, 1)
pi = np.clip((std_v_calc/avg_v)*1.8, 0.1, 0.5) if avg_v > 0 else 0.4
rel_vague = 1 - (avg_v/48) - (pi/2)

# Curves
y_ft = np.where(x_axis < 25, 1.0, 0.0)
y_mar = np.exp(-0.045 * x_axis)
y_mc = (1/(std_v_calc * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - avg_v)/std_v_calc)**2)
y_mc_norm = y_mc / (max(y_mc) if max(y_mc) > 0 else 1)
y_vague = np.where(x_axis < 24, 0.65, 0.65 * np.exp(-0.12 * (x_axis - 24)))

# --- 5. MAIN DASHBOARD ---
st.markdown("<p class='app-title'>⚡ Wind AI Reliability Engine</p>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: gray;'>Processing: {len(data)} Data Points | Avg: {avg_v:.2f} m/s</p>", unsafe_allow_html=True)

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Fault Tree", "SAFE" if rel_ft > 0.5 else "CRITICAL")
m2.metric("Markov Rel.", f"{rel_mar*100:.1f}%")
m3.metric("Monte Carlo", f"{rel_mc*100:.1f}%")
m4.metric("AI Vague Index", f"{pi:.2f}")

# Graphs
fig, ax = plt.subplots(2, 2, figsize=(12, 8), facecolor='#f8f9fa')
ax[0,0].step(x_axis, y_ft, color='#2c3e50'); ax[0,0].set_title("1. Fault Tree")
ax[0,1].plot(x_axis, y_mar, color='#e67e22'); ax[0,1].set_title("2. Markov Chain")
ax[1,0].fill_between(x_axis, 0, y_mc_norm, color='#3498db', alpha=0.4); ax[1,0].set_title("3. Monte Carlo")
ax[1,1].fill_between(x_axis, 0, y_vague, color='#f1c40f', alpha=0.6); ax[1,1].set_title("4. Vague Set AI")

for a in ax.flat:
    a.axvline(avg_v, color='red', linestyle='--')
    a.set_ylim(-0.1, 1.1)

st.pyplot(fig)