import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Hybrid Wind AI", layout="wide")
st.title("⚡ IEC 61400-1: 4-Layer Reliability Engine")

# --- SIDEBAR: WHERE YOU GIVE INPUT DATA ---
st.sidebar.header("Data Input Control")

# 1. CHOOSE YOUR INPUT METHOD HERE
input_choice = st.sidebar.radio("Input Method:", ["Slider (Fast Demo)", "Type List of Numbers (Custom Data)"])

if input_choice == "Type List of Numbers (Custom Data)":
    # 2. TYPE YOUR DATA HERE (e.g., 12, 15, 14, 22)
    data_in = st.sidebar.text_input("Enter Wind Speeds (comma separated):", "14, 16, 15, 19, 17")
    try:
        vals = [float(x.strip()) for x in data_in.split(",")]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        if std_v == 0: std_v = 0.1 # Prevent math errors
    except:
        st.sidebar.error("Invalid format! Use numbers like: 10, 12, 15")
        mean_v, std_v = 15.0, 2.0
else:
    mean_v = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 45.0, 15.0)
    std_v = st.sidebar.slider("Turbulence (Std Dev)", 0.1, 5.0, 2.0)

# --- MATH CALCULATIONS (FIXED NUMPY) ---
x = np.linspace(0, 50, 500)
y_ft = np.where(x < 25, 1.0, 0.0)
curr_ft = 1.0 if mean_v < 25 else 0.0
y_markov = np.exp(-0.03 * x)
curr_markov = np.exp(-0.03 * mean_v)
y_mc = (1/(std_v * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_v)/std_v)**2)
mc_samples = np.random.normal(mean_v, std_v, 1000)
curr_mc = np.sum(mc_samples < 25) / 1000

# Vague Set AI Logic
mu_vals = np.exp(-np.maximum(0, x - 15)**2 / 150)
nu_vals = 1 - np.exp(-np.maximum(0, x - 30)**2 / 60)
pi_vague_zone = 1 - mu_vals - nu_vals

curr_mu = np.exp(-max(0, mean_v - 15)**2 / 150)
curr_nu = 1 - np.exp(-max(0, mean_v - 30)**2 / 60)
curr_pi = 1 - curr_mu - curr_nu

# --- DASHBOARD METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fault Tree", "SAFE" if curr_ft > 0 else "CRITICAL")
c2.metric("Markov Reliability", f"{curr_markov*100:.1f}%")
c3.metric("MC Success Rate", f"{curr_mc*100:.1f}%")
c4.metric("Vague Uncertainty (π)", f"{max(0.0, curr_pi):.2f}")

# --- THE 4-PANEL GRAPH ---
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
plt.subplots_adjust(hspace=0.4)
ax[0,0].step(x, y_ft, color='black', where='post'); ax[0,0].set_title("1. Fault Tree")
ax[0,1].plot(x, y_markov, color='orange'); ax[0,1].set_title("2. Markov Chain")
ax[1,0].fill_between(x, 0, y_mc, color='blue', alpha=0.3); ax[1,0].set_title("3. Monte Carlo")
ax[1,1].fill_between(x, 0, np.maximum(0, pi_vague_zone), color='gold', alpha=0.6); ax[1,1].set_title("4. Vague Set AI")

for a in ax.flat:
    a.axvline(mean_v, color='red', linestyle='--')
    a.set_xlabel("Wind Speed (m/s)")
    a.grid(True, alpha=0.2)
st.pyplot(fig)

# --- COMPARISON BAR CHART ---
st.subheader("Final Reliability Comparison")
methods = ['Fault Tree', 'Markov', 'Monte Carlo', 'Vague AI']
reliability = [curr_ft, curr_markov, curr_mc, curr_mu]
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.barh(methods, reliability, color=['#2c3e50', '#e67e22', '#3498db', '#27ae60'])
ax2.set_xlim(0, 1.1)
st.pyplot(fig2)
