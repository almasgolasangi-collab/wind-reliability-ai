import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Hybrid Wind AI", layout="wide")
st.title("⚡ IEC 61400-1: 4-Layer Reliability Engine")

# --- SIDEBAR ---
st.sidebar.header("Data Input")
mean_v = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 40.0, 15.0)
std_v = st.sidebar.slider("Turbulence (Std Dev)", 0.1, 5.0, 2.0)

# --- MATH CALCULATIONS ---
x = np.linspace(0, 45, 500)

# 1. Fault Tree (Binary)
y_ft = np.where(x < 25, 1.0, 0.0)
curr_ft = 1.0 if mean_v < 25 else 0.0

# 2. Markov Chain (Stochastic Decay)
y_markov = np.exp(-0.03 * x)
curr_markov = np.exp(-0.03 * mean_v)

# 3. Monte Carlo (Probabilistic Distribution)
y_mc = (1/(std_v * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_v)/std_v)**2)
mc_samples = np.random.normal(mean_v, std_v, 1000)
curr_mc = np.sum(mc_samples < 25) / 1000

# 4. Vague Set (AI Uncertainty Layer)
# Membership mu and Non-membership nu
mu = np.exp(-max(0, x - 15)**2 / 100)
nu = 1 - np.exp(-max(0, x - 30)**2 / 50)
pi_vague_zone = 1 - mu - nu
curr_pi = 1 - (np.exp(-max(0, mean_v - 15)**2 / 100)) - (1 - np.exp(-max(0, mean_v - 30)**2 / 50))

# --- DASHBOARD METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fault Tree", "SAFE" if curr_ft > 0 else "CRITICAL")
c2.metric("Markov Reliability", f"{curr_markov*100:.1f}%")
c3.metric("MC Success Rate", f"{curr_mc*100:.1f}%")
c4.metric("Vague Uncertainty (π)", f"{max(0, curr_pi):.2f}")

# --- THE 4-PANEL GRAPH ---
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
plt.subplots_adjust(hspace=0.4)

# Plot 1: Fault Tree
ax[0,0].plot(x, y_ft, color='black', label="Binary Logic")
ax[0,0].axvline(mean_v, color='red', linestyle='--')
ax[0,0].set_title("1. Fault Tree (Deterministic)")
ax[0,0].set_ylim(-0.1, 1.1)

# Plot 2: Markov
ax[0,1].plot(x, y_markov, color='orange', label="Stochastic")
ax[0,1].axvline(mean_v, color='red', linestyle='--')
ax[0,1].set_title("2. Markov Chain (Stochastic)")

# Plot 3: Monte Carlo
ax[1,0].fill_between(x, 0, y_mc, color='blue', alpha=0.3)
ax[1,0].axvline(mean_v, color='red', linestyle='--')
ax[1,0].set_title("3. Monte Carlo (Probabilistic)")

# Plot 4: Vague Set
ax[1,1].fill_between(x, 0, pi_vague_zone, color='gold', alpha=0.5)
ax[1,1].axvline(mean_v, color='red', linestyle='--')
ax[1,1].set_title("4. Vague Set AI (Uncertainty Layer)")

for a in ax.flat:
    a.set_xlabel("Wind Speed (m/s)")
    a.grid(True, alpha=0.2)

st.pyplot(fig)

# --- COMPARISON SUMMARY ---
st.subheader("Method Comparison")
methods = ['Fault Tree', 'Markov', 'Monte Carlo', 'Vague AI']
reliability = [curr_ft, curr_markov, curr_mc, 1.0 - max(0, curr_pi)]
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.barh(methods, reliability, color=['#2c3e50', '#e67e22', '#3498db', '#f1c40f'])
ax2.set_xlim(0, 1.1)
ax2.set_title("Real-Time Reliability Score Comparison")
st.pyplot(fig2)