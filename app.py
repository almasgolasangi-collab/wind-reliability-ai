import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Hybrid Wind AI", layout="wide")
st.title("⚡ IEC 61400-1: 4-Layer Reliability Engine")
st.write("Comparing **Fault Tree**, **Markov**, **Monte Carlo**, and **Vague Set AI**.")

# --- SIDEBAR ---
st.sidebar.header("Data Input")
input_choice = st.sidebar.radio("Input Method:", ["Slider", "Data Stream (List)"])

if input_choice == "Data Stream (List)":
    data_in = st.sidebar.text_input("Enter Speeds (e.g. 12, 15, 14, 18):", "14, 16, 15, 19, 17")
    try:
        vals = [float(x.strip()) for x in data_in.split(",")]
        mean_v, std_v = np.mean(vals), np.std(vals)
    except:
        mean_v, std_v = 15.0, 2.0
else:
    mean_v = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 35.0, 15.0)
    std_v = st.sidebar.slider("Turbulence (Std Dev)", 0.1, 5.0, 1.8)

# --- 4-LAYER CALCULATIONS ---
ti = std_v / mean_v if mean_v > 0 else 0

# 1. Fault Tree (Binary Cliff at 25m/s)
ft_rel = 1.0 if mean_v < 25 else 0.0

# 2. Markov (State Transition - Simplified)
markov_rel = np.exp(-0.02 * mean_v) 

# 3. Monte Carlo (1000 Simulations)
mc_sims = np.random.normal(mean_v, std_v, 1000)
mc_rel = np.sum(mc_sims < 25) / 1000

# 4. Vague Set (The AI Layer)
# Membership (Safe) and Non-Membership (Risk)
mu = np.exp(-max(0, mean_v - 10)**2 / 150)
nu = 1 - np.exp(-max(0, mean_v - 28)**2 / 40)
pi_vague = 1 - mu - nu  # The "Vague" Uncertainty

# --- DASHBOARD METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fault Tree", "SAFE" if ft_rel > 0 else "FAIL")
c2.metric("Markov Reliability", f"{markov_rel*100:.1f}%")
c3.metric("Monte Carlo Prob", f"{mc_rel*100:.1f}%")
c4.metric("Vague Uncertainty (π)", f"{max(0, pi_vague):.2f}")

# --- THE COMPARISON GRAPH ---
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
x = np.linspace(0, 40, 300)

# Plotting the Logic Layers
ax.plot(x, np.where(x < 25, 1.0, 0.0), color='black', label="Fault Tree (Rigid)", linestyle='--')
ax.plot(x, np.exp(-0.02 * x), color='orange', label="Markov (Stochastic Decay)")
ax.fill_between(x, 0, np.exp(-((x - mean_v)**2)/(2*std_v**2)), color='blue', alpha=0.1, label="Monte Carlo (Samples)")
ax.fill_between(x, 0.4, 0.6, where=(x > 18) & (x < 32), color='gold', alpha=0.3, label="Vague Set (AI Uncertainty Zone)")

ax.axvline(mean_v, color='red', label="CURRENT SENSOR", linewidth=3)
ax.set_title("Evolution of Reliability Logic", fontsize=14)
ax.legend(loc='upper right', fontsize='small')
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Confidence / Reliability")
st.pyplot(fig)

st.success(f"**Final Verdict:** Our Vague AI detects **{max(0, pi_vague):.2f} uncertainty** which the Fault Tree ignores.")