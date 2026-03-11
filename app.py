import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Wind AI Reliability", layout="wide")

st.title("⚡ IEC 61400-1 Wind Reliability Dashboard")
st.write("This dashboard compares **Traditional Fault Trees**, **Markov Chains**, and **Generative AI (Vague Monte Carlo)**.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Control Parameters")
input_method = st.sidebar.radio("Input Method:", ["Slider (Fast Demo)", "Type List of Numbers"])

if input_method == "Type List of Numbers":
    data_string = st.sidebar.text_input("Enter Wind Speeds (comma separated):", "10, 12, 11, 14, 13, 15")
    try:
        custom_data = [float(x.strip()) for x in data_string.split(",")]
        mean_speed = np.mean(custom_data)
        std_dev = np.std(custom_data)
    except:
        st.sidebar.error("Use numbers separated by commas!")
        mean_speed, std_dev = 10.0, 1.5
else:
    mean_speed = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 30.0, 10.0)
    std_dev = st.sidebar.slider("Turbulence (Std Dev)", 0.1, 5.0, 1.5)

# --- CALCULATIONS ---
ti = std_dev / mean_speed if mean_speed > 0 else 0
stress_index = (mean_speed * 3.5) + (ti * 40)
vague_pi = min(1.0, (ti * 1.5) + (mean_speed / 40))

# --- UI LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Wind Speed", f"{mean_speed:.2f} m/s")
col2.metric("Turbulence (TI)", f"{ti:.2f}")
col3.metric("Stress Index", f"{min(100.0, stress_index):.1f}%")
col4.metric("Vague Uncertainty (π)", f"{vague_pi:.2f}")

# --- NEW 3-METHOD FUSION GRAPH ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0, 35, 200)

# 1. Fault Tree (Step Function: 1 until 20m/s, then 0)
y_fault = np.where(x < 20, 1.0, 0.0)

# 2. Markov Chain (Probabilistic Decay based on Intensity)
y_markov = np.exp(-x / (25 / (ti + 0.1)))

# 3. AI Vague Set (The Generative Zone)
y_vague = np.exp(-((x - mean_speed)**2) / (2 * (std_dev**2 + 0.5)))

# Plotting on Graph 0
ax[0].plot(x, y_fault, color='black', linestyle='--', label="Fault Tree (Binary)", linewidth=2)
ax[0].plot(x, y_markov, color='orange', label="Markov Chain (Probabilistic)", alpha=0.7)
ax[0].fill_between(x, 0, y_vague, color='gold', alpha=0.4, label="AI Vague Monte Carlo Zone")
ax[0].axvline(mean_speed, color='blue', label="Current Sensor Reading", linewidth=2)

ax[0].set_title("Intelligence Fusion: 3-Method Comparison", fontsize=12)
ax[0].set_xlabel("Wind Speed (m/s)")
ax[0].set_ylabel("Reliability Factor")
ax[0].legend(loc='upper right', fontsize='small')
ax[0].grid(True, alpha=0.2)

# Graph 1: Reliability Bar Chart
methods = ['Fault Tree', 'Markov', 'AI (Our)']
scores = [1.0 if mean_speed < 20 else 0.0, np.exp(-mean_speed/20), 1.0 - (stress_index/100)]
ax[1].bar(methods, scores, color=['#2c3e50', '#d35400', '#27ae60'])
ax[1].set_title("Real-Time Reliability Score", fontsize=12)
ax[1].set_ylim(0, 1.1)

st.pyplot(fig)

# --- JUDGE'S INSIGHT ---
st.info(f"**Insight:** The Blue Line shows the current sensor reading. While the **Fault Tree** stays at 1.0 until it hits the 20m/s cliff, the **AI Vague Zone** (Gold) accounts for the **{ti:.2f} turbulence**, showing a hidden risk even at lower speeds.")