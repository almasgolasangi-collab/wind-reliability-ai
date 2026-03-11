import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Wind AI Reliability", layout="wide")

# --- 1. LOGO & SIDEBAR ---
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.title("⚡ Wind AI Systems")

st.sidebar.header("Data Input Control")
input_choice = st.sidebar.radio("Input Method:", ["Slider (Fast Demo)", "Type List of Numbers"])

if input_choice == "Type List of Numbers":
    data_in = st.sidebar.text_input("Enter Wind Speeds:", "14, 16, 15, 19, 17")
    try:
        data = [float(x.strip()) for x in data_in.split(",")]
    except:
        data = [15.0, 16.0, 14.0]
else:
    mean_v = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 40.0, 11.26)
    std_v = st.sidebar.slider("Turbulence Level", 0.1, 5.0, 3.38)
    data = np.random.normal(mean_v, std_v, 10)

# --- 2. MATH ENGINE ---
avg_v = np.mean(data)
std_v_calc = np.std(data) if len(data) > 1 else 0.5
x_axis = np.linspace(0, 50, 500)

# A. Fault Tree
rel_ft = 1.0 if avg_v < 25 else 0.0
y_ft = np.where(x_axis < 25, 1.0, 0.0)

# B. Markov Chain
rel_mar = np.exp(-0.04 * avg_v)
y_mar = np.exp(-0.04 * x_axis)

# C. Monte Carlo (Fixed the curve)
y_mc = (1/(std_v_calc * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - avg_v)/std_v_calc)**2)
rel_mc = np.clip(1 - (avg_v / 40), 0, 1)

# D. Vague Set AI
pi = np.clip((std_v_calc/avg_v)*2, 0.1, 0.5) if avg_v > 0 else 0.5
y_vague = np.where(x_axis < 25, 0.5, 0.5 * np.exp(-0.1 * (x_axis - 25)))
rel_vague = 1 - (avg_v/50) - (pi/2)

# --- 3. CLEAN DASHBOARD METRICS ---
st.title("⚡ Wind AI: Multi-Method Reliability Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fault Tree Status", "SAFE" if rel_ft > 0.5 else "CRITICAL")
c2.metric("Markov Reliability", f"{rel_mar*100:.1f}%")
c3.metric("Monte Carlo Score", f"{rel_mc*100:.1f}%")
c4.metric("Vague Uncertainty (π)", f"{pi:.2f}")

# --- 4. IMPROVED GRAPHS ---
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

ax[0,0].step(x_axis, y_ft, color='black', label="Logic Path"); ax[0,0].set_title("1. Fault Tree (Deterministic)")
ax[0,1].plot(x_axis, y_mar, color='orange', label="Stochastic Decay"); ax[0,1].set_title("2. Markov Chain (Probabilistic)")
# Graph 3 Fix: Scaling the curve so it's visible
ax[1,0].fill_between(x_axis, 0, y_mc * 5, color='blue', alpha=0.3, label="Probability Density"); ax[1,0].set_title("3. Monte Carlo (Sampling)")
ax[1,1].fill_between(x_axis, 0, y_vague, color='gold', alpha=0.6, label="Uncertainty Zone"); ax[1,1].set_title("4. Vague Set AI (Uncertainty)")

for a in ax.flat:
    a.axvline(avg_v, color='red', linestyle='--', label="Current Wind")
    a.set_ylim(-0.1, 1.1); a.grid(True, alpha=0.2); a.legend(fontsize='x-small')

st.pyplot(fig)

# --- 5. BAR CHART ---
st.subheader("Reliability Method Comparison")
methods = ['Fault Tree', 'Markov', 'Monte Carlo', 'Vague AI']
scores = [rel_ft, rel_mar, rel_mc, rel_vague]
fig2, ax2 = plt.subplots(figsize=(10, 2.5))
ax2.barh(methods, scores, color=['#2c3e50', '#e67e22', '#3498db', '#27ae60'])
ax2.set_xlim(0, 1.1)
st.pyplot(fig2)