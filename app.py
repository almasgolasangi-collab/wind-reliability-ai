import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# --- 1. PAGE SETUP & UI CLEANUP ---
st.set_page_config(page_title="Wind AI Reliability", layout="wide")

# This hides the "Streamlit" menu and footer to make it look like a standalone App
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 2. LOGO & SIDEBAR ---
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.title("⚡ Wind AI Systems")

st.sidebar.header("Data Input Control")
input_choice = st.sidebar.radio("Input Method:", ["Slider (Fast Demo)", "Type List of Numbers"])

if input_choice == "Type List of Numbers":
    data_in = st.sidebar.text_input("Enter Wind Speeds (m/s):", "14, 16, 15, 19, 17")
    try:
        data = [float(x.strip()) for x in data_in.split(",")]
    except:
        data = [15.0, 16.0, 14.0]
else:
    mean_v = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 40.0, 11.26)
    std_v = st.sidebar.slider("Turbulence Level", 0.1, 5.0, 3.38)
    data = np.random.normal(mean_v, std_v, 10)

# --- 3. MATH ENGINE ---
avg_v = np.mean(data)
std_v_calc = np.std(data) if len(data) > 1 else 0.5
ti = std_v_calc / avg_v if avg_v > 0 else 0
x_axis = np.linspace(0, 50, 500)

# A. Fault Tree Logic
rel_ft = 1.0 if avg_v < 25 else 0.0
y_ft = np.where(x_axis < 25, 1.0, 0.0)

# B. Markov Chain Reliability
rel_mar = np.exp(-0.04 * avg_v)
y_mar = np.exp(-0.04 * x_axis)

# C. Monte Carlo (Sampling Curve)
y_mc = (1/(std_v_calc * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - avg_v)/std_v_calc)**2)
y_mc_norm = y_mc / (max(y_mc) if max(y_mc) > 0 else 1)
rel_mc = np.clip(1 - (avg_v / 42), 0, 1)

# D. Vague Set AI (Uncertainty Layer)
pi = np.clip((std_v_calc/avg_v)*2, 0.1, 0.5) if avg_v > 0 else 0.5
y_vague = np.where(x_axis < 25, 0.6, 0.6 * np.exp(-0.1 * (x_axis - 25)))
rel_vague = 1 - (avg_v/50) - (pi/2)

# --- 4. DASHBOARD METRICS ---
st.title("⚡ IEC 61400-1: Wind AI Reliability Engine")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fault Tree Logic", "SAFE" if rel_ft > 0.5 else "CRITICAL")
c2.metric("Markov Reliability", f"{rel_mar*100:.1f}%")
c3.metric("Monte Carlo Score", f"{rel_mc*100:.1f}%")
c4.metric("Vague Uncertainty (π)", f"{pi:.2f}")

# --- 5. VISUALIZATION (4-PANEL) ---
st.subheader("Reliability Analysis Comparison")
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot 1: Fault Tree
ax[0,0].step(x_axis, y_ft, color='black', label="Binary Logic")
ax[0,0].set_title("1. Fault Tree (Deterministic)")

# Plot 2: Markov
ax[0,1].plot(x_axis, y_mar, color='orange', label="Stochastic Decay")
ax[0,1].set_title("2. Markov Chain (Probabilistic)")

# Plot 3: Monte Carlo
ax[1,0].fill_between(x_axis, 0, y_mc_norm, color='blue', alpha=0.3, label="Probability Curve")
ax[1,0].set_title("3. Monte Carlo (Sampling)")

# Plot 4: Vague Set AI
ax[1,1].fill_between(x_axis, 0, y_vague, color='gold', alpha=0.6, label="Uncertainty Zone")
ax[1,1].set_title("4. Vague Set AI (Uncertainty)")

for a in ax.flat:
    a.axvline(avg_v, color='red', linestyle='--', label="Current Wind Speed")
    a.set_ylim(-0.1, 1.1)
    a.set_xlabel("Wind Speed (m/s)")
    a.grid(True, alpha=0.2)
    a.legend(loc='upper right', fontsize='x-small')

st.pyplot(fig)

# --- 6. FINAL RELIABILITY SUMMARY ---
st.markdown("---")
st.subheader("Global Reliability Summary")
methods = ['Fault Tree', 'Markov Chain', 'Monte Carlo', 'Vague AI']
scores = [rel_ft, rel_mar, rel_mc, rel_vague]

# Create a Horizontal Bar Chart
fig2, ax2 = plt.subplots(figsize=(10, 2.5))
ax2.barh(methods, scores, color=['#2c3e50', '#e67e22', '#3498db', '#27ae60'])
ax2.set_xlim(0, 1.1)
for i, v in enumerate(scores):
    ax2.text(v + 0.02, i, f"{v:.2f}", va='center', fontweight='bold')
st.pyplot(fig2)

# --- 7. EXPORT DATA (BONUS FEATURE) ---
st.sidebar.markdown("---")
if st.sidebar.button("Generate Reliability Report"):
    report_data = {
        "Metric": methods,
        "Reliability Score": [f"{s:.2f}" for s in scores],
        "Avg Wind (m/s)": [f"{avg_v:.2f}"] * 4
    }
    df = pd.DataFrame(report_data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name='Wind_Reliability_Report.csv',
        mime='text/csv',
    )