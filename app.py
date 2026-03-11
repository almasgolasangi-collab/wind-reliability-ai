import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Wind AI Reliability", layout="wide")

st.title("⚡ IEC 61400-1 Wind Reliability Dashboard")
st.write("This dashboard compares **Traditional Fault Trees** with **Generative AI (Vague Monte Carlo)** using custom data inputs.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Control Parameters")

# NEW FEATURE: Manual List Input
input_method = st.sidebar.radio("Input Method:", ["Slidder (Fast Demo)", "Type List of Numbers (Custom Data)"])

if input_method == "Type List of Numbers (Custom Data)":
    data_string = st.sidebar.text_input("Enter Wind Speeds (comma separated):", "10, 12, 11, 14, 13, 15")
    # Convert string to list of floats
    try:
        custom_data = [float(x.strip()) for x in data_string.split(",")]
        mean_speed = np.mean(custom_data)
        std_dev = np.std(custom_data)
    except:
        st.sidebar.error("Invalid format! Use numbers separated by commas.")
        mean_speed, std_dev = 10.0, 2.0
else:
    mean_speed = st.sidebar.slider("Mean Wind Speed (m/s)", 0.0, 30.0, 10.0)
    std_dev = st.sidebar.slider("Turbulence (Std Dev)", 0.1, 5.0, 1.5)

# --- CALCULATIONS (THE AI BRAIN) ---
ti = std_dev / mean_speed if mean_speed > 0 else 0
stress_index = (mean_speed * 3.5) + (ti * 40)
vague_pi = min(1.0, (ti * 1.5) + (mean_speed / 40))

# --- UI LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Wind Speed", f"{mean_speed:.2f} m/s")
col2.metric("Turbulence (TI)", f"{ti:.2f}")
col3.metric("Stress Index", f"{min(100.0, stress_index):.1f}%")
col4.metric("Vague Uncertainty (π)", f"{vague_pi:.2f}")

# --- GRAPHING ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Graph 1: Intelligence Fusion Map
x = np.linspace(0, 30, 100)
y_vague = np.exp(-((x - mean_speed)**2) / (2 * std_dev**2))
ax[0].fill_between(x, 0, y_vague, color='gold', alpha=0.3, label="Vague Uncertainty Zone")
ax[0].axvline(20, color='black', linestyle='--', label="Fault Tree Limit (20m/s)")
ax[0].axvline(mean_speed, color='blue', label="AI Risk Detection")
ax[0].set_title("Intelligence Fusion Map")
ax[0].legend()

# Graph 2: Comparison
methods = ['Fault Tree', 'Markov Chain', 'AI Monte Carlo']
scores = [1.0 if mean_speed < 20 else 0.1, 0.7 if ti < 0.2 else 0.4, 1.0 - (stress_index/100)]
ax[1].bar(methods, scores, color=['#2c3e50', '#d35400', '#27ae60'])
ax[1].set_title("Reliability Score (Higher is Safer)")
ax[1].set_ylim(0, 1.1)

st.pyplot(fig)

# --- STATUS BOX ---
if stress_index > 80 or mean_speed > 22:
    st.error("🚨 EMERGENCY: IEC 61400-1 Violation. Structural integrity compromised.")
elif vague_pi > 0.4:
    st.warning("⚠️ WARNING: High Vague Uncertainty. AI suggests early maintenance.")
else:
    st.success("✅ SYSTEM STABLE: Operating within safe structural limits.")