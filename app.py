import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- PAGE SETUP ---
st.set_page_config(page_title="Wind AI Reliability", layout="wide")

# --- 1. LOGO & SIDEBAR ---
# This looks for 'logo.png' in your GitHub repo
try:
    st.sidebar.image("logo.png", use_container_width=True)
except:
    st.sidebar.title("⚡ Wind AI Systems")
    st.sidebar.write("*(Upload logo.png to GitHub to see logo)*")

st.sidebar.markdown("---")
st.sidebar.header("Data Input Control")

# Choose between Demo mode or Manual Input
input_choice = st.sidebar.radio("Input Method:", ["Slider (Fast Demo)", "Type List of Numbers (Custom Data)"])

if input_choice == "Type List of Numbers (Custom Data)":
    data_in = st.sidebar.text_input("Enter Wind Speeds (comma separated):", "14, 16, 15, 19, 17")
    try:
        # Converting the typed string into a list of numbers
        data = [float(x.strip()) for x in data_in.split(",")]
    except:
        st.sidebar.error("Invalid format! Use numbers like: 10, 12, 15")
        data = [14.8, 15.2, 14.5]
else:
    # Generative Mode (Based on IEC 61400-1 Logic)
    intensity = st.sidebar.select_slider("Environment Intensity:", options=["low", "medium", "high"], value="medium")
    if intensity == "high": 
        v_hub, i_ref = 22.5, 0.16
    elif intensity == "low": 
        v_hub, i_ref = 8.0, 0.12
    else: 
        v_hub, i_ref = 14.8, 0.14
    
    sigma = i_ref * (0.75 * v_hub + 5.6)
    data = np.random.normal(v_hub, sigma, 10)

# --- 2. CALCULATION ENGINE ---
avg_v = np.mean(data)
std_v = np.std(data, ddof=1) if len(data) > 1 else 0.1
ti = std_v / avg_v if avg_v > 0 else 0

# Stress Index (The physical risk)
stress_idx = np.clip(((avg_v**2)/(25**2)) + (ti * 1.5), 0, 1)
stress_lbl = "CRITICAL" if stress_idx > 0.75 else "MODERATE" if stress_idx > 0.4 else "LOW"

# Vague Set AI Parameters (t, f, pi)
pi = np.clip(ti * 2.2, 0.1, 0.45) # Hesitation/Uncertainty
t = stress_idx * (1 - pi)        # Truth membership
f = (1 - stress_idx) * (1 - pi)  # Falsehood membership

# Comparison reliability scores
rel_ft = 1.0 if avg_v < 20.0 else 0.0
rel_mar = 1.0 - np.clip(avg_v / 26.0, 0, 1)
rel_mc = 1.0 - stress_idx

# --- 3. DASHBOARD METRICS ---
st.title("⚡ IEC 61400-1: Multi-Method Reliability Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Wind Speed", f"{avg_v:.2f} m/s")
c2.metric("Turbulence (TI)", f"{ti:.2f}")
c3.metric("Stress Index", f"{stress_idx*100:.1f}%")
c4.metric("Vague Uncertainty (π)", f"{pi:.2f}")

# --- 4. THE GRAPHING (PROFESSIONAL GRIDSPEC) ---
fig = plt.figure(figsize=(15, 9), facecolor='#f0f2f5')
gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# PANEL 1: FUSION MAP (Visualizing Uncertainty)
alphas = np.linspace(0, 1, 20)
ax1.fill_betweenx(alphas, t*alphas, (1-f)+(f*(1-alphas)), color='gold', alpha=0.3, label='Vague Uncertainty (π)')
ax1.axvline(1-rel_ft, color='black', ls='--', label='Fault Tree Risk')
ax1.axvline(1-rel_mar, color='darkorange', ls=':', label='Markov Risk')
ax1.axvline(stress_idx, color='blue', label='AI Risk Index', lw=3)
ax1.set_title("INTELLIGENCE FUSION MAP", fontweight='bold')
ax1.set_xlabel("Risk Probability (0 to 1)")
ax1.set_xlim(-0.05, 1.05)
ax1.legend()

# PANEL 2: RELIABILITY COMPARISON (Bar Chart)
bars = ax2.bar(['Fault Tree', 'Markov Chain', 'Monte Carlo AI'], [rel_ft, rel_mar, rel_mc], 
                color=['#34495e', '#e67e22', '#2ecc71'], edgecolor='black')
ax2.set_title("RELIABILITY COMPARISON", fontweight='bold')
ax2.set_ylabel("Reliability Score")
ax2.set_ylim(0, 1.2)
for b in bars:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.2f}", ha='center', fontweight='bold')

# PANEL 3: AUTONOMOUS DECISION LAYER
ax3.set_axis_off()
box = dict(boxstyle='round,pad=1', facecolor='white', edgecolor='#bdc3c7')

# Decision Logic
d_ft = ("SAFE", "green") if rel_ft > 0.5 else ("HALT", "red")
d_mar = ("OPTIMAL", "green") if rel_mar > 0.6 else ("CAUTION", "orange")
d_mc = (f"AI: {stress_lbl}", "green" if rel_mc > 0.6 else "red")

ax3.text(0.15, 0.4, f"FAULT TREE:\n{d_ft[0]}", color=d_ft[1], fontsize=12, fontweight='bold', ha='center', bbox=box)
ax3.text(0.50, 0.4, f"MARKOV CHAIN:\n{d_mar[0]}", color=d_mar[1], fontsize=12, fontweight='bold', ha='center', bbox=box)
ax3.text(0.85, 0.4, f"VAGUE AI ENGINE:\n{d_mc[0]}", color=d_mc[1], fontsize=12, fontweight='bold', ha='center', 
         bbox=dict(box, edgecolor=d_mc[1], linewidth=3))

st.pyplot(fig)