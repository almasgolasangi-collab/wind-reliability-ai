import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wind Reliability AI", layout="wide")

st.title("⚡ IEC 61400-1 Wind Reliability Dashboard")
st.markdown("""
This dashboard compares **Traditional Fault Trees** (Binary) with **Generative AI** (Vague Monte Carlo) 
to predict wind turbine structural failure risks based on IEC standards.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Control Parameters")
intensity = st.sidebar.select_slider(
    "Select Environment Intensity",
    options=["Low", "Medium", "High"],
    value="Medium"
)

manual_override = st.sidebar.checkbox("Manual Wind Speed Override")
if manual_override:
    avg_v = st.sidebar.slider("Mean Wind Speed (m/s)", 5.0, 30.0, 14.5)
    std_v = st.sidebar.slider("Standard Deviation (Turbulence)", 0.5, 5.0, 2.0)
else:
    # IEC Data Generation Logic
    if intensity == "High": v_hub, i_ref = 22.5, 0.16
    elif intensity == "Low": v_hub, i_ref = 8.0, 0.12
    else: v_hub, i_ref = 14.8, 0.14
    
    sigma = i_ref * (0.75 * v_hub + 5.6)
    data = np.random.normal(v_hub, sigma, 10)
    avg_v, std_v = np.mean(data), np.std(data)

# --- MATH ENGINE ---
ti = std_v / avg_v
stress_idx = np.clip(((avg_v**2)/(25**2)) + (ti * 1.5), 0, 1)
stress_lbl = "CRITICAL" if stress_idx > 0.75 else "MODERATE" if stress_idx > 0.4 else "LOW"

# AI/Vague Logic
mid = stress_idx
pi = np.clip(ti * 2.2, 0.1, 0.45)
L, U = np.clip(mid - 0.12, 0, 1), np.clip(mid + 0.12, 0, 1)
t = mid * (1 - pi)
f = (1 - mid) * (1 - pi)

# Comparison Methods
ft_risk = 1.0 if avg_v >= 20.0 else 0.0 # 0 if safe, 1 if broken
mar_risk = np.clip(avg_v / 26.0, 0, 1)
ai_risk = mid

# Reliability (1 - Risk)
rel_ft, rel_mar, rel_ai = 1-ft_risk, 1-mar_risk, 1-ai_risk

# --- METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Wind Speed", f"{avg_v:.2f} m/s")
col2.metric("Turbulence (TI)", f"{ti:.2f}")
col3.metric("Stress Index", f"{stress_idx*100:.1f}%")
col4.metric("Vague Uncertainty (π)", f"{pi:.2f}")

st.divider()

# --- VISUALIZATION ---
fig = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Plot 1: Intelligence Fusion
alphas = np.linspace(0, 1, 20)
ax1.fill_betweenx(alphas, t*alphas, (1-f)+(f*(1-alphas)), color='gold', alpha=0.3, label='Vague Uncertainty')
ax1.fill_betweenx(alphas, L+alphas*(M-L) if 'M' in locals() else L+alphas*(mid-L), U-alphas*(U-mid), color='blue', alpha=0.1, label='Monte Carlo Zone')
ax1.axvline(ft_risk, color='black', ls='--', label='Fault Tree Risk')
ax1.axvline(mar_risk, color='darkorange', ls=':', label='Markov Risk')
ax1.axvline(ai_risk, color='blue', label='AI Risk Index')
ax1.set_title("Intelligence Fusion Map")
ax1.set_xlim(-0.05, 1.05)
ax1.legend()

# Plot 2: Reliability Comparison
ax2.bar(['Fault Tree', 'Markov', 'AI'], [rel_ft, rel_mar, rel_ai], color=['#34495e', '#e67e22', '#2ecc71'])
ax2.set_title("Reliability Score (0.0 - 1.0)")
ax2.set_ylim(0, 1.1)

st.pyplot(fig)

# --- AUTONOMOUS DECISION LAYER ---
st.subheader("🤖 Autonomous Decision Layer")
d_col1, d_col2, d_col3 = st.columns(3)

with d_col1:
    st.info("**Fault Tree Logic**")
    status = "✅ SAFE" if rel_ft > 0.5 else "🚨 HALT"
    st.write(f"Status: {status}")
    st.caption("Logic: Binary Limit Crossing")

with d_col2:
    st.info("**Markov Logic**")
    status = "✅ OPTIMAL" if rel_mar > 0.75 else "⚠️ CAUTION" if rel_mar > 0.4 else "🚨 SHUTDOWN"
    st.write(f"Status: {status}")
    st.caption("Logic: Probability State Transition")

with d_col3:
    st.success("**Monte Carlo AI Logic**")
    status = "✅ SAFE" if rel_ai > 0.8 else "⚠️ WARNING" if rel_ai > 0.5 else "🚨 EMERGENCY"
    st.write(f"Status: {status} ({stress_lbl} STRESS)")
    st.caption("Logic: Generative Uncertainty Fusion")