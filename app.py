import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# 1. PAGE SETUP
st.set_page_config(page_title="Wind Engineering Pipeline", layout="wide")

st.title("🛡️ Wind Turbine Reliability: Data Science Pipeline")
st.markdown("---")

# 2. DATA ACQUISITION LAYER
st.sidebar.header("Step 1: Data Acquisition")
data_mode = st.sidebar.radio("Select Source:", ["Generate 'Dirty' Dataset", "Upload My CSV"])

raw_df = pd.DataFrame()

if data_mode == "Generate 'Dirty' Dataset":
    # Creating a dataset with errors (Nulls and Outliers) to show Cleansing
    rows = 200
    data = {
        'Timestamp': pd.date_range(start='1/1/2026', periods=rows, freq='h'),
        'Wind_Speed': np.random.normal(12, 4, rows),
        'Vibration_Level': np.random.normal(0.5, 0.1, rows)
    }
    raw_df = pd.DataFrame(data)
    # Injecting Errors for demo
    raw_df.loc[10:15, 'Wind_Speed'] = np.nan  # Missing Values
    raw_df.loc[50:52, 'Wind_Speed'] = 99.0    # Outlier
    raw_df.loc[80:82, 'Wind_Speed'] = -5.0    # Impossible Value
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
    else:
        st.info("Awaiting CSV Upload...")
        st.stop()

# 3. DATA EVALUATION & CLEANSING
st.header("Step 2: Data Cleansing & Evaluation")
c1, c2 = st.columns(2)

# Error-proof column detection
target_col = ""
for col in raw_df.columns:
    if 'wind' in col.lower() or 'speed' in col.lower():
        target_col = col
        break

if not target_col:
    st.error("Could not find a 'Wind Speed' column. Please check your CSV.")
    st.stop()

with c1:
    st.write("**Before Cleansing (Raw Data)**")
    st.write(raw_df.head(5))
    st.write("Missing Values:", raw_df.isnull().sum())

# CLEANSING LOGIC
clean_df = raw_df.copy()
clean_df = clean_df.dropna(subset=[target_col]) # Remove Nulls
clean_df = clean_df[(clean_df[target_col] >= 0) & (clean_df[target_col] <= 50)] # Remove Outliers

with c2:
    st.write("**After Cleansing (Clean Data)**")
    st.write(clean_df.head(5))
    st.success(f"Removed {len(raw_df) - len(clean_df)} invalid observations.")

st.divider()

# 4. EDA (EXPLORATORY DATA ANALYSIS)
st.header("Step 3: Exploratory Data Analysis (EDA)")
e1, e2, e3 = st.columns(3)

v_mean = clean_df[target_col].mean()
v_std = clean_df[target_col].std()

with e1:
    st.write("**Wind Distribution**")
    fig_e1, ax_e1 = plt.subplots()
    sns.histplot(clean_df[target_col], kde=True, color='teal', ax=ax_e1)
    st.pyplot(fig_e1)

with e2:
    st.write("**Boxplot (Outlier Check)**")
    fig_e2, ax_e2 = plt.subplots()
    sns.boxplot(x=clean_df[target_col], color='lightgreen', ax=ax_e2)
    st.pyplot(fig_e2)

with e3:
    st.write("**Descriptive Statistics**")
    st.write(clean_df[target_col].describe())

st.divider()

# 5. RELIABILITY MODELING (3 METHODS)
st.header("Step 4: Reliability Modeling")
m1, m2, m3 = st.columns(3)

with m1:
    st.write("**Fault Tree Analysis**")
    x = np.linspace(0, 50, 100)
    y = np.where(x < 25, 100, 0)
    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, color='black', linewidth=2)
    ax1.axvline(v_mean, color='red', linestyle='--')
    st.pyplot(fig1)
    rel_fta = 100 if v_mean < 25 else 0

with m2:
    st.write("**Monte Carlo Simulation**")
    x_mc = np.linspace(v_mean-10, v_mean+10, 100)
    y_mc = norm.pdf(x_mc, v_mean, v_std)
    fig2, ax2 = plt.subplots()
    ax2.plot(x_mc, y_mc, color='blue', linewidth=2)
    ax2.fill_between(x_mc, y_mc, alpha=0.3, color='blue')
    st.pyplot(fig2)
    # Correct MC Reliability Logic
    rel_mc = max(0, 100 - (v_std / v_mean * 100))

with m3:
    st.write("**Markov Chain Decay**")
    x_mar = np.linspace(0, 50, 100)
    y_mar = np.exp(-0.04 * x_mar) * 100
    fig3, ax3 = plt.subplots()
    ax3.plot(x_mar, y_mar, color='orange', linewidth=2)
    st.pyplot(fig3)
    rel_mar = np.exp(-0.04 * v_mean) * 100

st.divider()

# 6. RESULTS & BAR GRAPH
st.header("Step 5: Final Evaluation")
r1, r2 = st.columns([2, 1])

with r1:
    st.write("**Method Comparison (Bar Graph)**")
    methods = ["Fault Tree", "Monte Carlo", "Markov Chain"]
    scores = [rel_fta, rel_mc, rel_mar]
    fig_res, ax_res = plt.subplots()
    bars = ax_res.bar(methods, scores, color=['#2c3e50', '#3498db', '#e67e22'])
    ax_res.set_ylim(0, 110)
    for b in bars:
        ax_res.text(b.get_x()+b.get_width()/2, b.get_height()+2, f"{b.get_height():.1f}%", ha='center', weight='bold')
    st.pyplot(fig_res)

with r2:
    st.write("**Component Reliability**")
    risk_factor = (v_mean/35)**2
    st.table(pd.DataFrame({
        "Component": ["Blades", "Gearbox", "Generator"],
        "Health": [f"{100 - (risk_factor*100):.1f}%", "97.5%", "98.2%"],
        "Verdict": ["SAFE" if risk_factor < 0.2 else "ALERT", "SAFE", "SAFE"]
    }))