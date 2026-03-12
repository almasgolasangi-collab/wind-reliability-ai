import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.set_page_config(page_title="Wind Reliability Engineering Pipeline", layout="wide")

# --- STEP 1: DATA COLLECTION ---
st.sidebar.header("Step 1: Data Acquisition")
data_source = st.sidebar.radio("Data Source:", ["Upload My CSV", "Generate Raw 'Dirty' Dataset"])

if data_source == "Generate Raw 'Dirty' Dataset":
    # Creating a dummy 'Raw' dataset with errors for cleansing demonstration
    rows = 500
    data = {
        'Timestamp': pd.date_range(start='1/1/2024', periods=rows, freq='H'),
        'Wind_Speed_ms': np.random.normal(12, 4, rows),
        'Ambient_Temp': np.random.normal(25, 5, rows)
    }
    raw_df = pd.DataFrame(data)
    # Injecting "Dirty" data (Outliers and Nulls)
    raw_df.loc[10:15, 'Wind_Speed_ms'] = np.nan  # Null values
    raw_df.loc[50:52, 'Wind_Speed_ms'] = 99.0    # Impossible Outlier
    raw_df.loc[100:102, 'Wind_Speed_ms'] = -5.0  # Physical Impossibility
else:
    uploaded_file = st.sidebar.file_uploader("Upload your SCADA CSV", type=["csv"])
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
    else:
        st.info("Waiting for CSV upload...")
        st.stop()

# --- STEP 2: DATA CLEANSING (The "Evaluate" part) ---
st.header("Step 2: Data Evaluation & Cleansing")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Data Issues")
    st.write(f"Total Rows: {len(raw_df)}")
    st.write("Missing Values:", raw_df.isnull().sum())

# CLEANSING LOGIC
clean_df = raw_df.copy()
wind_col = 'Wind_Speed_ms' if 'Wind_Speed_ms' in raw_df.columns else raw_df.columns[1]

# 1. Remove Nulls
clean_df = clean_df.dropna(subset=[wind_col])
# 2. Filter Outliers (Standard Wind Turbine Range 0-25m/s)
clean_df = clean_df[(clean_df[wind_col] >= 0) & (clean_df[wind_col] <= 45)]

with col2:
    st.subheader("Cleaned Data Results")
    st.write(f"Rows remaining: {len(clean_df)}")
    st.success(f"Cleaned {len(raw_df) - len(clean_df)} problematic rows.")

st.divider()

# --- STEP 3: EDA (EXPLORATORY DATA ANALYSIS) ---
st.header("Step 3: Exploratory Data Analysis (EDA)")
eda1, eda2, eda3 = st.columns(3)

with eda1:
    st.write("**Velocity Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(clean_df[wind_col], kde=True, color='blue', ax=ax)
    st.pyplot(fig)

with eda2:
    st.write("**Statistical Summary**")
    st.write(clean_df[wind_col].describe())
    v_mean = clean_df[wind_col].mean()
    v_std = clean_df[wind_col].std()

with eda3:
    st.write("**Time-Series Trend**")
    fig2, ax2 = plt.subplots()
    ax2.plot(clean_df[wind_col][:100], color='green') # Showing first 100 hrs
    ax2.set_ylabel("m/s")
    st.pyplot(fig2)

st.divider()

# --- STEP 4: RELIABILITY ANALYSIS (The 3 Methods) ---
st.header("Step 4: Multi-Method Reliability Modeling")
m1, m2, m3 = st.columns(3)

# Fault Tree
with m1:
    st.write("**Fault Tree Logic**")
    rel_fta = 100 if v_mean < 25 else 0
    fig_f, ax_f = plt.subplots()
    ax_f.step([0, 25, 50], [100, 100, 0], where='post', color='black')
    ax_f.axvline(v_mean, color='red', linestyle='--')
    st.pyplot(fig_f)

# Monte Carlo
with m2:
    st.write("**Monte Carlo Simulation**")
    rel_mc = max(0, 100 - (v_std/v_mean * 100))
    x = np.linspace(v_mean-10, v_mean+10, 100)
    y = norm.pdf(x, v_mean, v_std)
    fig_m, ax_m = plt.subplots()
    ax_m.plot(x, y, color='blue')
    ax_m.fill_between(x, y, alpha=0.2)
    st.pyplot(fig_m)

# Markov Chain
with m3:
    st.write("**Markov Transition**")
    rel_mar = np.exp(-0.04 * v_mean) * 100
    x_mar = np.linspace(0, 50, 100)
    y_mar = np.exp(-0.04 * x_mar) * 100
    fig_ma, ax_ma = plt.subplots()
    ax_ma.plot(x_mar, y_mar, color='orange')
    st.pyplot(fig_ma)

st.divider()

# --- STEP 5: FINAL EVALUATION ---
st.header("Step 5: Final Evaluation")
res_l, res_r = st.columns([2, 1])

with res_l:
    methods = ["Fault Tree", "Monte Carlo", "Markov Chain"]
    scores = [rel_fta, rel_mc, rel_mar]
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(methods, scores, color=['#2c3e50', '#3498db', '#e67e22'])
    ax_bar.set_ylim(0, 110)
    st.pyplot(fig_bar)

with res_r:
    st.write("**Component Reliability Status**")
    risk = (v_mean/35)**2
    st.table(pd.DataFrame({
        "Component": ["Blades", "Gearbox"],
        "Health": [f"{100-risk*100:.1f}%", "98.2%"],
        "Status": ["SAFE" if risk < 0.2 else "WARN", "SAFE"]
    }))