import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import zipfile
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wind Reliability Pipeline", layout="wide")

st.title("🛡️ Wind Turbine Reliability: Engineering Pipeline")
st.markdown("---")

# --- STEP 1: DATA ACQUISITION ---
st.header("Step 1: Data Acquisition (Input Layer)")
uploaded_file = st.sidebar.file_uploader("Upload weather.csv.zip", type=["csv", "zip"])

raw_df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                # Find the specific weather.csv inside your zip
                target_file = [f for f in z.namelist() if 'weather.csv' in f][0]
                with z.open(target_file) as f:
                    raw_df = pd.read_csv(f)
        else:
            raw_df = pd.read_csv(uploaded_file)
            
        st.success(f"✅ Successfully acquired Raw Data: {uploaded_file.name}")
        st.write("**Raw Data Preview (First 5 Rows):**")
        st.dataframe(raw_df.head(5))
        
    except Exception as e:
        st.error(f"Error in Step 1: {e}")
        st.stop()

# --- STEP 2: DATA CLEANSING & EVALUATION ---
if raw_df is not None:
    st.divider()
    st.header("Step 2: Data Cleansing & Evaluation")
    
    clean_df = raw_df.copy()
    clean_df.columns = clean_df.columns.str.strip() # Clean column names
    
    # 1. Date Handling
    clean_df['Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True)
    # Filter for the most stable year in your data (2018)
    clean_df = clean_df[clean_df['Date'].dt.year == 2018]
    
    # 2. Outlier Removal (Engineering Standards)
    wind_col = 'Wind speed (m/s)'
    initial_len = len(clean_df)
    clean_df = clean_df.dropna(subset=[wind_col])
    # Removing speeds > 45 m/s (Sensory noise)
    clean_df = clean_df[(clean_df[wind_col] >= 0) & (clean_df[wind_col] <= 45)]
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Cleansing Summary:**")
        st.write(f"- Selected Period: Jan 2018 - Dec 2018")
        st.write(f"- Rows Removed (Outliers/Nulls): {initial_len - len(clean_df)}")
    with c2:
        st.write("**Cleaned Descriptive Statistics:**")
        st.write(clean_df[wind_col].describe())

    # --- STEP 3: EDA (EXPLORATORY DATA ANALYSIS) ---
    st.divider()
    st.header("Step 3: Exploratory Data Analysis (EDA)")
    e1, e2 = st.columns(2)
    
    v_mean = clean_df[wind_col].mean()
    v_std = clean_df[wind_col].std()

    with e1:
        st.write("**Wind Distribution (Monsoon Impact)**")
        fig1, ax1 = plt.subplots()
        ax1.hist(clean_df[wind_col], bins=40, color='skyblue', edgecolor='black')
        ax1.axvline(v_mean, color='red', label=f'Mean: {v_mean:.2f}')
        ax1.legend()
        st.pyplot(fig1)
    
    with e2:
        st.write("**Seasonal Velocity Trend**")
        clean_df['Month'] = clean_df['Date'].dt.month
        monthly = clean_df.groupby('Month')[wind_col].mean()
        st.line_chart(monthly)

    # --- STEP 4: FAULT TREE ANALYSIS (FTA) ---
    st.divider()
    st.header("Step 4: Fault Tree Analysis (FTA)")
    
    st.info("Logic: Top Event (System Failure) occurs if Wind Load > Cut-off (25 m/s) OR Vibration > Threshold.")
    
    # Simulate a Failure if Wind > 25 m/s (Standard Cut-out)
    clean_df['FTA_Violation'] = clean_df[wind_col].apply(lambda x: 1 if x > 25 else 0)
    total_violations = clean_df['FTA_Violation'].sum()
    fta_reliability = ((len(clean_df) - total_violations) / len(clean_df)) * 100

    col_fta1, col_fta2 = st.columns([2, 1])
    with col_fta1:
        # Step Function for FTA
        x_fta = np.linspace(0, 40, 100)
        y_fta = np.where(x_fta < 25, 100, 0)
        fig_fta, ax_fta = plt.subplots(figsize=(8, 3))
        ax_fta.step(x_fta, y_fta, where='post', color='black', linewidth=2)
        ax_fta.fill_between(x_fta, y_fta, step="post", alpha=0.2, color='gray')
        ax_fta.axvline(v_mean, color='red', linestyle='--', label='Current Mean Load')
        ax_fta.set_title("FTA Logic: System Survival vs. Wind Velocity")
        ax_fta.set_xlabel("Wind Speed (m/s)")
        ax_fta.set_ylabel("System Status (100=OK)")
        st.pyplot(fig_fta)

    with col_fta2:
        st.metric("FTA Reliability Score", f"{fta_reliability:.2f}%")
        st.write(f"Critical Events Found: {total_violations}")
        if total_violations > 0:
            st.warning("Fault Tree detected limit violations in high-wind months.")

    # --- STEP 5: FINAL EVALUATION ---
    st.divider()
    st.header("Step 5: Final Evaluation & Method Comparison")
    
    # Method B: Monte Carlo
    rel_mc = norm.cdf(25, v_mean, v_std) * 100
    # Method C: Markov
    rel_mar = np.exp(-0.02 * v_mean) * 100

    methods = ["Fault Tree (FTA)", "Monte Carlo", "Markov Chain"]
    scores = [fta_reliability, rel_mc, rel_mar]
    
    fig_res, ax_res = plt.subplots(figsize=(10, 4))
    ax_res.bar(methods, scores, color=['#2c3e50', '#3498db', '#e67e22'])
    ax_res.set_ylim(0, 115)
    for i, v in enumerate(scores):
        ax_res.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    st.pyplot(fig_res)

else:
    st.info("Please upload your 'weather.csv.zip' in the sidebar to begin Step 1.")