import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import zipfile
from specs import show_machine_specs  # Import your new specs file

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Wind Reliability Pipeline", layout="wide")
st.title("🛡️ Wind Turbine Reliability: End-to-End Pipeline")
st.markdown("This application implements a full Data Science pipeline for wind turbine reliability analysis.")
st.divider()

# --- STEP 1: DATA PROCESSING (Acquisition & Cleansing) ---
st.header("Step 1: Data Processing")
col_u1, col_u2 = st.columns(2)

with col_u1:
    st.subheader("A. Environmental Load (Weather)")
    weather_file = st.file_uploader("Upload weather.csv.zip", type=["zip", "csv"])

with col_u2:
    st.subheader("B. Maintenance Logs (Failures)")
    failure_file = st.file_uploader("Upload component_failures.csv", type=["csv"])

# CALL THE SPECS FUNCTION FROM specs.py
show_machine_specs()

weather_df = None
fail_df = None

if weather_file and failure_file:
    try:
        # Data Acquisition
        if weather_file.name.endswith('.zip'):
            with zipfile.ZipFile(weather_file) as z:
                target = [f for f in z.namelist() if 'weather.csv' in f][0]
                with z.open(target) as f: weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)
        
        fail_df = pd.read_csv(failure_file)
        
        # Data Cleansing & Normalization
        weather_df.columns = weather_df.columns.str.strip()
        weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
        weather_2018 = weather_df[weather_df['Date'].dt.year == 2018].copy()
        
        fail_df.columns = fail_df.columns.str.strip()
        fail_df['Date'] = pd.to_datetime(fail_df['Date'])
        
        st.success("✅ Step 1 Complete: Data normalized and filtered for 2018.")
    except Exception as e:
        st.error(f"Processing Error: {e}")

# --- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---
if 'weather_2018' in locals() and fail_df is not None:
    st.divider()
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    
    wind_col = 'Wind speed (m/s)'
    v_mean = weather_2018[wind_col].mean()
    v_std = weather_2018[wind_col].std()
    
    e1, e2 = st.columns(2)
    with e1:
        st.write("**Wind Speed Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(weather_2018[wind_col], bins=30, kde=True, color='teal', ax=ax)
        ax.axvline(v_mean, color='red', linestyle='--', label=f'Mean: {v_mean:.2f}')
        st.pyplot(fig)
        st.caption("Histogram showing the frequency of wind 'loads' on the turbine.")

    with e2:
        st.write("**Failure Event Timeline**")
        fig2, ax2 = plt.subplots()
        ax2.plot(weather_2018['Date'], weather_2018[wind_col], color='lightgray', alpha=0.5)
        for d in fail_df['Date']:
            ax2.axvline(d, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel("Wind Speed (m/s)")
        st.pyplot(fig2)
        st.caption("Red markers show when component failures occurred relative to wind speed.")

    # --- STEP 3: DATA MODELING (Reliability & FTA) ---
    st.divider()
    st.header("Step 3: Data Modeling")
    
    # Fault Tree Analysis (Deterministic)
    reliability_fta = (1 - (len(fail_df) / 365)) * 100
    
    # Probabilistic Modeling (Monte Carlo/Gaussian)
    rel_mc = norm.cdf(25, v_mean, v_std) * 100
    
    m1, m2 = st.columns(2)
    m1.metric("System Reliability (FTA)", f"{reliability_fta:.2f}%")
    m2.metric("Probabilistic Reliability", f"{rel_mc:.2f}%")

    # --- STEP 4: DATA ANALYSIS (NREL Benchmarking) ---
    st.divider()
    st.header("Step 4: Data Analysis (NREL GRD Benchmarking)")
    
    # Calculate Gearbox Failure Ratios based on user-uploaded NREL data 
    gb_only = fail_df[fail_df['Component'] == 'Gearbox']
    if not gb_only.empty:
        # Mapping specific modes found in NREL reports [cite: 5]
        your_bearings = len(gb_only[gb_only['Failure_Mode'].str.contains('Bearing', case=False)])
        bearing_pct = (your_bearings / len(gb_only)) * 100
        
        st.write(f"**Gearbox Bearing Failure Contribution:** {bearing_pct:.1f}%")
        st.progress(bearing_pct / 100)
        st.info("NREL Benchmark: Bearings typically represent ~76% of gearbox damage.")
    else:
        st.warning("No Gearbox failures recorded for benchmarking.")

    # --- STEP 5: DATA EVALUATION (Final Status) ---
    st.divider()
    st.header("Step 5: Data Evaluation")
    
    status = "STABLE" if reliability_fta > 95 else "CRITICAL"
    color = "green" if status == "STABLE" else "red"
    
    st.subheader(f"System Health Status: :{color}[{status}]")
    st.write(f"The turbine shows a {reliability_fta:.2f}% reliability for the 2018 operational period.")

else:
    st.info("Please upload the Weather ZIP and Component Failure CSV to begin the analysis.")