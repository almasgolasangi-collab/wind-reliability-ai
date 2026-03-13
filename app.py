import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import zipfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wind Reliability Pipeline", layout="wide")
st.title("🛡️ Wind Turbine Reliability: End-to-End Pipeline")
st.markdown("---")

# --- STEP 1: DATA PROCESSING (Acquisition & Cleansing) ---
st.header("Step 1: Data Processing")
col_u1, col_u2 = st.columns(2)

with col_u1:
    st.subheader("A. Environmental Load")
    weather_file = st.file_uploader("Upload weather.csv.zip", type=["zip", "csv"])
with col_u2:
    st.subheader("B. Maintenance Logs")
    failure_file = st.file_uploader("Upload component_failures.csv", type=["csv"])

weather_df = None
fail_df = None

if weather_file and failure_file:
    # Processing Logic
    try:
        if weather_file.name.endswith('.zip'):
            with zipfile.ZipFile(weather_file) as z:
                target = [f for f in z.namelist() if 'weather.csv' in f][0]
                with z.open(target) as f: weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)
        
        fail_df = pd.read_csv(failure_file)
        
        # Cleansing
        weather_df.columns = weather_df.columns.str.strip()
        weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
        weather_2018 = weather_df[weather_df['Date'].dt.year == 2018].copy()
        
        st.success("✅ Data Processed: Normalization and Temporal Filtering Complete.")
    except Exception as e:
        st.error(f"Processing Error: {e}")

# --- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---
if weather_df is not None and fail_df is not None:
    st.divider()
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    
    wind_col = 'Wind speed (m/s)'
    v_mean = weather_2018[wind_col].mean()
    
    e1, e2 = st.columns(2)
    with e1:
        st.write("**Wind Distribution Modeling**")
        fig, ax = plt.subplots()
        ax.hist(weather_2018[wind_col], bins=50, color='teal', alpha=0.7)
        ax.set_title("Annual Velocity Frequency")
        st.pyplot(fig)
    with e2:
        st.write("**Event Correlation (Wind vs. Failures)**")
        fig2, ax2 = plt.subplots()
        ax2.plot(weather_2018['Date'], weather_2018[wind_col], color='lightgray')
        # Map failure dates
        fail_dates = pd.to_datetime(fail_df['Date'])
        for d in fail_dates:
            ax2.axvline(d, color='red', linestyle='--', alpha=0.6)
        st.pyplot(fig2)

    # --- STEP 3: DATA MODELING (FTA & Monte Carlo) ---
    st.divider()
    st.header("Step 3: Data Modeling")
    
    # Fault Tree Logic (Top Event)
    reliability_fta = (1 - (len(fail_df) / 365)) * 100
    
    # Monte Carlo (Gaussian Reliability)
    v_std = weather_2018[wind_col].std()
    rel_mc = norm.cdf(25, v_mean, v_std) * 100
    
    m1, m2 = st.columns(2)
    m1.metric("Fault Tree Reliability (Deterministic)", f"{reliability_fta:.2f}%")
    m2.metric("Monte Carlo Reliability (Probabilistic)", f"{rel_mc:.2f}%")

    # --- STEP 4: DATA ANALYSIS (NREL Benchmarking) ---
    st.divider()
    st.header("Step 4: Data Analysis (NREL GRD Comparison)")
    
    # Using percentages from NREL Gearbox Reliability Database (GRD) 
    st.info("Comparing site-specific data to NREL GRD Benchmark (76% Bearings, 17% Gears).")
    
    gb_only = fail_df[fail_df['Component'] == 'Gearbox']
    your_bearings = len(gb_only[gb_only['Failure_Mode'].str.contains('Bearing', case=False)])
    
    st.write(f"**Gearbox Bearing Contribution:** { (your_bearings/len(gb_only))*100 if len(gb_only)>0 else 0 :.1f}%")
    st.progress((your_bearings/len(gb_only)) if len(gb_only)>0 else 0)

    # --- STEP 5: DATA EVALUATION (Final Status) ---
    st.divider()
    st.header("Step 5: Data Evaluation")
    
    status = "STABLE" if reliability_fta > 95 else "CRITICAL"
    color = "green" if status == "STABLE" else "red"
    
    st.subheader(f"System Health Status: :{color}[{status}]")
    st.write(f"Based on the 2018 load profile, the system maintains a reliability of {reliability_fta:.2f}%.")

else:
    st.info("Please upload your files to initialize the pipeline.")