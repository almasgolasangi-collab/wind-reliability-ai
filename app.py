import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import zipfile

# --- 1. MACHINE SPECIFICATIONS (Integrated to prevent ModuleNotFoundError) ---
def show_machine_specs():
    st.sidebar.header("⚙️ Machine Specifications")
    st.sidebar.write("**Gearbox Type:** 3-Stage (1 Planetary + 2 Helical)")
    st.sidebar.write("**Lubrication:** ISO VG 320 Synthetic")
    st.sidebar.write("**Reference:** NREL Gearbox Reliability Collaborative")
    st.sidebar.divider()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Wind Reliability AI Pipeline", layout="wide")
st.title("🛡️ Wind Turbine Reliability: Triple-Model Pipeline")
st.markdown("Developed for NREL Benchmark Analysis and Reliability Engineering.")
st.divider()

# --- STEP 1: DATA PROCESSING (Acquisition & Cleansing) ---
st.header("Step 1: Data Processing")
show_machine_specs()

col_u1, col_u2 = st.columns(2)
with col_u1:
    st.subheader("A. Environmental Load")
    weather_file = st.file_uploader("Upload weather.csv.zip", type=["zip", "csv"])
with col_u2:
    st.subheader("B. Maintenance Logs")
    failure_file = st.file_uploader("Upload component_failures.csv", type=["csv"])

if weather_file and failure_file:
    try:
        # Load Weather Data
        if weather_file.name.endswith('.zip'):
            with zipfile.ZipFile(weather_file) as z:
                target = [f for f in z.namelist() if 'weather.csv' in f][0]
                with z.open(target) as f: weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)
        
        # Load Failure Data
        fail_df = pd.read_csv(failure_file)
        
        # Cleansing & Normalization
        weather_df.columns = weather_df.columns.str.strip()
        weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
        weather_2018 = weather_df[weather_df['Date'].dt.year == 2018].copy()
        
        fail_df.columns = fail_df.columns.str.strip()
        fail_df['Date'] = pd.to_datetime(fail_df['Date'])
        
        st.success("✅ Step 1 Complete: Data normalized and temporal filtering applied.")
    except Exception as e:
        st.error(f"Processing Error: {e}")

    # --- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---
    st.divider()
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    
    wind_col = 'Wind speed (m/s)'
    v_mean = weather_2018[wind_col].mean()
    v_std = weather_2018[wind_col].std()
    
    e1, e2 = st.columns(2)
    with e1:
        st.write("**Wind Distribution (Load Profile)**")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(weather_2018[wind_col], bins=30, kde=True, color='teal', ax=ax)
        ax.axvline(v_mean, color='red', linestyle='--', label=f'Mean: {v_mean:.2f}')
        ax.legend()
        st.pyplot(fig)
    
    with e2:
        st.write("**Failure Event Correlation**")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(weather_2018['Date'], weather_2018[wind_col], color='lightgray', alpha=0.5)
        for d in fail_df['Date']:
            ax2.axvline(d, color='red', linestyle='--', alpha=0.8)
        ax2.set_ylabel("m/s")
        st.pyplot(fig2)

    # --- STEP 3: DATA MODELING (THE TRIPLE METHOD) ---
    st.divider()
    st.header("Step 3: Data Modeling")
    
    # 1. FTA (Deterministic)
    rel_fta = (1 - (len(fail_df) / 365)) * 100
    
    # 2. Monte Carlo (Probabilistic)
    rel_mc = norm.cdf(25, v_mean, v_std) * 100
    
    # 3. Markov Chain (Stochastic) - Reliability R(t) = exp(-lambda * t)
    lambda_rate = len(fail_df) / 365
    rel_markov = np.exp(-lambda_rate * 1) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Fault Tree (FTA)", f"{rel_fta:.2f}%")
    m2.metric("Monte Carlo", f"{rel_mc:.2f}%")
    m3.metric("Markov Chain", f"{rel_markov:.2f}%")

    # --- STEP 4: DATA ANALYSIS & BENCHMARKING ---
    st.divider()
    st.header("Step 4: Data Analysis (Model Comparison)")
    
    # Bar Graph Comparison
    models = ['Fault Tree', 'Monte Carlo', 'Markov Chain']
    scores = [rel_fta, rel_mc, rel_markov]
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    sns.barplot(x=models, y=scores, palette=colors, ax=ax_bar)
    ax_bar.set_ylim(min(scores)-5, 101)
    ax_bar.set_ylabel("Reliability (%)")
    
    # Adding text labels on bars
    for i, v in enumerate(scores):
        ax_bar.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
    
    st.pyplot(fig_bar)

    # --- STEP 5: DATA EVALUATION ---
    st.divider()
    st.header("Step 5: Data Evaluation")
    
    # Check NREL Benchmark for Gearbox
    gb_only = fail_df[fail_df['Component'] == 'Gearbox']
    if not gb_only.empty:
        your_bearings = len(gb_only[gb_only['Failure_Mode'].str.contains('Bearing', case=False)])
        bearing_pct = (your_bearings / len(gb_only)) * 100
        
        st.write(f"**Gearbox Bearing Failure Ratio:** {bearing_pct:.1f}%")
        st.caption("Target: NREL GRD Benchmark is ~76%.")
    
    final_status = "STABLE" if rel_fta > 95 else "CRITICAL"
    st.subheader(f"System Health Status: :{ 'green' if final_status == 'STABLE' else 'red' }[{final_status}]")

else:
    st.info("Upload your Weather ZIP and Component Failure CSV to initialize the pipeline.")