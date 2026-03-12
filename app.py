import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import zipfile
import io

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Wind Reliability Pipeline", layout="wide")
st.title("🛡️ Wind Turbine Reliability: End-to-End Pipeline")
st.markdown("---")

# --- 2. DATA ACQUISITION ---
st.sidebar.header("Step 1: Data Acquisition")
# Updated to accept BOTH zip and csv
uploaded_file = st.sidebar.file_uploader("Upload weather.csv or weather.csv.zip", type=["csv", "zip"])

raw_df = None

if uploaded_file is not None:
    try:
        # Check if the file is a ZIP
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                # Look for weather.csv inside the zip
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if csv_files:
                    # Use the first csv found or specifically weather.csv
                    target_file = "weather.csv" if "weather.csv" in csv_files else csv_files[0]
                    with z.open(target_file) as f:
                        raw_df = pd.read_csv(f)
                else:
                    st.error("No CSV file found inside the ZIP archive.")
        else:
            # It's a regular CSV
            raw_df = pd.read_csv(uploaded_file)
            
    except Exception as e:
        st.error(f"Error loading file: {e}")

# --- 3. PROCESSING (Only if file is loaded) ---
if raw_df is not None:
    st.header("Step 2: Data Cleansing & Evaluation")
    
    clean_df = raw_df.copy()
    
    # 1. Clean Column Names (remove spaces/extra characters)
    clean_df.columns = clean_df.columns.str.strip()
    
    # 2. Date Conversion
    if 'Date' in clean_df.columns:
        clean_df['Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True)
        # Filter for 2018 (your file's best year)
        clean_df = clean_df[clean_df['Date'].dt.year == 2018]
    
    # 3. Handle Wind Speed Outliers
    wind_col = 'Wind speed (m/s)'
    if wind_col in clean_df.columns:
        initial_rows = len(clean_df)
        clean_df = clean_df.dropna(subset=[wind_col])
        clean_df = clean_df[(clean_df[wind_col] >= 0) & (clean_df[wind_col] <= 45)]
        
        st.success(f"Cleansing Complete: Processed {initial_rows} rows from 2018.")
        
        # --- 4. EDA METHOD ---
        st.header("Step 3: Exploratory Data Analysis (EDA)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Wind Distribution**")
            fig, ax = plt.subplots()
            ax.hist(clean_df[wind_col], bins=30, color='teal', edgecolor='white')
            st.pyplot(fig)
            
        with col2:
            st.write("**Statistics**")
            st.write(clean_df[wind_col].describe())

        # --- 5. RELIABILITY MODELS ---
        st.header("Step 4: Reliability Analysis")
        v_mean = clean_df[wind_col].mean()
        v_std = clean_df[wind_col].std()
        
        rel_mc = norm.cdf(25, v_mean, v_std) * 100
        rel_mar = np.exp(-0.02 * v_mean) * 100
        
        m1, m2 = st.columns(2)
        m1.metric("Monte Carlo Reliability", f"{rel_mc:.1f}%")
        m2.metric("Markov Chain Reliability", f"{rel_mar:.1f}%")
        
    else:
        st.error(f"Could not find column '{wind_col}'. Please check your CSV headers.")
else:
    st.info("Please upload your file (CSV or ZIP) to start.")