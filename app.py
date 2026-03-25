import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Wind Reliability AI Pipeline",
    page_icon="💨",
    layout="wide"
)

st.title("🛡️ Wind Turbine Reliability: Advanced Modeling Pipeline")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Controls")
wind_stress_factor = st.sidebar.slider("Wind Load Increase (%)", 0, 50, 0)
cut_in = st.sidebar.slider("Cut-in Wind Speed (m/s)", 1, 5, 3)
cut_out = st.sidebar.slider("Cut-out Wind Speed (m/s)", 15, 30, 20)

# --- FILE LOADER ---
def load_file(file):
    if file is None: return None
    name = file.name.lower()
    try:
        if name.endswith(".csv"): return pd.read_csv(file)
        elif name.endswith(".xlsx"): return pd.read_excel(file, engine="openpyxl")
        elif name.endswith(".txt"): return pd.read_csv(file, engine='python')
        elif name.endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                for f in z.namelist():
                    if f.endswith(".csv"): return pd.read_csv(z.open(f))
                    elif f.endswith(".xlsx"): return pd.read_excel(z.open(f), engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return None

# --- FILE UPLOAD ---
st.header("Step 1: Upload Data")
col_u1, col_u2 = st.columns(2)
with col_u1:
    weather_file = st.file_uploader("Upload Weather Data", type=["csv", "xlsx", "txt", "zip"])
with col_u2:
    failure_file = st.file_uploader("Upload Failure Data", type=["csv", "xlsx", "txt"])

if weather_file and failure_file:
    weather_df = load_file(weather_file)
    fail_df = load_file(failure_file)

    if weather_df is not None and fail_df is not None:
        try:
            # --- CLEANING & SORTING (The Fix for Jagged Lines) ---
            weather_df.columns = weather_df.columns.str.strip()
            fail_df.columns = fail_df.columns.str.strip()

            if 'Date' in weather_df.columns:
                weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')
                weather_df = weather_df.dropna(subset=['Date'])
                # CRITICAL: Sort by date so the line chart flows left-to-right
                weather_df = weather_df.sort_values(by='Date')
            else:
                st.error("Error: 'Date' column missing in Weather Data.")
                st.stop()

            has_fail_date = 'Date' in fail_df.columns
            if has_fail_date:
                fail_df['Date'] = pd.to_datetime(fail_df['Date'], errors='coerce')

            wind_col = 'Wind speed (m/s)'
            v_mean = weather_df[wind_col].mean()
            v_std = weather_df[wind_col].std()
            sim_mean = v_mean * (1 + wind_stress_factor / 100)

            # --- TABS ---
            tab1, tab2, tab3, tab4 = st.tabs(["🧹 Data", "📊 EDA", "🧬 Modeling", "🏁 Final"])

            with tab1:
                st.subheader("Data Preview")
                st.dataframe(weather_df.head(10))

            with tab2:
                st.subheader("📊 Exploratory Data Analysis")
                
                # Plotting Time Series
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                
                # OPTIONAL: Smoothening (Daily Mean) to make the graph cleaner
                # If your data is hourly, this removes the "noise"
                weather_daily = weather_df.set_index('Date')[wind_col].resample('D').mean()
                
                ax3.plot(weather_daily.index, weather_daily.values, label='Avg Daily Wind Speed', color='#1f77b4', alpha=0.8)
                
                if has_fail_date:
                    for d in fail_df['Date'].dropna():
                        ax3.axvline(d, color='red', linestyle='--', alpha=0.4)
                    st.info("💡 Red dashed lines indicate failure timestamps.")
                
                ax3.set_title("Wind Speed Trends (Sorted Chronologically)")
                ax3.set_ylabel("m/s")
                plt.xticks(rotation=45)
                st.pyplot(fig3)

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    sns.histplot(weather_df[wind_col], bins=30, kde=True, ax=ax)
                    st.pyplot(fig)
                with c2:
                    fig2, ax2 = plt.subplots()
                    sns.boxplot(x=weather_df[wind_col], ax=ax2)
                    st.pyplot(fig2)

            with tab3:
                st.subheader("🧬 Reliability Models")
                
                # Monte Carlo Simulation
                n_sim = 10000
                samples = np.random.weibull(2, n_sim) * sim_mean
                samples += np.random.normal(0, v_std * 1.5, n_sim)
                rel_mc = (np.sum((samples >= cut_in) & (samples <= cut_out)) / n_sim) * 100
                
                # Simplified FTA
                p_wind_fault = np.mean((weather_df[wind_col] > cut_out) | (weather_df[wind_col] < cut_in))
                rel_fta = (1 - (p_wind_fault * 0.1)) * 100 # Assuming 10% fault chance on extreme wind

                m1, m2 = st.columns(2)
                m1.metric("FTA Reliability", f"{rel_fta:.2f}%")
                m2.metric("Monte Carlo Reliability", f"{rel_mc:.2f}%")

            with tab4:
                st.subheader("🏁 Final Evaluation")
                lolp = np.mean((samples < cut_in) | (samples > cut_out)) * 100
                st.metric("Loss of Load Probability (LOLP)", f"{lolp:.2f}%")
                
                status = "🟢 STABLE" if rel_mc > 80 else "🔴 CRITICAL"
                st.header(f"System Status: {status}")

        except Exception as e:
            st.error(f"Processing Error: {e}")
else:
    st.info("Waiting for Weather and Failure data files...")