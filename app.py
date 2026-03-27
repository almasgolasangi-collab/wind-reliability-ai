import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Wind Reliability", layout="wide")
st.title("🛡️ Wind Turbine Reliability Analysis")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙️ Controls")
wind_stress_factor = st.sidebar.slider("Wind Stress Increase (%)", 0, 50, 0)
mission_time = st.sidebar.slider("Mission Time (days)", 1, 60, 30)

# ---------------------------
# FILE UPLOAD
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    weather_file = st.file_uploader("Upload Wind Data", type=["csv", "zip"])

with col2:
    failure_file = st.file_uploader("Upload Failure Data", type=["csv"])

# ---------------------------
# MAIN
# ---------------------------
if weather_file and failure_file:
    try:
        # LOAD WEATHER
        if weather_file.name.endswith(".zip"):
            with zipfile.ZipFile(weather_file) as z:
                file_name = [f for f in z.namelist() if f.endswith(".csv")][0]
                with z.open(file_name) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        # CLEAN COLUMN NAMES
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        # ---------------------------
        # DATE DETECTION
        # ---------------------------
        def get_date(df):
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    return col
            st.error("❌ No Date column found")
            st.stop()

        weather_date = get_date(weather_df)
        fail_date = get_date(fail_df)

        weather_df[weather_date] = pd.to_datetime(weather_df[weather_date], dayfirst=True, errors='coerce')
        fail_df[fail_date] = pd.to_datetime(fail_df[fail_date], dayfirst=True, errors='coerce')

        weather_df.dropna(subset=[weather_date], inplace=True)
        fail_df.dropna(subset=[fail_date], inplace=True)

        # ---------------------------
        # WIND COLUMN DETECTION
        # ---------------------------
        def get_wind(df):
            for col in df.columns:
                if col.upper() in ["WS10M", "WS50M"]:
                    return col
                if "wind" in col.lower():
                    return col
            st.error("❌ No wind column found")
            st.stop()

        wind_col = get_wind(weather_df)
        weather_df.rename(columns={wind_col: "Wind"}, inplace=True)

        # ---------------------------
        # YEAR FILTER
        # ---------------------------
        year = st.selectbox("Select Year", sorted(weather_df[weather_date].dt.year.unique()))

        weather_year = weather_df[weather_df[weather_date].dt.year == year]
        fail_year = fail_df[fail_df[fail_date].dt.year == year]

        # SORT + RESAMPLE
        weather_year = weather_year.sort_values(by=weather_date)
        weather_year = weather_year.set_index(weather_date)

        daily_wind = weather_year["Wind"].resample('D').mean()

        # ---------------------------
        # BASIC STATS
        # ---------------------------
        total_days = len(daily_wind)
        total_hours = len(weather_year)
        failures = len(fail_year)

        # ---------------------------
        # TABS
        # ---------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧹 Cleaning",
            "⚙️ Processing",
            "📊 EDA",
            "📈 Visualization",
            "🧬 Modeling"
        ])

        # CLEANING
        with tab1:
            st.subheader("Data Cleaning")
            st.dataframe(weather_df.head())
            st.dataframe(fail_df.head())

        # PROCESSING
        with tab2:
            st.subheader("Data Processing")
            st.write(f"Year: {year}")
            st.write(f"Total Days: {total_days}")
            st.write(f"Total Hours: {total_hours}")
            st.write(f"Failures: {failures}")

        # ---------------------------
        # EDA
        # ---------------------------
        with tab3:
            st.subheader("EDA")

            v_mean = daily_wind.mean()
            v_std = daily_wind.std()

            st.write(f"Mean Wind Speed: {v_mean:.2f}")
            st.write(f"Std Dev: {v_std:.2f}")

            fig1, ax1 = plt.subplots()
            sns.histplot(daily_wind, bins=30, kde=True, ax=ax1)
            ax1.set_title("Wind Speed Distribution")
            st.pyplot(fig1)

        # ---------------------------
        # VISUALIZATION
        # ---------------------------
        with tab4:
            st.subheader("Wind vs Failures")

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(daily_wind.index, daily_wind.values)

            fail_days = pd.to_datetime(fail_year[fail_date]).dt.date
            for d in fail_days:
                ax2.axvline(pd.to_datetime(d), alpha=0.3)

            ax2.set_title("Wind Speed vs Failures")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Wind Speed")

            st.pyplot(fig2)

        # ---------------------------
        # MODELING (FINAL FIX)
        # ---------------------------
        with tab5:
            st.subheader("Reliability Modeling")

            if total_days == 0:
                st.error("No data available")
                st.stop()

            # ✅ FAILURE RATE (per day)
            min_lambda = 1 / (total_days * 2)
            lambda_rate = max(failures / total_days, min_lambda)

            # Repair rate (3 days repair time)
            mu = 1 / 3

            # Mission time in DAYS
            t = mission_time

            # ---------------------------
            # FTA (Exponential Reliability)
            # ---------------------------
            rel_fta = np.exp(-lambda_rate * t) * 100

            # ---------------------------
            # Markov Model
            # ---------------------------
            rel_markov = (
                (mu / (lambda_rate + mu)) +
                (lambda_rate / (lambda_rate + mu)) * np.exp(-(lambda_rate + mu) * t)
            ) * 100

            # ---------------------------
            # Monte Carlo (Improved)
            # ---------------------------
            sim_mean = v_mean * (1 + wind_stress_factor / 100)
            samples = np.random.normal(sim_mean, v_std, 10000)

            cut_out = 20  # turbine cut-out speed

            stress_factor = samples / cut_out
            failure_prob = np.clip(stress_factor, 0, 1)

            rel_mc = (1 - np.mean(failure_prob)) * 100

            # ---------------------------
            # DISPLAY
            # ---------------------------
            c1, c2, c3 = st.columns(3)
            c1.metric("FTA (Time-based)", f"{rel_fta:.2f}%")
            c2.metric("Markov", f"{rel_markov:.2f}%")
            c3.metric("Monte Carlo", f"{rel_mc:.2f}%")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Upload both datasets to start analysis")