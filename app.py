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
st.title("🛡️ Wind Turbine Reliability System")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙️ Machine Info")
st.sidebar.write("Gearbox: 3-Stage")
st.sidebar.write("Lubrication: ISO VG 320")

st.sidebar.header("🛠️ Wind Stress")
wind_stress_factor = st.sidebar.slider("Wind Load Increase (%)", 0, 50, 0)

# ---------------------------
# FILE UPLOAD
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    weather_file = st.file_uploader("Upload Weather CSV/ZIP", type=["csv", "zip"])

with col2:
    failure_file = st.file_uploader("Upload Failure CSV", type=["csv"])

# ---------------------------
# MAIN LOGIC
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

        # CLEAN
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        # DATE DETECTION
        def get_date(df):
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    return col
            st.error("No date column found")
            st.stop()

        weather_date = get_date(weather_df)
        fail_date = get_date(fail_df)

        weather_df[weather_date] = pd.to_datetime(weather_df[weather_date], dayfirst=True, errors='coerce')
        fail_df[fail_date] = pd.to_datetime(fail_df[fail_date], dayfirst=True, errors='coerce')

        weather_df.dropna(subset=[weather_date], inplace=True)
        fail_df.dropna(subset=[fail_date], inplace=True)

        # WIND COLUMN
        def get_wind(df):
            for col in df.columns:
                if col.upper() in ["WS10M", "WS50M"]:
                    return col
                if "wind" in col.lower():
                    return col
            st.error("No wind column found")
            st.write(df.columns)
            st.stop()

        wind_col = get_wind(weather_df)

        weather_df.rename(columns={wind_col: "Wind"}, inplace=True)
        wind_col = "Wind"

        # YEAR FILTER
        year = st.selectbox("Select Year", sorted(weather_df[weather_date].dt.year.unique()))
        weather_year = weather_df[weather_df[weather_date].dt.year == year]

        # SORT + RESAMPLE
        weather_year = weather_year.sort_values(by=weather_date)
        weather_year = weather_year.set_index(weather_date)

        daily_wind = weather_year[wind_col].resample('D').mean()

        # =========================
        # TABS
        # =========================
        tab1, tab2, tab3 = st.tabs(["EDA", "Modeling", "Final"])

        # =========================
        # EDA
        # =========================
        with tab1:
            st.subheader("Wind Data Analysis")

            v_mean = daily_wind.mean()
            v_std = daily_wind.std()

            # HISTOGRAM (CORRECT)
            fig1, ax1 = plt.subplots()
            sns.histplot(daily_wind, bins=30, kde=True, ax=ax1)
            ax1.set_title("Wind Speed Distribution")
            st.pyplot(fig1)

            # TIME SERIES (CLEAN)
            fig2, ax2 = plt.subplots(figsize=(12,5))
            ax2.plot(daily_wind.index, daily_wind.values)

            fail_days = pd.to_datetime(fail_df[fail_date]).dt.date
            for d in fail_days:
                ax2.axvline(pd.to_datetime(d), alpha=0.2)

            ax2.set_title("Wind Speed vs Failures")
            st.pyplot(fig2)

        # =========================
        # MODELING
        # =========================
        with tab2:
            st.subheader("Reliability Models")

            lambda_rate = len(fail_df) / 365
            mu = 1 / 5   # repair rate (assumption)

            # -------- FTA --------
            p = [0.02, 0.03, 0.01]  # component probs (can adjust)
            P_system = 1 - np.prod([1 - pi for pi in p])
            rel_fta = (1 - P_system) * 100

            # -------- MARKOV --------
            rel_markov = (mu / (lambda_rate + mu)) * 100

            # -------- MONTE CARLO --------
            sim_mean = v_mean * (1 + wind_stress_factor / 100)
            samples = np.random.normal(sim_mean, v_std, 10000)

            threshold = v_mean + 2*v_std
            rel_mc = np.mean(samples < threshold) * 100

            # DISPLAY
            c1, c2, c3 = st.columns(3)
            c1.metric("FTA", f"{rel_fta:.2f}%")
            c2.metric("Markov", f"{rel_markov:.2f}%")
            c3.metric("Monte Carlo", f"{rel_mc:.2f}%")

        # =========================
        # FINAL
        # =========================
        with tab3:
            st.subheader("Final Decision")

            if rel_mc > 95:
                status = "EXCELLENT"
            elif rel_mc > 85:
                status = "STABLE"
            else:
                status = "CRITICAL"

            st.header(status)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload both datasets")