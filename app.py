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

# Failure rates (per year)
st.sidebar.subheader("Failure Rates (per year)")
lambda_g = st.sidebar.number_input("Gearbox Failure Rate", value=1.0)
lambda_gen = st.sidebar.number_input("Generator Failure Rate", value=0.5)
lambda_blade = st.sidebar.number_input("Blade Failure Rate", value=0.7)

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

        # CLEAN
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        # DATE COLUMN
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

        # WIND COLUMN
        def get_wind(df):
            for col in df.columns:
                if col.upper() in ["WS10M", "WS50M", "WINDSPEED_80M"]:
                    return col
                if "wind" in col.lower():
                    return col
            st.error("❌ No wind column found")
            st.stop()

        wind_col = get_wind(weather_df)
        weather_df.rename(columns={wind_col: "Wind"}, inplace=True)

        # YEAR FILTER
        year = st.selectbox("Select Year", sorted(weather_df[weather_date].dt.year.unique()))

        weather_year = weather_df[weather_df[weather_date].dt.year == year]
        fail_year = fail_df[fail_df[fail_date].dt.year == year]

        weather_year = weather_year.sort_values(by=weather_date)
        weather_year = weather_year.set_index(weather_date)

        daily_wind = weather_year["Wind"].resample('D').mean()

        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧹 Cleaning",
            "⚙️ Processing",
            "📊 EDA",
            "📈 Visualization",
            "🧬 Modeling"
        ])

        # EDA
        with tab3:
            v_mean = daily_wind.mean()
            v_std = daily_wind.std()

            st.write(f"Mean Wind Speed: {v_mean:.2f}")
            st.write(f"Std Dev: {v_std:.2f}")

            fig1, ax1 = plt.subplots()
            sns.histplot(daily_wind, bins=30, kde=True, ax=ax1)
            st.pyplot(fig1)

        # ===========================
        # MODELING (FINAL CORRECTED)
        # ===========================
        with tab5:
            st.subheader("Reliability Modeling (Final)")

            # Convert time
            t_years = mission_time / 365
            stress = 1 + wind_stress_factor / 100

            # Component failure probabilities
            gearbox_fail = 1 - np.exp(-lambda_g * stress * t_years)
            generator_fail = 1 - np.exp(-lambda_gen * stress * t_years)
            blade_fail = 1 - np.exp(-lambda_blade * stress * t_years)

            # AND + OR (FTA)
            Q_and = generator_fail * blade_fail
            Q_system = 1 - ((1 - gearbox_fail) * (1 - Q_and))
            rel_fta = (1 - Q_system) * 100

            # MCS (corrected)
            Q_mcs = gearbox_fail + Q_and - (gearbox_fail * Q_and)
            rel_mcs = (1 - Q_mcs) * 100

            # MARKOV (fixed)
            lambda_total = lambda_g + lambda_gen + lambda_blade
            mu = (1 / 7) * 365

            rel_markov = (
                (mu / (lambda_total + mu)) +
                (lambda_total / (lambda_total + mu)) * np.exp(-(lambda_total + mu) * t_years)
            ) * 100

            # ===========================
            # MONTE CARLO (FINAL FIXED)
            # ===========================
            N = 10000

            sim_mean = v_mean * stress
            samples = np.random.normal(sim_mean, v_std, N)
            samples = np.clip(samples, 0, None)

            # Wind failure
            wind_fail = samples > 20

            # Component failures
            gearbox_mc = np.random.rand(N) < gearbox_fail
            generator_mc = np.random.rand(N) < generator_fail
            blade_mc = np.random.rand(N) < blade_fail

            # AND + OR logic
            and_mc = generator_mc & blade_mc
            system_fail_mc = wind_fail | gearbox_mc | and_mc

            rel_mc = (1 - np.mean(system_fail_mc)) * 100

            # DISPLAY
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FTA", f"{rel_fta:.2f}%")
            c2.metric("MCS", f"{rel_mcs:.2f}%")
            c3.metric("Markov", f"{rel_markov:.2f}%")
            c4.metric("Monte Carlo", f"{rel_mc:.2f}%")

            # INTERPRETATION
            st.write("### 📊 Interpretation")
            st.write(f"FTA: {rel_fta:.2f}% (structural)")
            st.write(f"Markov: {rel_markov:.2f}% (dynamic)")
            st.write(f"Monte Carlo: {rel_mc:.2f}% (stochastic realistic)")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Upload both datasets to start analysis")