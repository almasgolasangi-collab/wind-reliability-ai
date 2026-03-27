import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Wind Reliability System", layout="wide")
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
        # LOAD WEATHER DATA
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
            st.write(df.columns)
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
            st.write(df.columns)
            st.stop()

        wind_col = get_wind(weather_df)

        # Rename for consistency
        weather_df.rename(columns={wind_col: "Wind"}, inplace=True)
        wind_col = "Wind"

        # ---------------------------
        # YEAR SELECTION
        # ---------------------------
        years = weather_df[weather_date].dt.year.unique()
        year = st.selectbox("Select Year", sorted(years))

        weather_year = weather_df[weather_df[weather_date].dt.year == year]

        # SORT + RESAMPLE
        weather_year = weather_year.sort_values(by=weather_date)
        weather_year = weather_year.set_index(weather_date)

        daily_wind = weather_year[wind_col].resample('D').mean()

        # FILTER FAILURE SAME YEAR (IMPORTANT 🔥)
        fail_year = fail_df[fail_df[fail_date].dt.year == year]

        # ---------------------------
        # TABS
        # ---------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧹 Data Cleaning",
            "⚙️ Data Processing",
            "📊 EDA",
            "📈 Visualization",
            "🧬 Modeling"
        ])

        # ---------------------------
        # TAB 1: CLEANING
        # ---------------------------
        with tab1:
            st.subheader("Data Cleaning")
            st.write("Missing values removed and date formats standardized.")
            st.dataframe(weather_df.head())
            st.dataframe(fail_df.head())

        # ---------------------------
        # TAB 2: PROCESSING
        # ---------------------------
        with tab2:
            st.subheader("Data Processing")
            st.write(f"Selected Year: {year}")
            st.write(f"Failures in this year: {len(fail_year)}")
            st.dataframe(daily_wind.head())

        # ---------------------------
        # TAB 3: EDA
        # ---------------------------
        with tab3:
            st.subheader("Exploratory Data Analysis")

            v_mean = daily_wind.mean()
            v_std = daily_wind.std()

            st.write(f"Mean Wind Speed: {v_mean:.2f}")
            st.write(f"Standard Deviation: {v_std:.2f}")

            fig1, ax1 = plt.subplots()
            sns.histplot(daily_wind, bins=30, kde=True, ax=ax1)
            ax1.set_title("Wind Speed Distribution")
            st.pyplot(fig1)

        # ---------------------------
        # TAB 4: VISUALIZATION
        # ---------------------------
        with tab4:
            st.subheader("Wind vs Failures")

            fig2, ax2 = plt.subplots(figsize=(12, 5))

            # Plot wind
            ax2.plot(daily_wind.index, daily_wind.values)

            # Plot failures (aligned correctly)
            fail_days = pd.to_datetime(fail_year[fail_date]).dt.date
            for d in fail_days:
                ax2.axvline(pd.to_datetime(d), alpha=0.3)

            ax2.set_title("Wind Speed vs Failures")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Wind Speed")

            st.pyplot(fig2)

        # ---------------------------
        # TAB 5: MODELING
        # ---------------------------
        with tab5:
            st.subheader("Reliability Modeling")

            lambda_rate = len(fail_year) / 365
            mu = 1 / 5  # repair rate

            # FTA
            p = [0.02, 0.03, 0.01]
            P_system = 1 - np.prod([1 - pi for pi in p])
            rel_fta = (1 - P_system) * 100

            # Markov
            rel_markov = (mu / (lambda_rate + mu)) * 100

            # Monte Carlo
            sim_mean = v_mean * (1 + wind_stress_factor / 100)
            samples = np.random.normal(sim_mean, v_std, 10000)

            threshold = v_mean + 2 * v_std
            rel_mc = np.mean(samples < threshold) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("FTA", f"{rel_fta:.2f}%")
            c2.metric("Markov", f"{rel_markov:.2f}%")
            c3.metric("Monte Carlo", f"{rel_mc:.2f}%")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Upload both datasets to begin")