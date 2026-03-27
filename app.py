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
st.header("Upload Data")

col1, col2 = st.columns(2)

with col1:
    weather_file = st.file_uploader("Upload Weather CSV or ZIP", type=["csv", "zip"])

with col2:
    failure_file = st.file_uploader("Upload Failure CSV", type=["csv"])

# ---------------------------
# MAIN LOGIC
# ---------------------------
if weather_file and failure_file:
    try:
        # ---------------------------
        # LOAD WEATHER
        # ---------------------------
        if weather_file.name.endswith(".zip"):
            with zipfile.ZipFile(weather_file) as z:
                file_name = [f for f in z.namelist() if f.endswith(".csv")][0]
                with z.open(file_name) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        # ---------------------------
        # CLEAN COLUMN NAMES
        # ---------------------------
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        # ---------------------------
        # FIX DATE COLUMN (AUTO)
        # ---------------------------
        def get_date_col(df):
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    return col
            st.error("❌ No Date column found")
            st.write(df.columns)
            st.stop()

        weather_date = get_date_col(weather_df)
        fail_date = get_date_col(fail_df)

        weather_df[weather_date] = pd.to_datetime(weather_df[weather_date], dayfirst=True, errors='coerce')
        fail_df[fail_date] = pd.to_datetime(fail_df[fail_date], dayfirst=True, errors='coerce')

        weather_df = weather_df.dropna(subset=[weather_date])
        fail_df = fail_df.dropna(subset=[fail_date])

        # ---------------------------
        # FIX WIND COLUMN (AUTO)
        # ---------------------------
        def get_wind_col(df):
            for col in df.columns:
                if col.upper() in ["WS10M", "WS50M"]:
                    return col
                if "wind" in col.lower():
                    return col
            st.error("❌ No Wind Speed column found")
            st.write(df.columns)
            st.stop()

        wind_col = get_wind_col(weather_df)

        # Rename for consistency
        weather_df.rename(columns={wind_col: "Wind speed"}, inplace=True)
        wind_col = "Wind speed"

        # ---------------------------
        # SELECT YEAR
        # ---------------------------
        years = weather_df[weather_date].dt.year.unique()
        selected_year = st.selectbox("Select Year", sorted(years))

        weather_year = weather_df[weather_df[weather_date].dt.year == selected_year]

        # ---------------------------
        # TABS
        # ---------------------------
        tab1, tab2, tab3, tab4 = st.tabs(["Data", "EDA", "Model", "Final"])

        # ---------------------------
        # TAB 1
        # ---------------------------
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(weather_year.head())
            st.dataframe(fail_df.head())

        # ---------------------------
        # TAB 2 (EDA)
        # ---------------------------
        with tab2:
            st.subheader("Wind Analysis")

            v_mean = weather_year[wind_col].mean()
            v_std = weather_year[wind_col].std()

            fig, ax = plt.subplots()
            sns.histplot(weather_year[wind_col], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.plot(weather_year[weather_date], weather_year[wind_col])

            ymin = weather_year[wind_col].min()
            ymax = weather_year[wind_col].max()

            ax2.vlines(fail_df[fail_date], ymin=ymin, ymax=ymax, color='red', alpha=0.5)
            st.pyplot(fig2)

        # ---------------------------
        # TAB 3 (MODEL)
        # ---------------------------
        with tab3:
            st.subheader("Reliability Modeling")

            lambda_rate = len(fail_df) / 365
            t = 1

            # FTA
            rel_fta = np.exp(-lambda_rate * t) * 100

            # Monte Carlo
            sim_mean = v_mean * (1 + wind_stress_factor / 100)
            samples = np.random.normal(sim_mean, v_std, 10000)
            rel_mc = np.mean(samples < 25) * 100

            # Markov
            rel_markov = rel_fta

            # RUL
            RUL = 1 / lambda_rate if lambda_rate > 0 else np.inf

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FTA", f"{rel_fta:.2f}%")
            c2.metric("Monte Carlo", f"{rel_mc:.2f}%")
            c3.metric("Markov", f"{rel_markov:.2f}%")
            c4.metric("RUL (years)", f"{RUL:.2f}")

            if rel_mc < 90:
                st.warning("⚠️ High Failure Risk")

        # ---------------------------
        # TAB 4 (FINAL)
        # ---------------------------
        with tab4:
            st.subheader("Final Status")

            if rel_fta > 97:
                status = "EXCELLENT"
            elif rel_fta > 90:
                status = "STABLE"
            else:
                status = "CRITICAL"

            st.header(f"{status}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Upload both files to start")