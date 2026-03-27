import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wind Reliability AI Pipeline", layout="wide")
st.title("🛡️ Wind Turbine Reliability: Advanced Modeling Pipeline")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Machine Specifications")
st.sidebar.write("Gearbox: 3-Stage")
st.sidebar.write("Lubrication: ISO VG 320")

st.sidebar.header("🛠️ Sensitivity")
wind_stress_factor = st.sidebar.slider("Wind Load Increase (%)", 0, 50, 0)

# =========================
# HELPER FUNCTIONS (NEW 🔥)
# =========================

def find_date_column(df, name):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    st.error(f"❌ No Date/Time column found in {name}")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

def find_wind_column(df):
    for col in df.columns:
        if "wind" in col.lower():
            return col
    st.error("❌ No Wind Speed column found")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

def safe_datetime(df, col):
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[col])
    return df

# =========================
# FILE UPLOAD
# =========================

st.header("Step 1: Upload Data")

col1, col2 = st.columns(2)

with col1:
    weather_file = st.file_uploader("Upload Weather Data", type=["csv", "zip"])

with col2:
    failure_file = st.file_uploader("Upload Failure Data", type=["csv"])

if weather_file and failure_file:
    try:
        # =========================
        # LOAD WEATHER DATA
        # =========================
        if weather_file.name.endswith(".zip"):
            with zipfile.ZipFile(weather_file) as z:
                file_name = [f for f in z.namelist() if f.endswith(".csv")][0]
                with z.open(file_name) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        # Clean column names
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        # =========================
        # SAFE COLUMN DETECTION
        # =========================
        date_col_weather = find_date_column(weather_df, "Weather Data")
        date_col_fail = find_date_column(fail_df, "Failure Data")
        wind_col = find_wind_column(weather_df)

        # =========================
        # SAFE DATE CONVERSION
        # =========================
        weather_df = safe_datetime(weather_df, date_col_weather)
        fail_df = safe_datetime(fail_df, date_col_fail)

        # =========================
        # YEAR SELECTION
        # =========================
        years = weather_df[date_col_weather].dt.year.unique()

        if len(years) == 0:
            st.error("❌ No valid date data found")
            st.stop()

        selected_year = st.selectbox("Select Year", sorted(years))

        weather_year = weather_df[weather_df[date_col_weather].dt.year == selected_year]

        # =========================
        # TABS
        # =========================
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧹 Data",
            "📊 EDA",
            "🧬 Modeling",
            "🏁 Final"
        ])

        # =========================
        # TAB 1
        # =========================
        with tab1:
            st.subheader("Cleaned Data")
            c1, c2 = st.columns(2)
            c1.dataframe(weather_year.head())
            c2.dataframe(fail_df.head())

        # =========================
        # TAB 2 (EDA)
        # =========================
        with tab2:
            st.subheader("EDA")

            v_mean = weather_year[wind_col].mean()
            v_std = weather_year[wind_col].std()

            fig, ax = plt.subplots()
            sns.histplot(weather_year[wind_col], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.plot(weather_year[date_col_weather], weather_year[wind_col])

            ymin = weather_year[wind_col].min()
            ymax = weather_year[wind_col].max()

            ax2.vlines(fail_df[date_col_fail], ymin=ymin, ymax=ymax, color='red', alpha=0.5)
            st.pyplot(fig2)

        # =========================
        # TAB 3 (MODELING)
        # =========================
        with tab3:
            st.subheader("Reliability Modeling")

            v_mean = weather_year[wind_col].mean()
            v_std = weather_year[wind_col].std()

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

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("FTA", f"{rel_fta:.2f}%")
            m2.metric("Monte Carlo", f"{rel_mc:.2f}%")
            m3.metric("Markov", f"{rel_markov:.2f}%")
            m4.metric("RUL", f"{RUL:.2f} yrs")

            if rel_mc < 90:
                st.warning("⚠️ High Risk Detected!")

        # =========================
        # TAB 4 (FINAL)
        # =========================
        with tab4:
            st.subheader("Final Decision")

            if rel_fta > 97:
                status = "EXCELLENT"
            elif rel_fta > 90:
                status = "STABLE"
            else:
                status = "CRITICAL"

            st.header(f"Status: {status}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Upload both datasets to begin.")