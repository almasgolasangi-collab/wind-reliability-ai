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
st.sidebar.write("**Gearbox Type:** 3-Stage (Planetary/Helical)")
st.sidebar.write("**Lubrication:** ISO VG 320")
st.sidebar.divider()

st.sidebar.header("🛠️ Sensitivity Simulation")
st.sidebar.info("Adjust load parameters to see reliability impact.")
wind_stress_factor = st.sidebar.slider("Simulate Wind Load Increase (%)", 0, 50, 0)

# --- FILE UPLOAD ---
st.header("Step 1: Data Acquisition")

col1, col2 = st.columns(2)
with col1:
    weather_file = st.file_uploader("Upload weather.csv.zip or csv", type=["zip", "csv"])
with col2:
    failure_file = st.file_uploader("Upload component_failures.csv", type=["csv"])

if weather_file and failure_file:
    try:
        # --- LOAD WEATHER DATA ---
        if weather_file.name.endswith('.zip'):
            with zipfile.ZipFile(weather_file) as z:
                target = [f for f in z.namelist() if 'weather' in f.lower()][0]
                with z.open(target) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        # --- CLEAN DATA ---
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
        fail_df['Date'] = pd.to_datetime(fail_df['Date'])

        weather_2018 = weather_df[weather_df['Date'].dt.year == 2018].copy()

        weather_2018 = weather_2018.dropna()
        fail_df = fail_df.dropna()

        # Auto-detect wind column
        wind_col = [col for col in weather_2018.columns if 'wind' in col.lower()][0]

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧹 Data Cleansing",
            "📊 EDA",
            "🧬 Reliability Modeling",
            "🏁 Final Evaluation"
        ])

        # =========================
        # TAB 1: CLEANING
        # =========================
        with tab1:
            st.subheader("Cleaned Data")
            c1, c2 = st.columns(2)
            c1.dataframe(weather_2018.head())
            c2.dataframe(fail_df.head())

            csv = weather_2018.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Cleaned Data", csv, "cleaned_data.csv")

        # =========================
        # TAB 2: EDA
        # =========================
        with tab2:
            st.subheader("Exploratory Data Analysis")

            v_mean = weather_2018[wind_col].mean()
            v_std = weather_2018[wind_col].std()

            e1, e2 = st.columns(2)

            with e1:
                fig, ax = plt.subplots()
                sns.histplot(weather_2018[wind_col], bins=30, kde=True, ax=ax)
                ax.set_title("Wind Speed Distribution")
                st.pyplot(fig)

            with e2:
                fig2, ax2 = plt.subplots()
                ax2.plot(weather_2018['Date'], weather_2018[wind_col])

                ymin = weather_2018[wind_col].min()
                ymax = weather_2018[wind_col].max()

                ax2.vlines(fail_df['Date'], ymin=ymin, ymax=ymax, color='red', alpha=0.5)
                ax2.set_title("Failures vs Wind Timeline")

                st.pyplot(fig2)

        # =========================
        # TAB 3: RELIABILITY
        # =========================
        with tab3:
            st.subheader("Reliability Modeling (Corrected)")

            v_mean = weather_2018[wind_col].mean()
            v_std = weather_2018[wind_col].std()

            t = 1  # year
            lambda_rate = len(fail_df) / 365

            # FTA (Exponential)
            rel_fta = np.exp(-lambda_rate * t) * 100

            # Monte Carlo
            sim_mean = v_mean * (1 + wind_stress_factor / 100)
            samples = np.random.normal(sim_mean, v_std, 10000)

            threshold = 25
            rel_mc = np.mean(samples < threshold) * 100

            # Markov (Approx)
            rel_markov = np.exp(-lambda_rate * t) * 100

            # RUL
            RUL = 1 / lambda_rate if lambda_rate > 0 else np.inf

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("FTA", f"{rel_fta:.2f}%")
            m2.metric("Monte Carlo", f"{rel_mc:.2f}%")
            m3.metric("Markov", f"{rel_markov:.2f}%")
            m4.metric("RUL (years)", f"{RUL:.2f}")

            if rel_mc < 90:
                st.warning("⚠️ High failure risk detected!")

            # Graph
            fig_bar, ax_bar = plt.subplots()
            sns.barplot(x=['FTA', 'Monte Carlo', 'Markov'],
                        y=[rel_fta, rel_mc, rel_markov],
                        ax=ax_bar)
            st.pyplot(fig_bar)

        # =========================
        # TAB 4: FINAL
        # =========================
        with tab4:
            st.subheader("Final Evaluation")

            gb = fail_df[fail_df['Component'] == 'Gearbox']

            if not gb.empty:
                bearing_ratio = (
                    len(gb[gb['Failure_Mode'].str.contains('Bearing', case=False)])
                    / len(gb)
                ) * 100

                st.write(f"Gearbox Bearing Failure Ratio: {bearing_ratio:.2f}%")
                st.progress(bearing_ratio / 100)

            # Status
            if rel_fta > 97:
                status = "EXCELLENT"
            elif rel_fta > 90:
                status = "STABLE"
            else:
                status = "CRITICAL"

            st.header(f"Final Status: {status}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload datasets to start analysis.")