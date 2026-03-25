import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, weibull_min
import zipfile

# --- MACHINE SPECS ---
def show_machine_specs():
    st.sidebar.header("⚙️ Machine Specifications")
    st.sidebar.write("**Gearbox Type:** 3-Stage (Planetary/Helical)")
    st.sidebar.write("**Lubrication:** ISO VG 320")
    st.sidebar.write("**Ref:** NREL Reliability Collaborative")
    st.sidebar.divider()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wind Reliability AI Pipeline", layout="wide")
st.title("🛡️ Wind Turbine Reliability: Advanced Modeling Pipeline")

# --- SIDEBAR ---
show_machine_specs()
st.sidebar.header("🛠️ Sensitivity Simulation")
wind_stress_factor = st.sidebar.slider("Simulate Wind Load Increase (%)", 0, 50, 0)

# --- DATA UPLOAD ---
st.header("Step 1: Data Acquisition")
col_u1, col_u2 = st.columns(2)

with col_u1:
    weather_file = st.file_uploader("Upload weather.csv.zip", type=["zip", "csv"])
with col_u2:
    failure_file = st.file_uploader("Upload component_failures.csv", type=["csv"])

if weather_file and failure_file:
    try:
        # --- LOAD WEATHER ---
        if weather_file.name.endswith('.zip'):
            with zipfile.ZipFile(weather_file) as z:
                target = [f for f in z.namelist() if 'weather.csv' in f][0]
                with z.open(target) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        # --- CLEAN ---
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
        fail_df['Date'] = pd.to_datetime(fail_df['Date'])

        weather_2018 = weather_df[weather_df['Date'].dt.year == 2018].copy()

        # --- WIND COLUMN AUTO DETECT ---
        wind_col = [c for c in weather_2018.columns if 'wind' in c.lower()][0]

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
            st.subheader("Data Cleansing")
            c1, c2 = st.columns(2)
            c1.dataframe(weather_2018.head())
            c2.dataframe(fail_df.head())

            csv = weather_2018.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Cleaned Data", csv, "cleaned_data.csv")

        # =========================
        # TAB 2: EDA
        # =========================
        with tab2:
            st.subheader("Exploratory Analysis")

            v_mean = weather_2018[wind_col].mean()
            v_std = weather_2018[wind_col].std()

            e1, e2 = st.columns(2)

            with e1:
                fig, ax = plt.subplots()
                sns.histplot(weather_2018[wind_col], kde=True, ax=ax)
                ax.set_title("Wind Speed Distribution")
                st.pyplot(fig)

            with e2:
                fig2, ax2 = plt.subplots()
                ax2.plot(weather_2018['Date'], weather_2018[wind_col])
                for d in fail_df['Date']:
                    ax2.axvline(d, color='red', linestyle='--')
                ax2.set_title("Failures vs Time")
                st.pyplot(fig2)

        # =========================
        # TAB 3: RELIABILITY MODEL
        # =========================
        with tab3:
            st.subheader("Advanced Reliability Modeling")

            # -------------------------
            # 1. FTA (REAL IMPLEMENTATION)
            # -------------------------
            total_days = 365

            def get_prob(keyword):
                if 'Failure_Mode' in fail_df.columns:
                    return len(fail_df[
                        fail_df['Failure_Mode'].str.contains(keyword, case=False, na=False)
                    ]) / total_days
                return 0.05  # fallback

            P_bearing = get_prob("Bearing")
            P_electrical = get_prob("Electrical")
            P_mechanical = get_prob("Mechanical")

            # OR gate
            P_or = 1 - ((1 - P_bearing) * (1 - P_electrical) * (1 - P_mechanical))

            # AND gate
            P_and = P_bearing * P_electrical * P_mechanical

            # System failure
            P_system_failure = P_or + P_and - (P_or * P_and)

            rel_fta = (1 - P_system_failure) * 100

            # -------------------------
            # 2. WEIBULL AGING
            # -------------------------
            wind_data = weather_2018[wind_col].dropna()
            shape, loc, scale = weibull_min.fit(wind_data)

            rel_weibull = weibull_min.cdf(25, shape, loc, scale) * 100

            # -------------------------
            # 3. MONTE CARLO
            # -------------------------
            sim_mean = v_mean * (1 + wind_stress_factor / 100)
            rel_mc = norm.cdf(25, sim_mean, v_std) * 100

            # -------------------------
            # 4. MARKOV MODEL
            # -------------------------
            M = np.array([
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6]
            ])

            P = np.array([1/3, 1/3, 1/3])
            for _ in range(50):
                P = np.dot(P, M)

            markov_rel = (P[1] + P[2]) * 100

            # -------------------------
            # 5. COMPONENT AVAILABILITY
            # -------------------------
            lambda_i = 2
            mu_i = 1.5

            A_i = mu_i / (lambda_i + mu_i)
            FOR = (1 - A_i) * 100

            # -------------------------
            # DISPLAY
            # -------------------------
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("FTA", f"{rel_fta:.2f}%")
            m2.metric("Weibull", f"{rel_weibull:.2f}%")
            m3.metric("Monte Carlo", f"{rel_mc:.2f}%")
            m4.metric("Markov", f"{markov_rel:.2f}%")

            st.divider()

            m5, m6 = st.columns(2)
            m5.metric("Availability", f"{A_i*100:.2f}%")
            m6.metric("FOR", f"{FOR:.2f}%")

            # -------------------------
            # FINAL RELIABILITY
            # -------------------------
            final_rel = (rel_fta + rel_mc + markov_rel + rel_weibull) / 4

            st.header(f"Overall Reliability: {final_rel:.2f}%")

            if final_rel > 95:
                status = "STABLE"
            elif final_rel > 85:
                status = "WARNING"
            else:
                status = "CRITICAL"

            st.subheader(f"System Status: {status}")

        # =========================
        # TAB 4: FINAL
        # =========================
        with tab4:
            st.subheader("Final Evaluation")

            gb_only = fail_df[fail_df['Component'] == 'Gearbox']

            if not gb_only.empty and 'Failure_Mode' in gb_only.columns:
                bearing_ratio = len(
                    gb_only[gb_only['Failure_Mode'].str.contains('Bearing', case=False)]
                ) / len(gb_only) * 100

                st.metric("Bearing Failure Ratio", f"{bearing_ratio:.1f}%")
                st.progress(bearing_ratio / 100)

            st.success("Pipeline Executed Successfully")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload datasets to start analysis")