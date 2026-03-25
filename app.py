import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Wind Reliability AI Pipeline",
    page_icon="wind_reliability_icon.ico",
    layout="wide"
)

st.title("🛡️ Wind Turbine Reliability: Advanced Modeling Pipeline")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Controls")

wind_stress_factor = st.sidebar.slider("Wind Load Increase (%)", 0, 50, 0)
cut_in = st.sidebar.slider("Cut-in Wind Speed (m/s)", 1, 5, 3)
cut_out = st.sidebar.slider("Cut-out Wind Speed (m/s)", 20, 30, 25)

# --- FILE UPLOAD ---
st.header("Step 1: Data Upload")

weather_file = st.file_uploader("Upload weather.csv or .zip", type=["csv", "zip"])
failure_file = st.file_uploader("Upload component_failures.csv", type=["csv"])

if weather_file and failure_file:

    try:
        # --- LOAD DATA ---
        if weather_file.name.endswith('.zip'):
            with zipfile.ZipFile(weather_file) as z:
                file = [f for f in z.namelist() if 'weather.csv' in f][0]
                with z.open(file) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        # --- CLEANING ---
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
        fail_df['Date'] = pd.to_datetime(fail_df['Date'])

        weather_2018 = weather_df[weather_df['Date'].dt.year == 2018].copy()

        wind_col = 'Wind speed (m/s)'

        v_mean = weather_2018[wind_col].mean()
        v_std = weather_2018[wind_col].std()

        sim_mean = v_mean * (1 + wind_stress_factor / 100)

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🧹 Data", "📊 EDA", "🧬 Modeling", "🏁 Final"]
        )

        # =========================
        # TAB 1: DATA
        # =========================
        with tab1:
            st.subheader("Cleaned Data")
            st.dataframe(weather_2018.head())

        # =========================
        # TAB 2: ADVANCED EDA
        # =========================
        with tab2:

            st.subheader("📊 Advanced Exploratory Data Analysis")

            # --- STATS ---
            st.markdown("### 🔢 Statistical Summary")
            st.write(weather_2018[wind_col].describe())

            # --- DISTRIBUTION + BOXPLOT ---
            st.markdown("### 📉 Distribution & Outliers")
            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                sns.histplot(weather_2018[wind_col], bins=30, kde=True, ax=ax1)
                ax1.set_title("Wind Distribution")
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots()
                sns.boxplot(x=weather_2018[wind_col], ax=ax2)
                ax2.set_title("Outlier Detection")
                st.pyplot(fig2)

            # --- TIME SERIES ---
            st.markdown("### 📈 Wind vs Failures")

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(weather_2018['Date'], weather_2018[wind_col])

            for d in fail_df['Date']:
                ax3.axvline(d, color='red', linestyle='--', alpha=0.5)

            st.pyplot(fig3)

            # --- MONTHLY TREND ---
            st.markdown("### 📅 Monthly Trend")

            monthly_avg = weather_2018.groupby(weather_2018['Date'].dt.month)[wind_col].mean()

            fig4, ax4 = plt.subplots()
            monthly_avg.plot(marker='o', ax=ax4)
            st.pyplot(fig4)

            # --- FAILURE INSIGHT ---
            st.markdown("### ⚠️ Failure Insight")

            failure_days = weather_2018[weather_2018['Date'].isin(fail_df['Date'])]

            if not failure_days.empty:
                avg_failure = failure_days[wind_col].mean()
                avg_normal = weather_2018[wind_col].mean()

                st.write(f"Failure Wind Avg: {avg_failure:.2f}")
                st.write(f"Normal Wind Avg: {avg_normal:.2f}")

                if avg_failure > avg_normal:
                    st.warning("Failures linked to high wind")
                else:
                    st.info("Failures likely due to other factors")

        # =========================
        # TAB 3: MODELING
        # =========================
        with tab3:

            st.subheader("🧬 Reliability Modeling")

            # 🔴 FTA (Binary Simulation)
            n_fta = 10000
            failures = 0

            p_high = np.mean(weather_2018[wind_col] > cut_out)
            p_low = np.mean(weather_2018[wind_col] < cut_in)
            p_wind = p_high + p_low

            p_gearbox = max(0.01, len(fail_df[fail_df['Component'] == 'Gearbox']) / 365)
            p_electrical = max(0.01, len(fail_df[fail_df['Component'] == 'Electrical']) / 365)

            for _ in range(n_fta):
                event_wind = np.random.rand() < p_wind
                event_gearbox = np.random.rand() < p_gearbox
                event_electrical = np.random.rand() < p_electrical

                failure = (event_wind and event_gearbox) or event_electrical

                if failure:
                    failures += 1

            rel_fta = (1 - failures / n_fta) * 100

            # 🟢 MONTE CARLO
            n_sim = 10000
            k = 2
            c = sim_mean

            samples = np.random.weibull(k, n_sim) * c
            samples += np.random.normal(0, v_std * 0.5, n_sim)

            safe_runs = np.sum((samples >= cut_in) & (samples <= cut_out))
            rel_mc = (safe_runs / n_sim) * 100

            # 🔵 MARKOV
            P = np.array([
                [0.85, 0.10, 0.05],
                [0.10, 0.75, 0.15],
                [0.00, 0.00, 1.00]
            ])

            state = np.array([1, 0, 0])

            for _ in range(10):
                state = np.dot(state, P)

            rel_markov = (state[0] + state[1]) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("FTA", f"{rel_fta:.2f}%")
            c2.metric("Monte Carlo", f"{rel_mc:.2f}%")
            c3.metric("Markov", f"{rel_markov:.2f}%")

        # =========================
        # TAB 4: FINAL
        # =========================
        with tab4:

            st.subheader("🏁 Final Evaluation")

            lolp = np.mean((samples < cut_in) | (samples > cut_out)) * 100

            power = np.where(
                samples < cut_in, 0,
                np.where(samples < cut_out, samples**3, 0)
            )

            wpg = np.mean(power)

            col1, col2 = st.columns(2)
            col1.metric("LOLP (%)", f"{lolp:.2f}")
            col2.metric("WPG", f"{wpg:.2f}")

            status = "STABLE" if rel_fta > 90 else "CRITICAL"

            st.header(f"Final Decision: {'🟢 STABLE' if status=='STABLE' else '🔴 CRITICAL'}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload datasets to begin.")