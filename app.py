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
        # TAB 2: EDA
        # =========================
        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                sns.histplot(weather_2018[wind_col], bins=30, kde=True, ax=ax)
                ax.set_title("Wind Distribution")
                st.pyplot(fig)

            with col2:
                fig2, ax2 = plt.subplots()
                ax2.plot(weather_2018['Date'], weather_2018[wind_col])

                for d in fail_df['Date']:
                    ax2.axvline(d, color='red', linestyle='--')

                ax2.set_title("Failures vs Time")
                st.pyplot(fig2)

        # =========================
        # TAB 3: MODELING
        # =========================
        with tab3:

            st.subheader("Advanced Reliability Models")

            # 🔴 FAULT TREE ANALYSIS (FIXED)
            p_high = np.mean(weather_2018[wind_col] > cut_out)
            p_low = np.mean(weather_2018[wind_col] < cut_in)

            p_wind = p_high + p_low

            p_gearbox = max(0.01, len(fail_df[fail_df['Component'] == 'Gearbox']) / 365)
            p_electrical = max(0.01, len(fail_df[fail_df['Component'] == 'Electrical']) / 365)

            p_and = p_wind * p_gearbox
            p_total = p_and + p_electrical - (p_and * p_electrical)

            rel_fta = (1 - p_total) * 100

            # 🟢 TRUE MONTE CARLO (FIXED)
            n_sim = 10000

            # Weibull distribution (realistic wind)
            k = 2
            c = sim_mean
            samples = np.random.weibull(k, n_sim) * c

            # Add turbulence
            samples += np.random.normal(0, v_std * 0.5, n_sim)

            # Reliability condition
            safe_runs = np.sum((samples >= cut_in) & (samples <= cut_out))
            rel_mc = (safe_runs / n_sim) * 100

            # 🔵 MARKOV CHAIN
            P = np.array([
                [0.85, 0.10, 0.05],
                [0.10, 0.75, 0.15],
                [0.00, 0.00, 1.00]
            ])

            state = np.array([1, 0, 0])

            for _ in range(10):
                state = np.dot(state, P)

            rel_markov = (state[0] + state[1]) * 100

            # --- DISPLAY ---
            c1, c2, c3 = st.columns(3)
            c1.metric("FTA Reliability", f"{rel_fta:.2f}%")
            c2.metric("Monte Carlo", f"{rel_mc:.2f}%")
            c3.metric("Markov Chain", f"{rel_markov:.2f}%")

            # --- GRAPH ---
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(['FTA', 'Monte Carlo', 'Markov'],
                       [rel_fta, rel_mc, rel_markov])
            ax_bar.set_ylabel("Reliability (%)")
            ax_bar.set_title("Reliability Comparison")
            st.pyplot(fig_bar)

        # =========================
        # TAB 4: FINAL
        # =========================
        with tab4:

            st.subheader("Final Evaluation")

            # 🟣 LOLP
            lolp = np.mean((samples < cut_in) | (samples > cut_out)) * 100

            # 🟢 WPG
            power = np.where(
                samples < cut_in, 0,
                np.where(samples < cut_out, samples**3, 0)
            )

            wpg = np.mean(power)

            col1, col2 = st.columns(2)
            col1.metric("LOLP (%)", f"{lolp:.2f}")
            col2.metric("Wind Power Index", f"{wpg:.2f}")

            # FINAL STATUS
            status = "STABLE" if rel_fta > 90 else "CRITICAL"

            st.header(f"Final Decision: {'🟢 STABLE' if status=='STABLE' else '🔴 CRITICAL'}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload both datasets to begin.")