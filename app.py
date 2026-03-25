import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, weibull_min
import zipfile

# --- CONFIG ---
st.set_page_config(page_title="Wind Reliability AI Pipeline", layout="wide")
st.title("🛡️ Wind Turbine Reliability: Research-Level Modeling")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Simulation Controls")
wind_stress_factor = st.sidebar.slider("Wind Load Increase (%)", 0, 50, 0)

# --- UPLOAD ---
col1, col2 = st.columns(2)
weather_file = col1.file_uploader("Upload weather.csv.zip", type=["zip", "csv"])
failure_file = col2.file_uploader("Upload failures.csv", type=["csv"])

if weather_file and failure_file:
    try:
        # --- LOAD DATA ---
        if weather_file.name.endswith('.zip'):
            with zipfile.ZipFile(weather_file) as z:
                target = [f for f in z.namelist() if 'weather.csv' in f][0]
                with z.open(target) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        weather_df['Date'] = pd.to_datetime(weather_df['Date'], dayfirst=True)
        fail_df['Date'] = pd.to_datetime(fail_df['Date'])

        weather_2018 = weather_df[weather_df['Date'].dt.year == 2018]

        wind_col = [c for c in weather_2018.columns if 'wind' in c.lower()][0]

        tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧠 Modeling", "📈 Results"])

        # =========================
        # 📊 EDA
        # =========================
        with tab1:
            st.subheader("Wind Data Analysis")

            fig, ax = plt.subplots()
            sns.histplot(weather_2018[wind_col], kde=True, ax=ax)
            st.pyplot(fig)

        # =========================
        # 🧠 MODELING
        # =========================
        with tab3:

            st.subheader("Advanced Reliability Modeling")

            v_mean = weather_2018[wind_col].mean()
            v_std = weather_2018[wind_col].std()

            # ======================
            # 1. FTA (Improved)
            # ======================
            total_days = 365

            def failure_rate(keyword):
                if 'Failure_Mode' in fail_df.columns:
                    return len(fail_df[
                        fail_df['Failure_Mode'].str.contains(keyword, case=False, na=False)
                    ]) / total_days
                return 0.02

            λ_bearing = failure_rate("Bearing")
            λ_elec = failure_rate("Electrical")
            λ_mech = failure_rate("Mechanical")

            # Convert λ → probability
            P_bearing = 1 - np.exp(-λ_bearing)
            P_elec = 1 - np.exp(-λ_elec)
            P_mech = 1 - np.exp(-λ_mech)

            # OR gate
            P_or = 1 - ((1 - P_bearing)*(1 - P_elec)*(1 - P_mech))

            # AND gate
            P_and = P_bearing * P_elec * P_mech

            P_failure = P_or + P_and - (P_or * P_and)
            rel_fta = (1 - P_failure) * 100

            # ======================
            # 2. WEIBULL AGING
            # ======================
            wind_data = weather_2018[wind_col].dropna()
            shape, loc, scale = weibull_min.fit(wind_data)

            rel_weibull = weibull_min.cdf(25, shape, loc, scale) * 100

            # Aging curve
            t = np.linspace(0, 10, 100)
            aging_curve = np.exp(-(t/scale)**shape)

            # ======================
            # 3. MONTE CARLO (REAL)
            # ======================
            simulations = 1000
            results = []

            for _ in range(simulations):
                sample = np.random.normal(v_mean*(1+wind_stress_factor/100), v_std)
                results.append(sample < 25)

            rel_mc = np.mean(results) * 100

            # ======================
            # 4. MARKOV (DATA-DRIVEN)
            # ======================
            wind = weather_2018[wind_col].values

            states = np.digitize(wind, bins=[5, 12])  # 0,1,2

            M = np.zeros((3,3))

            for i in range(len(states)-1):
                M[states[i], states[i+1]] += 1

            M = M / M.sum(axis=1, keepdims=True)

            P = np.array([1/3,1/3,1/3])
            for _ in range(50):
                P = P @ M

            markov_rel = (1 - P[0]) * 100

            # ======================
            # 5. COMPONENT AVAILABILITY
            # ======================
            components = [
                (3,1.5),
                (2,1.4),
                (1,2.5)
            ]

            A_system = np.prod([mu/(lam+mu) for lam,mu in components])
            FOR = (1 - A_system) * 100

            # ======================
            # 6. LOLP (IMPORTANT)
            # ======================
            demand = np.percentile(wind, 60)
            LOLP = np.mean(wind < demand) * 100

            # ======================
            # FINAL RELIABILITY
            # ======================
            final_rel = (
                0.3*rel_fta +
                0.3*rel_mc +
                0.2*markov_rel +
                0.2*rel_weibull
            )

            # ======================
            # DISPLAY
            # ======================
            m1,m2,m3,m4 = st.columns(4)

            m1.metric("FTA", f"{rel_fta:.2f}%")
            m2.metric("Monte Carlo", f"{rel_mc:.2f}%")
            m3.metric("Markov", f"{markov_rel:.2f}%")
            m4.metric("Weibull", f"{rel_weibull:.2f}%")

            st.divider()

            m5,m6 = st.columns(2)
            m5.metric("Availability", f"{A_system*100:.2f}%")
            m6.metric("FOR", f"{FOR:.2f}%")

            st.metric("LOLP", f"{LOLP:.2f}%")

            st.header(f"Final Reliability: {final_rel:.2f}%")

            # ======================
            # PLOTS
            # ======================
            st.subheader("Weibull Aging Curve")
            fig2, ax2 = plt.subplots()
            ax2.plot(t, aging_curve)
            ax2.set_title("Reliability vs Time")
            st.pyplot(fig2)

    except Exception as e:
        st.error(e)

else:
    st.info("Upload datasets to start")