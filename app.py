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

# --- FILE UPLOAD ---
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

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 EDA",
            "🧩 Modeling",
            "🧠 Reliability",
            "📈 Results"
        ])

        # =========================
        # 📊 EDA
        # =========================
        with tab1:
            st.subheader("Wind Data Analysis")

            fig, ax = plt.subplots()
            sns.histplot(weather_2018[wind_col], kde=True, ax=ax)
            ax.set_title("Wind Speed Distribution")
            st.pyplot(fig)

        # =========================
        # 🧩 MODELING
        # =========================
        with tab2:
            st.subheader("System Modeling")

            # --- Wind Turbine Power Curve ---
            st.markdown("### 🌬️ Wind Turbine Power Model")

            v = np.linspace(0, 25, 100)
            power = np.piecewise(
                v,
                [v < 3, (v >= 3) & (v < 12), (v >= 12) & (v < 25), v >= 25],
                [0,
                 lambda v: (v-3)/(12-3),
                 1,
                 0]
            )

            fig, ax = plt.subplots()
            ax.plot(v, power)
            ax.set_title("Power Curve")
            st.pyplot(fig)

            # --- Component Availability ---
            st.markdown("### ⚙️ Component Model")

            components = {
                "Electrical": (3, 1.5),
                "Sensor": (2, 1.4),
                "Blade": (1, 2.5)
            }

            avail = []
            for name, (lam, mu) in components.items():
                A = mu / (lam + mu)
                avail.append(A)
                st.write(f"{name} Availability = {A:.3f}")

            eta = np.prod(avail)
            st.success(f"System Availability (η): {eta:.3f}")

            # --- Markov States ---
            st.markdown("### 🔄 Wind States")

            wind = weather_2018[wind_col].values
            states = np.digitize(wind, bins=[5, 12])
            st.bar_chart(pd.Series(states).value_counts().sort_index())

        # =========================
        # 🧠 RELIABILITY
        # =========================
        with tab3:
            st.subheader("Reliability Models")

            v_mean = weather_2018[wind_col].mean()
            v_std = weather_2018[wind_col].std()

            # --- FTA ---
            total_days = 365

            def failure_rate(keyword):
                if 'Failure_Mode' in fail_df.columns:
                    return len(fail_df[
                        fail_df['Failure_Mode'].str.contains(keyword, case=False, na=False)
                    ]) / total_days
                return 0.02

            λ1 = failure_rate("Bearing")
            λ2 = failure_rate("Electrical")
            λ3 = failure_rate("Mechanical")

            P1 = 1 - np.exp(-λ1)
            P2 = 1 - np.exp(-λ2)
            P3 = 1 - np.exp(-λ3)

            P_or = 1 - ((1-P1)*(1-P2)*(1-P3))
            P_and = P1*P2*P3
            P_fail = P_or + P_and - (P_or*P_and)

            rel_fta = (1 - P_fail) * 100

            # --- Weibull ---
            shape, loc, scale = weibull_min.fit(weather_2018[wind_col])
            rel_weibull = weibull_min.cdf(25, shape, loc, scale) * 100

            # --- Monte Carlo ---
            sim = 1000
            mc = []
            for _ in range(sim):
                sample = np.random.normal(v_mean*(1+wind_stress_factor/100), v_std)
                mc.append(sample < 25)

            rel_mc = np.mean(mc) * 100

            # --- Markov ---
            states = np.digitize(weather_2018[wind_col], bins=[5,12])
            M = np.zeros((3,3))

            for i in range(len(states)-1):
                M[states[i], states[i+1]] += 1

            M = M / M.sum(axis=1, keepdims=True)

            P = np.array([1/3,1/3,1/3])
            for _ in range(50):
                P = P @ M

            markov_rel = (1 - P[0]) * 100

            # --- Display ---
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("FTA", f"{rel_fta:.2f}%")
            c2.metric("Monte Carlo", f"{rel_mc:.2f}%")
            c3.metric("Markov", f"{markov_rel:.2f}%")
            c4.metric("Weibull", f"{rel_weibull:.2f}%")

        # =========================
        # 📈 RESULTS
        # =========================
        with tab4:
            st.subheader("Final Results")

            # Availability
            A_sys = eta
            FOR = (1 - A_sys) * 100

            # LOLP
            demand = np.percentile(weather_2018[wind_col], 60)
            LOLP = np.mean(weather_2018[wind_col] < demand) * 100

            # Final Reliability
            final_rel = (0.3*rel_fta + 0.3*rel_mc + 0.2*markov_rel + 0.2*rel_weibull)

            st.metric("Final Reliability", f"{final_rel:.2f}%")
            st.metric("Availability", f"{A_sys*100:.2f}%")
            st.metric("FOR", f"{FOR:.2f}%")
            st.metric("LOLP", f"{LOLP:.2f}%")

            if final_rel > 95:
                st.success("System Stable")
            elif final_rel > 85:
                st.warning("System Moderate Risk")
            else:
                st.error("System Critical")

    except Exception as e:
        st.error(e)

else:
    st.info("Upload datasets to start analysis")