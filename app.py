import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Wind Turbine Reliability Model", layout="wide")
st.title("🛡️ Wind Turbine Reliability - System Modeling Approach")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Controls")
wind_stress = st.sidebar.slider("Wind Stress (%)", 0, 50, 0)

# -------------------------------
# FILE UPLOAD
# -------------------------------
col1, col2 = st.columns(2)
weather_file = col1.file_uploader("Upload weather.csv or .zip", type=["csv", "zip"])
failure_file = col2.file_uploader("Upload failures.csv", type=["csv"])

if weather_file and failure_file:
    try:
        # -------------------------------
        # LOAD DATA
        # -------------------------------
        if weather_file.name.endswith(".zip"):
            with zipfile.ZipFile(weather_file) as z:
                fname = [f for f in z.namelist() if "weather" in f][0]
                with z.open(fname) as f:
                    weather_df = pd.read_csv(f)
        else:
            weather_df = pd.read_csv(weather_file)

        fail_df = pd.read_csv(failure_file)

        weather_df["Date"] = pd.to_datetime(weather_df["Date"], dayfirst=True)
        fail_df["Date"] = pd.to_datetime(fail_df["Date"])

        weather_2018 = weather_df[weather_df["Date"].dt.year == 2018].copy()

        # detect wind column
        wind_col = [c for c in weather_2018.columns if "wind" in c.lower()][0]

        # apply stress
        weather_2018["Wind_Adjusted"] = weather_2018[wind_col] * (1 + wind_stress/100)

        # -------------------------------
        # TABS
        # -------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 EDA",
            "🧩 Modeling",
            "🧠 Reliability",
            "📈 Results"
        ])

        # ==============================
        # 📊 EDA
        # ==============================
        with tab1:
            st.subheader("Wind Data")

            fig, ax = plt.subplots()
            ax.hist(weather_2018["Wind_Adjusted"], bins=30)
            ax.set_title("Wind Speed Distribution")
            st.pyplot(fig)

        # ==============================
        # 🧩 MODELING (CORE SYSTEM)
        # ==============================
        with tab2:
            st.subheader("System Modeling")

            # -------------------------
            # 1. WIND → POWER MODEL
            # -------------------------
            def wind_to_power(v):
                if v < 3:
                    return 0
                elif v < 12:
                    return (v - 3) / (12 - 3)
                elif v < 25:
                    return 1
                else:
                    return 0

            weather_2018["Power"] = weather_2018["Wind_Adjusted"].apply(wind_to_power)

            st.markdown("### 🌬️ Wind → Power Model")

            v = np.linspace(0, 25, 100)
            power_curve = [wind_to_power(x) for x in v]

            fig, ax = plt.subplots()
            ax.plot(v, power_curve)
            ax.set_title("Wind Turbine Power Curve")
            st.pyplot(fig)

            # -------------------------
            # 2. POWER → STATE MODEL
            # -------------------------
            def get_state(p):
                if p == 0:
                    return 0   # Failure
                elif p < 1:
                    return 1   # Partial
                else:
                    return 2   # Full

            weather_2018["State"] = weather_2018["Power"].apply(get_state)

            st.markdown("### 🔄 System States")
            st.write("0 = Failure, 1 = Partial, 2 = Full")

            st.bar_chart(weather_2018["State"].value_counts())

        # ==============================
        # 🧠 RELIABILITY
        # ==============================
        with tab3:
            st.subheader("Reliability Modeling")

            # -------------------------
            # 3. MARKOV MODEL
            # -------------------------
            states = weather_2018["State"].values
            M = np.zeros((3,3))

            for i in range(len(states)-1):
                M[states[i], states[i+1]] += 1

            M = M / M.sum(axis=1, keepdims=True)

            P = np.array([1/3,1/3,1/3])
            for _ in range(50):
                P = P @ M

            markov_rel = (1 - P[0]) * 100

            # -------------------------
            # 4. FTA (FAILURE MODEL)
            # -------------------------
            lambda_rate = len(fail_df) / 365
            P_fail = 1 - np.exp(-lambda_rate)
            rel_fta = (1 - P_fail) * 100

            # -------------------------
            # 5. MONTE CARLO
            # -------------------------
            simulations = 1000
            mc = []

            for _ in range(simulations):
                sample = np.random.choice(weather_2018["Power"])
                mc.append(sample > 0)

            rel_mc = np.mean(mc) * 100

            # -------------------------
            # DISPLAY
            # -------------------------
            c1, c2, c3 = st.columns(3)
            c1.metric("FTA Reliability", f"{rel_fta:.2f}%")
            c2.metric("Markov Reliability", f"{markov_rel:.2f}%")
            c3.metric("Monte Carlo", f"{rel_mc:.2f}%")

        # ==============================
        # 📈 RESULTS
        # ==============================
        with tab4:
            st.subheader("Final Results")

            # -------------------------
            # LOLP
            # -------------------------
            demand = 0.6
            LOLP = np.mean(weather_2018["Power"] < demand) * 100

            # -------------------------
            # FINAL RELIABILITY
            # -------------------------
            final_rel = (rel_fta + markov_rel + rel_mc) / 3

            st.metric("Final Reliability", f"{final_rel:.2f}%")
            st.metric("LOLP", f"{LOLP:.2f}%")

            if final_rel > 95:
                st.success("System Stable")
            elif final_rel > 85:
                st.warning("Moderate Risk")
            else:
                st.error("Critical System")

    except Exception as e:
        st.error(e)

else:
    st.info("Upload datasets to start")