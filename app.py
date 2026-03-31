# =====================================
# STREAMLIT APP: WIND TURBINE RELIABILITY
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wind Turbine Reliability", layout="wide")

st.title("🌬 Wind Turbine Reliability Analysis")

# ---------------------------
# FILE UPLOAD
# ---------------------------
st.sidebar.header("Upload Data")

failure_file = st.sidebar.file_uploader("Upload Failure Data (CSV)", type=["csv"])
wind_file = st.sidebar.file_uploader("Upload Wind Data (CSV)", type=["csv"])

if failure_file and wind_file:

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    df = pd.read_csv(failure_file)
    wind_df = pd.read_csv(wind_file)

    df.columns = df.columns.str.strip()
    wind_df.columns = wind_df.columns.str.strip().str.lower()

    st.subheader("📊 Raw Data Preview")
    st.write(df.head())

    # ---------------------------
    # CLEANING
    # ---------------------------
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    # ---------------------------
    # CLASSIFICATION
    # ---------------------------
    def classify(row):
        text = " ".join(map(str, row)).lower()
        if "bearing" in text:
            return "Bearing"
        elif "gear" in text:
            return "Gear"
        elif "oil" in text or "lubrication" in text:
            return "Lubrication"
        else:
            return "Other"

    df["Component"] = df.apply(classify, axis=1)

    # ---------------------------
    # EDA
    # ---------------------------
    st.subheader("📊 Component Distribution")
    counts = df["Component"].value_counts()
    st.write(counts)

    fig1, ax1 = plt.subplots()
    counts.plot(kind='bar', ax=ax1)
    ax1.set_title("Failure Count by Component")
    st.pyplot(fig1)

    # ---------------------------
    # FAILURE RATE
    # ---------------------------
    total_hours = (df[date_col].max() - df[date_col].min()).days * 24

    lambda_base = {
        comp: max(counts.get(comp, 0) / total_hours, 1/(2*total_hours))
        for comp in ["Bearing", "Gear", "Lubrication"]
    }

    mu_dict = {
        "Bearing": 1/(5*24),
        "Gear": 1/(7*24),
        "Lubrication": 1/(2*24)
    }

    # =========================================================
    # 🔴 FTA
    # =========================================================
    t = 200

    R_fta = {c: np.exp(-lambda_base[c]*t) for c in lambda_base}
    R_fta_sys = np.prod(list(R_fta.values()))

    A_fta = {c: mu_dict[c]/(lambda_base[c]+mu_dict[c]) for c in lambda_base}
    A_fta_sys = np.prod(list(A_fta.values()))

    # =========================================================
    # 🔴 MARKOV
    # =========================================================
    wind_col = [c for c in wind_df.columns if "wind" in c][0]
    wind_values = wind_df[wind_col].dropna().values

    wind_norm = (wind_values - np.min(wind_values)) / (np.max(wind_values) - np.min(wind_values))
    avg_wind = np.mean(wind_norm)

    k = 0.05

    lambda_markov = {
        c: lambda_base[c] * (1 + k * avg_wind)
        for c in lambda_base
    }

    R_markov = {c: np.exp(-lambda_markov[c]*t) for c in lambda_markov}
    R_markov_sys = np.prod(list(R_markov.values()))

    # =========================================================
    # 🔴 MONTE CARLO
    # =========================================================
    simulation_time = 200
    num_sim = 200

    cut_in, cut_out = 5, 20

    success = 0
    LOLE_list = []

    for sim in range(num_sim):

        failed_once = False
        downtime = 0

        for comp in lambda_base:

            lam_base = lambda_base[comp]
            mu = mu_dict[comp]

            t_sim = 0
            state = 1

            while t_sim < simulation_time:

                wind = np.random.choice(wind_values)

                if wind < cut_in or wind > cut_out:
                    lam = lam_base * 2
                else:
                    lam = lam_base

                if state == 1:
                    ttf = np.random.exponential(1/lam)

                    if t_sim + ttf >= simulation_time:
                        break

                    t_sim += ttf
                    state = 0
                    failed_once = True

                else:
                    ttr = np.random.exponential(1/mu)

                    if t_sim + ttr >= simulation_time:
                        downtime += (simulation_time - t_sim)
                        break

                    t_sim += ttr
                    downtime += ttr
                    state = 1

        if not failed_once:
            success += 1

        LOLE_list.append(downtime)

    R_mc = success / num_sim
    LOLE_avg = np.mean(LOLE_list)

    # =========================================================
    # 📊 RESULTS
    # =========================================================
    st.subheader("📊 Reliability Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("FTA Reliability", f"{R_fta_sys*100:.2f}%")
    col1.metric("FTA Availability", f"{A_fta_sys*100:.2f}%")

    col2.metric("Markov Reliability", f"{R_markov_sys*100:.2f}%")

    col3.metric("Monte Carlo Reliability", f"{R_mc*100:.2f}%")
    col3.metric("LOLE (hours)", f"{LOLE_avg:.2f}")

    # ---------------------------
    # COMPARISON CHART
    # ---------------------------
    st.subheader("📊 Method Comparison")

    methods = ["FTA", "Markov", "Monte Carlo"]
    values = [R_fta_sys*100, R_markov_sys*100, R_mc*100]

    fig2, ax2 = plt.subplots()
    ax2.bar(methods, values)
    ax2.set_ylabel("Reliability (%)")
    ax2.set_title("Reliability Comparison")
    st.pyplot(fig2)

else:
    st.info("⬅ Upload both files to begin analysis")