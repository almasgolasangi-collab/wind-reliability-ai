# =====================================
# STREAMLIT APP
# WIND TURBINE RELIABILITY ANALYSIS
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Wind Turbine Reliability",
    layout="wide"
)

st.title("🌬 Wind Turbine Reliability Analysis")

# =====================================
# FILE UPLOAD
# =====================================

st.sidebar.header("📂 Upload Files")

failure_file = st.sidebar.file_uploader(
    "Upload Failure Data CSV",
    type=["csv"]
)

wind_file = st.sidebar.file_uploader(
    "Upload Wind Data CSV",
    type=["csv"]
)

# =====================================
# MAIN PROGRAM
# =====================================

if failure_file and wind_file:

    # =====================================
    # LOAD FAILURE DATA
    # =====================================

    df = pd.read_csv(failure_file)

    df.columns = df.columns.str.strip()

    st.subheader("📊 Failure Data Preview")
    st.write(df.head())

    # =====================================
    # LOAD WIND DATA
    # =====================================

    wind_df = pd.read_csv(
        wind_file,
        skiprows=10
    )

    wind_df.columns = wind_df.columns.str.strip()

    st.subheader("🌬 Wind Data Preview")
    st.write(wind_df.head())

    st.write("Wind Columns:", wind_df.columns.tolist())

    # =====================================
    # WIND COLUMN
    # =====================================

    wind_col = "WS50M"

    if wind_col not in wind_df.columns:

        st.error("❌ WS50M column not found")
        st.stop()

    wind_values = wind_df[wind_col].dropna().values

    # =====================================
    # DATE CLEANING
    # =====================================

    date_col = None

    for col in df.columns:

        if "date" in col.lower():

            date_col = col
            break

    if date_col is None:

        st.error("❌ Date column not found")
        st.stop()

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors='coerce'
    )

    df = df.dropna(subset=[date_col])

    # =====================================
    # COMPONENT CLASSIFICATION
    # =====================================

    def classify(row):

        text = " ".join(
            map(str, row)
        ).lower()

        if "bearing" in text:
            return "Bearing"

        elif "gear" in text:
            return "Gear"

        elif "oil" in text or "lubrication" in text:
            return "Lubrication"

        else:
            return "Other"

    df["Component"] = df.apply(
        classify,
        axis=1
    )

    # =====================================
    # COMPONENT DISTRIBUTION
    # =====================================

    st.subheader("📊 Component Distribution")

    counts = df["Component"].value_counts()

    st.write(counts)

    fig1, ax1 = plt.subplots()

    counts.plot(
        kind='bar',
        ax=ax1
    )

    ax1.set_xlabel("Component")
    ax1.set_ylabel("Failure Count")
    ax1.set_title(
        "Failure Count by Component"
    )

    st.pyplot(fig1)

    # =====================================
    # TOTAL OPERATING TIME
    # =====================================

    total_hours = (
        (df[date_col].max() - df[date_col].min()).days
    ) * 24

    # =====================================
    # FAILURE RATE (λ)
    # =====================================

    lambda_base = {

        comp: max(
            counts.get(comp, 0) / total_hours,
            1 / (2 * total_hours)
        )

        for comp in [
            "Bearing",
            "Gear",
            "Lubrication"
        ]
    }

    # =====================================
    # REPAIR RATE (μ)
    # =====================================

    mu_dict = {

        # 3 DAYS
        "Bearing": 1 / (3 * 24),

        # 5 DAYS
        "Gear": 1 / (5 * 24),

        # 2 DAYS
        "Lubrication": 1 / (2 * 24)
    }

    # =====================================
    # FTA MODEL
    # =====================================

    t = 200

    R_fta = {

        c: np.exp(
            -lambda_base[c] * t
        )

        for c in lambda_base
    }

    R_fta_sys = np.prod(
        list(R_fta.values())
    )

    A_fta = {

        c: mu_dict[c] / (
            lambda_base[c] + mu_dict[c]
        )

        for c in lambda_base
    }

    A_fta_sys = np.prod(
        list(A_fta.values())
    )

    # =====================================
    # MARKOV MODEL
    # =====================================

    wind_norm = (

        wind_values - np.min(wind_values)

    ) / (

        np.max(wind_values) - np.min(wind_values)

    )

    avg_wind = np.mean(wind_norm)

    k = 0.05

    lambda_markov = {

        c: lambda_base[c] * (
            1 + k * avg_wind
        )

        for c in lambda_base
    }

    R_markov = {

        c: np.exp(
            -lambda_markov[c] * t
        )

        for c in lambda_markov
    }

    R_markov_sys = np.prod(
        list(R_markov.values())
    )

    # =====================================
    # MONTE CARLO SIMULATION
    # =====================================

    simulation_time = 8760   # 1 YEAR

    num_sim = 300

    cut_in = 5
    cut_out = 20

    LOLE_list = []

    uptime_list = []

    for sim in range(num_sim):

        downtime = 0

        for comp in lambda_base:

            # =====================================
            # INCREASED FAILURE RATE
            # =====================================

            lam_base = lambda_base[comp] * 3

            mu = mu_dict[comp]

            t_sim = 0

            state = 1

            while t_sim < simulation_time:

                wind = np.random.choice(
                    wind_values
                )

                # =====================================
                # WIND EFFECT
                # =====================================

                if wind < cut_in or wind > cut_out:

                    lam = lam_base * 1.5

                else:

                    lam = lam_base

                # =====================================
                # WORKING STATE
                # =====================================

                if state == 1:

                    ttf = np.random.exponential(
                        1 / lam
                    )

                    if t_sim + ttf >= simulation_time:
                        break

                    t_sim += ttf

                    state = 0

                # =====================================
                # FAILED STATE
                # =====================================

                else:

                    ttr = np.random.exponential(
                        1 / mu
                    )

                    if t_sim + ttr >= simulation_time:

                        downtime += (
                            simulation_time - t_sim
                        )

                        break

                    t_sim += ttr

                    downtime += ttr

                    state = 1

        uptime = simulation_time - downtime

        reliability = uptime / simulation_time

        uptime_list.append(reliability)

        LOLE_list.append(downtime)

    # =====================================
    # MONTE CARLO RESULTS
    # =====================================

    R_mc = np.mean(uptime_list)

    LOLE_avg = np.mean(
        LOLE_list
    )

    LOLP = LOLE_avg / simulation_time

    # =====================================
    # RESULTS DISPLAY
    # =====================================

    st.subheader("📊 Reliability Results")

    col1, col2, col3 = st.columns(3)

    # FTA

    col1.metric(
        "FTA Reliability",
        f"{R_fta_sys*100:.2f}%"
    )

    col1.metric(
        "FTA Availability",
        f"{A_fta_sys*100:.2f}%"
    )

    # MARKOV

    col2.metric(
        "Markov Reliability",
        f"{R_markov_sys*100:.2f}%"
    )

    # MONTE CARLO

    col3.metric(
        "Monte Carlo Reliability",
        f"{R_mc*100:.2f}%"
    )

    col3.metric(
        "LOLE (hours/year)",
        f"{LOLE_avg:.2f}"
    )

    col3.metric(
        "LOLP",
        f"{LOLP:.4f}"
    )

    # =====================================
    # RELIABILITY COMPARISON GRAPH
    # =====================================

    st.subheader("📈 Reliability Comparison")

    methods = [
        "FTA",
        "Markov",
        "Monte Carlo"
    ]

    values = [
        R_fta_sys * 100,
        R_markov_sys * 100,
        R_mc * 100
    ]

    fig2, ax2 = plt.subplots()

    ax2.plot(
        methods,
        values,
        marker='o',
        linewidth=2
    )

    ax2.set_ylabel(
        "Reliability (%)"
    )

    ax2.set_title(
        "Reliability Comparison"
    )

    ax2.grid(True)

    st.pyplot(fig2)

else:

    st.info(
        "⬅ Upload both failure and wind datasets"
    )