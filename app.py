# ============================================================
# STREAMLIT DASHBOARD
# WIND TURBINE RELIABILITY ANALYSIS
# FTA + MARKOV + MONTE CARLO
# ============================================================

# RUN USING:
# streamlit run app.py

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Wind Turbine Reliability Dashboard",
    layout="wide"
)

st.title("Wind Turbine Reliability Dashboard")

st.markdown("""
This dashboard performs reliability analysis using:

- Fault Tree Analysis (FTA)
- Markov Chain Analysis
- Monte Carlo Simulation
""")

# ============================================================
# FILE UPLOAD
# ============================================================

st.sidebar.header("Upload Files")

wind_file = st.sidebar.file_uploader(
    "Upload Wind Data CSV",
    type=["csv"]
)

component_file = st.sidebar.file_uploader(
    "Upload Component Failure Excel",
    type=["xlsx"]
)

# ============================================================
# CHECK FILES
# ============================================================

if wind_file is not None and component_file is not None:

    # ========================================================
    # LOAD DATA
    # ========================================================

    wind_df = pd.read_csv(wind_file)

    comp_df = pd.read_excel(component_file)

    # ========================================================
    # DISPLAY DATA
    # ========================================================

    st.subheader("Wind Data")

    st.dataframe(wind_df.head())

    st.subheader("Component Failure Data")

    st.dataframe(comp_df.head())

    # ========================================================
    # WIND COLUMN
    # ========================================================

    wind_col = "WS80M"

    wind_values = wind_df[
        wind_col
    ].dropna().values

    # ========================================================
    # COMPONENT PARAMETERS
    # ========================================================

    components = comp_df["Component"]

    failure_rate_year = comp_df[
        "Failure_Rate_per_Year"
    ].values

    MTTR = comp_df[
        "MTTR_Hours"
    ].values

    # ========================================================
    # FAILURE RATE CONVERSION
    # ========================================================

    lambda_hour = (
        failure_rate_year / 8760
    )

    # ========================================================
    # REPAIR RATE
    # ========================================================

    mu = 1 / MTTR

    # ========================================================
    # PARAMETER TABLE
    # ========================================================

    parameter_df = pd.DataFrame({

        "Component": components,

        "Failure Rate per Hour":
            lambda_hour,

        "Repair Rate":
            mu
    })

    st.subheader("System Parameters")

    st.dataframe(parameter_df)

    # ========================================================
    # FTA RELIABILITY
    # ========================================================

    mission_time = 200

    R_components = np.exp(
        -lambda_hour * mission_time
    )

    R_fta_system = np.prod(
        R_components
    )

    # ========================================================
    # FTA AVAILABILITY
    # ========================================================

    A_components = mu / (
        lambda_hour + mu
    )

    A_fta_system = np.prod(
        A_components
    )

    # ========================================================
    # MARKOV RELIABILITY
    # ========================================================

    wind_norm = (

        wind_values - np.min(wind_values)

    ) / (

        np.max(wind_values)
        - np.min(wind_values)

    )

    avg_wind = np.mean(
        wind_norm
    )

    k = 0.05

    lambda_markov = (
        lambda_hour * (
            1 + k * avg_wind
        )
    )

    R_markov_components = np.exp(
        -lambda_markov * mission_time
    )

    R_markov_system = np.prod(
        R_markov_components
    )

    # ========================================================
    # MONTE CARLO AVAILABILITY
    # ========================================================

    simulation_time = 8760

    num_sim = 300

    cut_in = 5
    cut_out = 25

    LOLE_list = []

    availability_list = []

    for sim in range(num_sim):

        total_downtime = 0

        for i in range(len(components)):

            lam_base = lambda_hour[i]

            repair_rate = mu[i]

            t = 0

            state = 1

            downtime = 0

            while t < simulation_time:

                wind = np.random.choice(
                    wind_values
                )

                if wind < cut_in or wind > cut_out:

                    lam = lam_base * 1.1

                else:

                    lam = lam_base

                # WORKING STATE

                if state == 1:

                    ttf = np.random.exponential(
                        1 / lam
                    )

                    if t + ttf >= simulation_time:
                        break

                    t += ttf

                    state = 0

                # FAILED STATE

                else:

                    ttr = np.random.exponential(
                        1 / repair_rate
                    )

                    if t + ttr >= simulation_time:

                        downtime += (
                            simulation_time - t
                        )

                        break

                    t += ttr

                    downtime += ttr

                    state = 1

            total_downtime += downtime

        availability = (

            simulation_time
            - total_downtime

        ) / simulation_time

        availability_list.append(
            availability
        )

        LOLE_list.append(
            total_downtime
        )

    MonteCarlo_Availability = np.mean(
        availability_list
    )

    LOLE = np.mean(
        LOLE_list
    )

    LOLP = LOLE / simulation_time

    # ========================================================
    # MONTE CARLO RELIABILITY
    # ========================================================

    mission_time_mc = 300

    success_count = 0

    num_sim_rel = 300

    for sim in range(num_sim_rel):

        failed = False

        for i in range(len(components)):

            lam_base = lambda_hour[i]

            t = 0

            while t < mission_time_mc:

                wind = np.random.choice(
                    wind_values
                )

                if wind < cut_in or wind > cut_out:

                    lam = lam_base * 1.1

                else:

                    lam = lam_base

                ttf = np.random.exponential(
                    1 / lam
                )

                t += ttf

                if t < mission_time_mc:

                    failed = True
                    break

            if failed:
                break

        if not failed:

            success_count += 1

    MonteCarlo_Reliability = (
        success_count / num_sim_rel
    )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results = pd.DataFrame({

        "Method": [

            "FTA Reliability",

            "FTA Availability",

            "Markov Reliability",

            "Monte Carlo Reliability",

            "Monte Carlo Availability"
        ],

        "Value (%)": [

            R_fta_system * 100,

            A_fta_system * 100,

            R_markov_system * 100,

            MonteCarlo_Reliability * 100,

            MonteCarlo_Availability * 100
        ]
    })

    st.subheader("Final Results")

    st.dataframe(results)

    # ========================================================
    # RELIABILITY GRAPH
    # ========================================================

    st.subheader("Reliability Comparison")

    methods = [

        "FTA",

        "Markov",

        "Monte Carlo"
    ]

    values = [

        R_fta_system * 100,

        R_markov_system * 100,

        MonteCarlo_Reliability * 100
    ]

    fig1, ax1 = plt.subplots(figsize=(8,5))

    ax1.plot(

        methods,

        values,

        marker='o',

        linewidth=3
    )

    ax1.set_ylabel("Reliability (%)")

    ax1.set_title(
        "Reliability Comparison"
    )

    ax1.grid(True)

    st.pyplot(fig1)

    # ========================================================
    # COMPONENT FAILURE GRAPH
    # ========================================================

    st.subheader("Component Failure Rates")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.bar(

        components,

        failure_rate_year
    )

    ax2.set_ylabel(
        "Failure Rate per Year"
    )

    ax2.set_title(
        "Component Failure Rates"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig2)

    # ========================================================
    # WIND SPEED GRAPH
    # ========================================================

    st.subheader("Wind Speed Variation")

    fig3, ax3 = plt.subplots(figsize=(12,5))

    ax3.plot(
        wind_values[:500]
    )

    ax3.set_ylabel(
        "Wind Speed (m/s)"
    )

    ax3.set_xlabel(
        "Time Index"
    )

    ax3.set_title(
        "Wind Speed Variation"
    )

    ax3.grid(True)

    st.pyplot(fig3)

    # ========================================================
    # LOLE GRAPH
    # ========================================================

    st.subheader("LOLE")

    fig4, ax4 = plt.subplots(figsize=(5,4))

    ax4.bar(
        ["LOLE"],
        [LOLE]
    )

    ax4.set_ylabel(
        "Hours/Year"
    )

    ax4.set_title(
        "Loss of Load Expectation"
    )

    st.pyplot(fig4)

    # ========================================================
    # LOLP GRAPH
    # ========================================================

    st.subheader("LOLP")

    fig5, ax5 = plt.subplots(figsize=(5,4))

    ax5.bar(
        ["LOLP"],
        [LOLP]
    )

    ax5.set_ylabel(
        "Probability"
    )

    ax5.set_title(
        "Loss of Load Probability"
    )

    st.pyplot(fig5)

    # ========================================================
    # CONCLUSION
    # ========================================================

    st.subheader("Conclusion")

    st.markdown("""

    1. FTA provides analytical reliability estimation.

    2. Markov method includes probabilistic
       state transitions and wind effect.

    3. Monte Carlo Reliability evaluates
       probability of no failure during
       mission time.

    4. Monte Carlo Availability evaluates
       operational performance considering
       failures and repairs.

    5. Monte Carlo simulation provides
       realistic reliability assessment
       under varying wind conditions.
    """)

else:

    st.info(
        "Please upload both files to continue."
    )