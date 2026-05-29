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
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Wind Turbine Reliability Dashboard",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("Wind Turbine Reliability Dashboard")

st.markdown("""
This dashboard performs:

- Fault Tree Analysis (FTA)
- Markov Chain Reliability
- Monte Carlo Simulation
- LOLE and LOLP Analysis
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Upload Files")

wind_file = st.sidebar.file_uploader(
    "Upload Wind CSV File",
    type=["csv"]
)

component_file = st.sidebar.file_uploader(
    "Upload Component Excel File",
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
    # FAILURE RATE
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
    # TOP METRICS
    # ========================================================

    st.subheader("Wind Turbine Reliability")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "FTA Reliability",
        f"{R_fta_system * 100:.2f}%"
    )

    col2.metric(
        "Markov Reliability",
        f"{R_markov_system * 100:.2f}%"
    )

    col3.metric(
        "Monte Carlo Reliability",
        f"{MonteCarlo_Reliability * 100:.2f}%"
    )

    col4.metric(
        "FTA Availability",
        f"{A_fta_system * 100:.2f}%"
    )

    col5.metric(
        "LOLE (hours)",
        f"{LOLE:.2f}"
    )

    # ========================================================
    # RELIABILITY COMPARISON GRAPH
    # ========================================================

    st.subheader("Method Comparison")

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

    fig1, ax1 = plt.subplots(figsize=(9,6))

    bars = ax1.bar(

        methods,

        values,

        width=0.8
    )

    # VALUE LABELS

    for bar in bars:

        height = bar.get_height()

        ax1.text(

            bar.get_x() + bar.get_width()/2,

            height + 0.5,

            f"{height:.2f}%",

            ha='center',

            fontsize=12
        )

    ax1.set_ylabel(
        "Reliability (%)",
        fontsize=14
    )

    ax1.set_xlabel(
        "Methods",
        fontsize=13
    )

    ax1.set_title(
        "Reliability Comparison",
        fontsize=20
    )

    ax1.set_ylim(0, 100)

    ax1.grid(

        axis='y',

        linestyle='--',

        alpha=0.7
    )

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

    ax2.grid(axis='y')

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

    fig4, ax4 = plt.subplots(figsize=(6,4))

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

    ax4.grid(axis='y')

    st.pyplot(fig4)

    # ========================================================
    # LOLP GRAPH
    # ========================================================

    st.subheader("LOLP")

    fig5, ax5 = plt.subplots(figsize=(6,4))

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

    ax5.grid(axis='y')

    st.pyplot(fig5)

    # ========================================================
    # FINAL RESULTS TABLE
    # ========================================================

    st.subheader("Final Results")

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

    st.dataframe(results)

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