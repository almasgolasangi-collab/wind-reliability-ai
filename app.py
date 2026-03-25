import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Wind Reliability AI Pipeline",
    layout="wide"
)

st.title("🛡️ Wind Turbine Reliability: Advanced Modeling Pipeline")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Controls")

wind_stress_factor = st.sidebar.slider("Wind Load Increase (%)", 0, 50, 0)
cut_in = st.sidebar.slider("Cut-in Wind Speed (m/s)", 1, 5, 3)
cut_out = st.sidebar.slider("Cut-out Wind Speed (m/s)", 15, 30, 20)

# --- FIXED SMART FILE LOADER ---
def load_file(file):
    if file is None:
        return None

    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith(".xlsx"):
            return pd.read_excel(file, engine="openpyxl")
        elif name.endswith(".txt"):
            return pd.read_csv(file, engine='python')
        elif name.endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                for f in z.namelist():
                    if f.endswith(".csv"):
                        return pd.read_csv(z.open(f))
                    elif f.endswith(".xlsx"):
                        return pd.read_excel(z.open(f), engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    return None

# --- FILE UPLOAD ---
st.header("Step 1: Upload Data")

col_u1, col_u2 = st.columns(2)
with col_u1:
    weather_file = st.file_uploader("Upload Weather Data", type=["csv", "xlsx", "txt", "zip"])
with col_u2:
    failure_file = st.file_uploader("Upload Failure Data", type=["csv", "xlsx", "txt"])

if weather_file and failure_file:
    weather_df = load_file(weather_file)
    fail_df = load_file(failure_file)

    if weather_df is not None and fail_df is not None:
        try:
            # --- CLEANING & VALIDATION ---
            weather_df.columns = weather_df.columns.str.strip()
            fail_df.columns = fail_df.columns.str.strip()

            # Process Weather Date (Required for Time Series)
            if 'Date' in weather_df.columns:
                weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')
                weather_df = weather_df.dropna(subset=['Date'])
            else:
                st.error("Critical Error: 'Date' column not found in Weather Data.")
                st.stop()

            # Process Failure Date (Optional fallback to prevent KeyError)
            has_fail_date = 'Date' in fail_df.columns
            if has_fail_date:
                fail_df['Date'] = pd.to_datetime(fail_df['Date'], errors='coerce')
            
            wind_col = 'Wind speed (m/s)'
            if wind_col not in weather_df.columns:
                st.error(f"Column '{wind_col}' not found in weather data.")
                st.stop()

            v_mean = weather_df[wind_col].mean()
            v_std = weather_df[wind_col].std()
            sim_mean = v_mean * (1 + wind_stress_factor / 100)

            # --- TABS ---
            tab1, tab2, tab3, tab4 = st.tabs(["🧹 Data", "📊 EDA", "🧬 Modeling", "🏁 Final"])

            with tab1:
                st.subheader("Raw Weather Data")
                st.dataframe(weather_df.head())
                st.subheader("Raw Failure Data")
                st.dataframe(fail_df.head())

            with tab2:
                st.subheader("📊 Exploratory Data Analysis")
                st.write(weather_df.describe())

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    sns.histplot(weather_df[wind_col], bins=30, kde=True, ax=ax)
                    ax.set_title("Wind Distribution")
                    st.pyplot(fig)
                with c2:
                    fig2, ax2 = plt.subplots()
                    sns.boxplot(x=weather_df[wind_col], ax=ax2)
                    ax2.set_title("Outliers")
                    st.pyplot(fig2)

                # Time series with conditional failure markers
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(weather_df['Date'], weather_df[wind_col], label='Wind Speed')
                
                if has_fail_date:
                    for d in fail_df['Date'].dropna():
                        ax3.axvline(d, color='red', linestyle='--', alpha=0.5)
                    st.caption("🔴 Red dashed lines indicate failure events.")
                else:
                    st.warning("Failure markers skipped: No 'Date' column in failure dataset.")
                
                ax3.set_title("Wind Speed Over Time")
                st.pyplot(fig3)

            with tab3:
                st.subheader("🧬 Reliability Models")
                
                # FTA Logic
                n_fta = 10000
                p_high = np.mean(weather_df[wind_col] > cut_out)
                p_low = np.mean(weather_df[wind_col] < cut_in)
                p_wind = p_high + p_low

                # Check for 'Component' column before filtering
                if 'Component' in fail_df.columns:
                    p_gearbox = max(0.05, len(fail_df[fail_df['Component'] == 'Gearbox']) / 365)
                    p_electrical = max(0.04, len(fail_df[fail_df['Component'] == 'Electrical']) / 365)
                else:
                    p_gearbox, p_electrical = 0.05, 0.04 # Fallback defaults

                failures = 0
                for _ in range(n_fta):
                    event_wind = np.random.rand() < p_wind
                    event_gearbox = np.random.rand() < event_wind # Simplified logic
                    event_electrical = np.random.rand() < p_electrical
                    if (event_wind and event_gearbox) or event_electrical:
                        failures += 1
                
                rel_fta = (1 - failures / n_fta) * 100

                # Monte Carlo
                n_sim = 10000
                samples = np.random.weibull(2, n_sim) * sim_mean
                samples += np.random.normal(0, v_std * 1.8, n_sim)
                safe = np.sum((samples >= cut_in) & (samples <= cut_out))
                rel_mc = (safe / n_sim) * 100

                # Markov
                P = np.array([[0.85, 0.10, 0.05], [0.10, 0.75, 0.15], [0.00, 0.00, 1.00]])
                state = np.array([1, 0, 0])
                for _ in range(10): state = np.dot(state, P)
                rel_markov = (state[0] + state[1]) * 100

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("FTA", f"{rel_fta:.2f}%")
                mc2.metric("Monte Carlo", f"{rel_mc:.2f}%")
                mc3.metric("Markov", f"{rel_markov:.2f}%")

            with tab4:
                st.subheader("🏁 Final Evaluation")
                lolp = np.mean((samples < cut_in) | (samples > cut_out)) * 100
                power = np.where(samples < cut_in, 0, np.where(samples < cut_out, samples**3, 0))
                wpg = np.mean(power)

                st.metric("Loss of Load Prob (LOLP)", f"{lolp:.2f}%")
                st.metric("Wind Power Generation (WPG)", f"{wpg:.2f}")

                status = "STABLE" if rel_fta > 90 else "CRITICAL"
                st.header(f"System Status: {'🟢 STABLE' if status=='STABLE' else '🔴 CRITICAL'}")

        except Exception as e:
            st.error(f"Execution Error: {e}")
else:
    st.info("Please upload both Weather and Failure datasets to proceed.")