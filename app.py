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
cut_out = st.sidebar.slider("Cut-out Wind Speed (m/s)", 15, 30, 20)

# --- FIXED SMART FILE LOADER ---
def load_file(file):
    import pandas as pd
    import zipfile

    if file is None:
        return None

    name = file.name.lower()

    try:
        # CSV
        if name.endswith(".csv"):
            return pd.read_csv(file)

        # EXCEL (FIXED)
        elif name.endswith(".xlsx"):
            return pd.read_excel(file, engine="openpyxl")

        # TXT
        elif name.endswith(".txt"):
            return pd.read_csv(file, engine='python')

        # ZIP
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

    st.error("Unsupported format")
    return None

# --- FILE UPLOAD ---
st.header("Step 1: Upload Data")

weather_file = st.file_uploader("Upload Weather Data", type=["csv", "xlsx", "txt", "zip"])
failure_file = st.file_uploader("Upload Failure Data", type=["csv", "xlsx", "txt"])

if weather_file and failure_file:

    weather_df = load_file(weather_file)
    fail_df = load_file(failure_file)

    if weather_df is None or fail_df is None:
        st.stop()

    try:
        # --- CLEANING ---
        weather_df.columns = weather_df.columns.str.strip()
        fail_df.columns = fail_df.columns.str.strip()

        # DEBUG VIEW (VERY IMPORTANT)
        st.write("Weather Columns:", weather_df.columns)
        st.write("Failure Columns:", fail_df.columns)

        weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')
        fail_df['Date'] = pd.to_datetime(fail_df['Date'], errors='coerce')

        weather_df = weather_df.dropna(subset=['Date'])

        wind_col = 'Wind speed (m/s)'

        v_mean = weather_df[wind_col].mean()
        v_std = weather_df[wind_col].std()

        sim_mean = v_mean * (1 + wind_stress_factor / 100)

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🧹 Data", "📊 EDA", "🧬 Modeling", "🏁 Final"]
        )

        # =========================
        # TAB 1: DATA
        # =========================
        with tab1:
            st.dataframe(weather_df.head())

        # =========================
        # TAB 2: ADVANCED EDA
        # =========================
        with tab2:
            st.subheader("📊 Exploratory Data Analysis")

            st.write(weather_df.describe())

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                sns.histplot(weather_df[wind_col], bins=30, kde=True, ax=ax)
                ax.set_title("Wind Distribution")
                st.pyplot(fig)

            with col2:
                fig2, ax2 = plt.subplots()
                sns.boxplot(x=weather_df[wind_col], ax=ax2)
                ax2.set_title("Outliers")
                st.pyplot(fig2)

            # Time series
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(weather_df['Date'], weather_df[wind_col])

            for d in fail_df['Date']:
                ax3.axvline(d, color='red', linestyle='--', alpha=0.5)

            ax3.set_title("Wind vs Failures")
            st.pyplot(fig3)

        # =========================
        # TAB 3: MODELING
        # =========================
        with tab3:

            st.subheader("🧬 Reliability Models")

            # 🔴 FTA
            n_fta = 10000
            failures = 0

            p_high = np.mean(weather_df[wind_col] > cut_out)
            p_low = np.mean(weather_df[wind_col] < cut_in)
            p_wind = p_high + p_low

            p_gearbox = max(0.05, len(fail_df[fail_df['Component'] == 'Gearbox']) / 365)
            p_electrical = max(0.04, len(fail_df[fail_df['Component'] == 'Electrical']) / 365)

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
            samples += np.random.normal(0, v_std * 1.8, n_sim)

            # extreme wind injection
            idx = np.random.choice(n_sim, int(0.08 * n_sim))
            samples[idx] *= 2

            safe = np.sum((samples >= cut_in) & (samples <= cut_out))
            rel_mc = (safe / n_sim) * 100

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

            st.metric("LOLP", f"{lolp:.2f}%")
            st.metric("WPG", f"{wpg:.2f}")

            status = "STABLE" if rel_fta > 90 else "CRITICAL"
            st.header(f"Final: {'🟢 STABLE' if status=='STABLE' else '🔴 CRITICAL'}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload both datasets to begin.")