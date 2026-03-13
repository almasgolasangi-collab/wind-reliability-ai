import streamlit as st

def show_machine_specs():
    st.sidebar.header("⚙️ Machine Specifications")
    st.sidebar.write("**Gearbox Type:** 3-Stage (1 Planetary + 2 Helical)")
    st.sidebar.write("**Lubrication:** ISO VG 320 Synthetic")
    st.sidebar.write("**Reference:** NREL Gearbox Reliability Collaborative")
    st.sidebar.divider()