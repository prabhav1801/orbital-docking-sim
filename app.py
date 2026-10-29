"""Streamlit dashboard: run one docking trajectory and plot it."""
import pandas as pd
import streamlit as st

from src.dynamics.cw_propagation import ClohessyWiltshireDynamics
from src.control.lqr_baseline import LQRBaselineController
from src.simulation import run_episode

st.set_page_config(page_title="Orbital Docking GNC Dashboard", layout="wide")

st.sidebar.title("Simulation config")
controller_name = st.sidebar.selectbox("Controller", ["Saturated LQR", "LQR gain x 0.88"])
initial_y = st.sidebar.slider("Initial along-track offset (m)", -200.0, -50.0, -100.0, 5.0)
max_thrust = st.sidebar.slider("Max thrust per axis (N)", 1.0, 20.0, 5.0, 0.5)
mass = st.sidebar.slider("Spacecraft mass (kg)", 100.0, 1000.0, 500.0, 50.0)
run = st.sidebar.button("Run", type="primary")

st.title("Spacecraft rendezvous and docking")
st.caption("Clohessy-Wiltshire dynamics, 400 km circular orbit, perfect state feedback")

if run:
    sim = ClohessyWiltshireDynamics(alt=400.0, mass=mass, max_thrust=max_thrust)
    gain_scale = 1.0 if controller_name == "Saturated LQR" else 0.88
    controller = LQRBaselineController(sim, gain_scale=gain_scale)

    result = run_episode(sim, controller, [0.0, initial_y, 0.0, 0.0, 0.0, 0.0],
                         horizon=600, dt=1.0, record=True)
    df = pd.DataFrame(result["trajectory"][:, :4], columns=["Time (s)", "X (m)", "Y (m)", "Z (m)"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", "Captured" if result["success"] else "Timeout")
    c2.metric("Total delta-v", f"{result['delta_v']:.2f} m/s")
    c3.metric("Capture time", f"{result['capture_time']:.0f} s" if result["success"] else "-")

    st.subheader("Relative position")
    st.line_chart(df.set_index("Time (s)")[["X (m)", "Y (m)"]])
else:
    st.info("Set the parameters in the sidebar and press Run.")
