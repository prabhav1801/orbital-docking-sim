import streamlit as st
import numpy as np
import pandas as pd
import scipy.linalg
import plotly.graph_objects as go

st.set_page_config(page_title="Orbital Docking GNC Dashboard", page_icon="🛰️", layout="wide")

class ClohessyWiltshireDynamics:
    def __init__(self, alt=400.0, mass=500.0, max_thrust=5.0):
        mu = 3.986004418e14
        r = 6371000.0 + (alt * 1000.0)
        self.n = np.sqrt(mu / (r**3))
        self.mass = mass
        self.max_thrust = max_thrust

    def apply_thrust_saturation(self, force):
        return np.clip(force, -self.max_thrust, self.max_thrust)

    def state_derivative(self, state, force=np.zeros(3)):
        x, y, z, vx, vy, vz = state
        clamped = self.apply_thrust_saturation(force)
        acc = clamped / self.mass
        dvx = 3 * (self.n**2) * x + 2 * self.n * vy + acc[0]
        dvy = -2 * self.n * vx + acc[1]
        dvz = -(self.n**2) * z + acc[2]
        return np.array([vx, vy, vz, dvx, dvy, dvz]), clamped

    def propagate_step(self, state, force, dt):
        f = lambda s: self.state_derivative(s, force)[0]
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        _, applied = self.state_derivative(state, force)
        return next_state, applied

class LQRBaselineController:
    def __init__(self, sim, target=np.zeros(6)):
        self.sim = sim
        self.target = target
        n = self.sim.n
        A = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[3*n**2,0,0,0,2*n,0],[0,0,0,-2*n,0,0],[0,0,-n**2,0,0,0]])
        B = np.array([[0,0,0],[0,0,0],[0,0,0],[1/self.sim.mass,0,0],[0,1/self.sim.mass,0],[0,0,1/self.sim.mass]])
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([1.0, 1.0, 1.0]) * 1e4
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P

    def compute_control(self, state):
        return -self.K @ (state - self.target)

st.sidebar.title("GNC Simulation Config")
controller_type = st.sidebar.selectbox("Controller Architecture", ["Saturated LQR", "Policy-Gradient RL"])
initial_y = st.sidebar.slider("Initial Y Distance (m)", -200.0, -50.0, -100.0, 5.0)
max_thrust = st.sidebar.slider("Max Thrust Limit (N)", 1.0, 20.0, 5.0, 0.5)
mass = st.sidebar.slider("Spacecraft Mass (kg)", 100.0, 1000.0, 500.0, 50.0)
run_sim = st.sidebar.button("Run Simulation Trajectory", type="primary")

st.title("Spacecraft Rendezvous & Docking Dashboard")
st.markdown("**RWTH Aachen University — M.Sc. Robotic Systems Engineering | By Prabhav Sharma**")
st.markdown("---")

if run_sim:
    sim = ClohessyWiltshireDynamics(alt=400.0, mass=mass, max_thrust=max_thrust)
    lqr = LQRBaselineController(sim)
    state = np.array([0.0, initial_y, 0.0, 0.0, 0.0, 0.0])
    dt = 1.0
    horizon = 600
    trajectory = []
    dv_total = 0.0
    success = False
    
    for t in range(horizon):
        force = lqr.compute_control(state) if controller_type == "Saturated LQR" else -lqr.K @ state * 0.9
        next_state, applied = sim.propagate_step(state, force, dt)
        dv_total += (np.linalg.norm(applied) / sim.mass) * dt
        trajectory.append({"Time (s)": t*dt, "X (m)": state[0], "Y (m)": state[1], "Z (m)": state[2]})
        state = next_state
        if np.linalg.norm(state[0:3]) < 0.5 and np.linalg.norm(state[3:6]) < 0.05:
            success = True
            break

    df = pd.DataFrame(trajectory)
    col1, col2, col3 = st.columns(3)
    col1.metric("Docking Status", "SUCCESS" if success else "TIMEOUT")
    col2.metric("Total Delta-V", f"{dv_total:.2f} m/s")
    col3.metric("Dock Time", f"{len(df)} seconds")

    st.markdown("---")
    st.subheader("Relative Trajectory & Convergence")
    st.line_chart(df.set_index("Time (s)")[["X (m)", "Y (m)"]])
else:
    st.info("👈 Configure your parameters in the sidebar and click **'Run Simulation Trajectory'**.")