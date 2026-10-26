# 🛰️ Spacecraft Autonomous Rendezvous & Docking GNC Testbed

> **Author:** Prabhav Sharma | M.Sc. Robotic Systems Engineering (Data & AI Track), RWTH Aachen University  
> **Domain:** Guidance, Navigation, and Control (GNC) | Spacecraft Dynamics | Reinforcement Learning & Optimal Control

---

## 📌 Overview
This repository implements an end-to-end Guidance, Navigation, and Control (GNC) simulation and evaluation framework for autonomous spacecraft orbital rendezvous and docking. 

The framework models relative orbital motion using linear **Clohessy-Wiltshire (Hill's) equations**, enforcing physical actuator constraints (thruster saturation) and comparing a classical **Saturated LQR baseline** against a **Policy-Gradient (RL) controller** through rigorous Monte Carlo verification and an interactive engineering telemetry dashboard.

---

## 🚀 Key Features
* **Rigorous Dynamics & Saturation:** 6-DOF/3-DOF relative orbital motion propagation with Runge-Kutta numerical integration and realistic thrust clipping ($5\,\text{N}$ max limit).
* **Control Architectures:** 
  * Continuous-time Algebraic Riccati Equation (CARE) solved LQR feedback controller.
  * Optimized Policy-Gradient / RL control policy exhibiting enhanced fuel efficiency.
* **500-Trial Monte Carlo Verification (`benchmark_eval.py`):** Automated stress-testing across randomized initial positions, velocities, and disturbances.
* **Interactive Telemetry Dashboard (`app.py`):** A modular Streamlit web app for real-time trajectory visualization, controller toggling, and instant engineering scorecard generation.

---

## 📊 Performance Benchmark Results
Evaluated over **500 randomized Monte Carlo trajectories** (initial along-track distance $-100\,\text{m}$, radial offsets, and velocity noise):

| Metric | Saturated LQR Baseline | Policy-Gradient (RL) Agent |
| :--- | :---: | :---: |
| **Success Rate** | **100.0%** | **100.0%** |
| **Mean $\Delta v$ Consumption** | $1.33\,\text{m/s}$ | **$1.13\,\text{m/s}$** *(~15% savings)* |
| **Terminal Position Error** | $< 0.5\,\text{m}$ | $< 0.5\,\text{m}$ |

---

## 🛠️ Repository Structure
```text
orbital_docking_sim/
├── app.py                        # Interactive Streamlit telemetry dashboard
├── benchmark_eval.py             # 500-trial Monte Carlo evaluation script
└── README.md
```

---

## 💻 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/prabhav1801/orbital-docking-sim.git
cd orbital-docking-sim
```

### 2. Install Dependencies
```bash
pip install streamlit plotly scipy pandas numpy torch
```

### 3. Run the 500-Trial Monte Carlo Benchmark
```bash
python benchmark_eval.py
```

### 4. Launch the Interactive Telemetry Dashboard
```bash
streamlit run app.py
```
*This will spin up a local server at `http://localhost:8501`, allowing you to adjust initial conditions, spacecraft mass, and thruster limits in real time.*