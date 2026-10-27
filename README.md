# Autonomous Orbital Rendezvous & Docking GNC Testbed 🛰️⚙️

A Python simulation testbed for spacecraft proximity operations and terminal docking, based on linear Clohessy–Wiltshire (Hill's) relative orbital dynamics, a saturated LQR controller and Monte Carlo verification.

---

## 📂 Project Structure

```text
.
├── app.py                  # Streamlit dashboard (trajectory plots, mission metrics)
├── benchmark_eval.py       # 500-trial Monte Carlo benchmark (self-contained)
├── main.py                 # Day 1: single LQR docking run
├── main_day2.py            # Day 2: noisy-camera + Kalman filter run (prototype)
├── train.py                # Entry point for the RL prototype
└── src/
    ├── dynamics/cw_propagation.py   # CW equations, RK4, per-axis thrust saturation
    ├── control/lqr_baseline.py      # LQR gain from the continuous-time ARE (CARE)
    ├── vision/
    │   ├── camera_sensor.py         # Noisy camera model (5% dropout), Kalman filter class
    │   └── vision_pose_node.py      # ArUco (DICT_4X4_50) detection + solvePnP pose
    ├── ros2/docking_nodes.py        # ROS 2-style node wrappers (mock publish layer, no rclpy)
    └── ai/ppo_docking.py            # Actor-critic RL environment/training prototype
```

---

## 🛠️ Engineering Overview

- **Dynamics:** linear CW relative motion in the LVLH frame (x radial, y along-track, z cross-track), 400 km circular orbit, RK4 integration at dt = 1 s, thrust clipped to ±5 N per axis, 500 kg spacecraft.
- **Control:** LQR gain from the continuous-time Riccati equation (Q = diag(100,100,100,10,10,10), R = 10⁴·I) with thrust saturation applied in the dynamics.
- **Heuristic baseline:** the LQR gain scaled by 0.88 (untrained, used only for comparison).
- **Vision (prototype):** ArUco marker detection and `solvePnP` relative pose with assumed camera intrinsics. Not connected to the benchmark.
- **ROS 2-style wrappers:** classes that mimic node structure and print to stdout. No `rclpy` dependency.
- **Dashboard:** `app.py` plots relative X and Y against time, with dock status, total Δv and dock time. Initial distance, thrust limit and mass are adjustable.

---

## 📊 Monte Carlo Verification

500 randomized trials (`benchmark_eval.py`, seed `42`, dt = 1 s, 600 s horizon):

| Parameter | Range |
| :--- | :--- |
| Along-track offset y₀ | [−150, −50] m |
| Radial / cross-track offsets x₀, z₀ | ±5 m |
| Initial velocities | ±0.05 m/s per axis |

**Capture criterion:** position error < 0.5 m **and** speed < 0.05 m/s. A trial is a success if this is met within 600 s.

| Controller | Capture rate | Mean Δv | Mean capture time |
| :--- | :--- | :--- | :--- |
| Saturated LQR | 100.0% | 1.32 m/s | ~505 s |
| LQR gain × 0.88 (heuristic) | 100.0% | 1.31 m/s | ~519 s |

Notes:
- The two controllers show no meaningful Δv difference (0.01 m/s). The scaled gain converges slightly slower.
- Mean capture time is close to the 600 s window, so a shorter horizon would reduce the success rate.
- Mean terminal position error is ≈ 49 cm in both cases. This only reflects the 0.5 m stopping threshold, not control accuracy.
- Perturbations, sensor noise and estimator error are **not** included in this benchmark.

---

## 🧪 Prototype Status

These modules are experimental and **not part of the benchmark or dashboard**:

- `src/vision/camera_sensor.py`: Gaussian noise and 5% dropout model, plus a Kalman filter with an identity process model (no CW dynamics). Not validated in closed loop.
- `src/ai/ppo_docking.py`: actor-critic environment with a simple policy-gradient-style update. It is not PPO and has not been run to a validated result.
- `main_day2.py`: noisy-vision run script. It has not been run end-to-end after refactoring.

---

## 🚀 Getting Started

```bash
pip install numpy scipy pandas plotly streamlit
pip install opencv-python torch    # only for the vision / RL prototypes
```

Run the benchmark:
```bash
python benchmark_eval.py
```

Launch the dashboard:
```bash
streamlit run app.py
```
