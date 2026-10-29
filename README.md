# Orbital rendezvous and docking testbed

Python simulation of terminal spacecraft docking using linear Clohessy-Wiltshire (Hill) relative dynamics, a saturated LQR controller, and a Monte Carlo benchmark. A noisy-camera / Kalman-filter loop, an ArUco pose estimator and an RL environment are included as prototypes.

## Layout

```text
app.py                     Streamlit dashboard (single trajectory, delta-v, capture time)
benchmark_eval.py          500-trial Monte Carlo benchmark
run_lqr.py                 single docking run, perfect state feedback
run_noisy_nav.py           docking run with noisy camera + Kalman filter
train_rl.py                trains the policy-gradient prototype (needs torch)
src/
  dynamics/cw_propagation.py   CW equations, RK4 step, per-axis thrust saturation
  control/lqr_baseline.py      LQR gain from the continuous-time Riccati equation
  simulation.py                episode runner and capture criterion
  vision/camera_sensor.py      noisy full-state camera model, CW Kalman filter
  vision/vision_pose_node.py   ArUco detection + solvePnP pose (assumed intrinsics)
  ros2/docking_nodes.py        ROS 2-style wrappers; no rclpy, publish() prints
  rl/policy_gradient.py        docking environment + actor-critic prototype
```

## Model

- Relative motion in the LVLH frame (x radial, y along-track, z cross-track), 400 km circular orbit, 500 kg spacecraft.
- RK4 integration at dt = 1 s, thrust clipped to ±5 N per axis.
- LQR: Q = diag(100, 100, 100, 10, 10, 10), R = 10⁴·I, gain from the continuous-time ARE, saturation applied in the dynamics.
- Capture: position error < 0.5 m and speed < 0.05 m/s.

## Monte Carlo results

`python benchmark_eval.py`: 500 trials, seed 42, 600 s horizon. Initial conditions: y₀ ∈ [−150, −50] m, x₀, z₀ ∈ ±5 m, each velocity component ∈ ±0.05 m/s. Perfect state feedback, no disturbances or sensor noise.

| Controller | Capture rate | Mean Δv | Mean capture time |
| :--- | :--- | :--- | :--- |
| Saturated LQR | 100.0% | 1.32 m/s | 505 s |
| LQR gain × 0.88 | 100.0% | 1.31 m/s | 519 s |

- The two gains differ by 0.01 m/s in Δv; the scaled gain converges slightly slower.
- Mean capture time is close to the 600 s horizon, so a shorter horizon would lower the capture rate. R = 10⁴ makes the controller slow, and the trajectory overshoots the target by a few metres before settling.
- Mean final position error (≈ 49 cm) just reflects the 0.5 m stopping threshold, not control accuracy.

## Prototypes (not part of the benchmark)

- **Noisy navigation** (`run_noisy_nav.py`): camera model with 0.15 m / 0.02 m/s Gaussian noise and 5% frame dropout, filtered by a Kalman filter that uses the discretised CW model. One seeded run captures at ~516 s; no statistics over many runs yet.
- **Vision pose** (`vision_pose_node.py`): ArUco DICT_4X4_50 + solvePnP with assumed camera intrinsics. Checked on a synthetic marker only; not connected to the control loop.
- **RL** (`src/rl/policy_gradient.py`): actor-critic with a one-step advantage-weighted update. It is not PPO and has not been trained to a validated result or compared with LQR.
- **ROS 2-style nodes**: structural sketch only.

## Running

```bash
pip install -r requirements.txt
python benchmark_eval.py
python run_lqr.py
python run_noisy_nav.py
streamlit run app.py
```
