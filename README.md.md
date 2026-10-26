# Autonomous Spacecraft Rendezvous & Docking Simulation

A modular 6-DOF relative orbital dynamics and GNC simulation framework comparing deep reinforcement learning (PPO) against classical control under hardware constraints.

## Tech Stack & Environment
- **OS**: Tested on Windows 10 & Ubuntu Linux (via dual-boot / ROS 2 nodes)
- **Language**: Python 3.10+ / C++
- **Frameworks**: PyTorch, OpenCV, NumPy, SciPy

## Running the Simulation
- Physics baseline: `python main.py`
- Perception & EKF: `python main_day2.py`
- Telemetry benchmark: `python benchmark_eval.py`