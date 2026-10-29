"""Single docking run from 100 m behind the target with perfect state feedback."""
import numpy as np

from src.dynamics.cw_propagation import ClohessyWiltshireDynamics
from src.control.lqr_baseline import LQRBaselineController
from src.simulation import run_episode


def main():
    sim = ClohessyWiltshireDynamics(alt=400.0, mass=500.0, max_thrust=5.0)
    controller = LQRBaselineController(sim)
    state0 = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])

    result = run_episode(sim, controller, state0, horizon=600, dt=1.0, record=True)

    print(f"{'t (s)':>6} {'x (m)':>8} {'y (m)':>8} {'z (m)':>8} {'|v| (m/s)':>10}")
    for row in result["trajectory"][::60]:
        t, x, y, z = row[:4]
        print(f"{t:6.0f} {x:8.2f} {y:8.2f} {z:8.2f} {np.linalg.norm(row[4:7]):10.3f}")

    if result["success"]:
        print(f"\nCaptured at t = {result['capture_time']:.0f} s, delta-v = {result['delta_v']:.2f} m/s")
    else:
        print(f"\nNot captured within 600 s, delta-v = {result['delta_v']:.2f} m/s")


if __name__ == "__main__":
    main()
