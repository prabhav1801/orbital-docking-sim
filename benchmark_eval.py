"""Monte Carlo comparison of the saturated LQR gain and a 0.88-scaled gain.

Perfect state feedback, no disturbances or sensor noise.
"""
import numpy as np

from src.dynamics.cw_propagation import ClohessyWiltshireDynamics
from src.control.lqr_baseline import LQRBaselineController
from src.simulation import run_episode


def sample_initial_state(rng):
    # Draw order (y, x, z, vx, vy, vz) is kept so that seed 42 reproduces the
    # numbers reported in the README.
    y = rng.uniform(-150.0, -50.0)
    x = rng.uniform(-5.0, 5.0)
    z = rng.uniform(-5.0, 5.0)
    vx = rng.uniform(-0.05, 0.05)
    vy = rng.uniform(-0.05, 0.05)
    vz = rng.uniform(-0.05, 0.05)
    return np.array([x, y, z, vx, vy, vz])


def summarize(name, results):
    n = len(results)
    ok = [r for r in results if r["success"]]
    print(f"{name}")
    print(f"  capture rate:      {100 * len(ok) / n:.1f}%")
    print(f"  mean delta-v:      {np.mean([r['delta_v'] for r in results]):.2f} m/s")
    if ok:
        print(f"  mean capture time: {np.mean([r['capture_time'] for r in ok]):.0f} s")
    print(f"  mean final |pos|:  {100 * np.mean([np.linalg.norm(r['final_state'][:3]) for r in results]):.1f} cm")


def main(num_trials=500, seed=42, horizon=600, dt=1.0):
    sim = ClohessyWiltshireDynamics(alt=400.0, mass=500.0, max_thrust=5.0)
    controllers = {
        "Saturated LQR": LQRBaselineController(sim),
        "LQR gain x 0.88": LQRBaselineController(sim, gain_scale=0.88),
    }
    rng = np.random.RandomState(seed)
    initial_states = [sample_initial_state(rng) for _ in range(num_trials)]

    print(f"{num_trials} trials, seed {seed}, horizon {horizon} s, dt {dt} s")
    for name, ctrl in controllers.items():
        results = [run_episode(sim, ctrl, s0, horizon, dt) for s0 in initial_states]
        summarize(name, results)


if __name__ == "__main__":
    main()
