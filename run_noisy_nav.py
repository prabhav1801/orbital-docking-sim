"""Docking run with the LQR driven by a Kalman-filtered noisy camera measurement."""
import numpy as np

from src.dynamics.cw_propagation import ClohessyWiltshireDynamics
from src.control.lqr_baseline import LQRBaselineController
from src.simulation import is_captured
from src.vision.camera_sensor import OnboardCameraSensor, CWKalmanFilter


def main(seed=0, horizon=600, dt=1.0):
    sim = ClohessyWiltshireDynamics(alt=400.0, mass=500.0, max_thrust=5.0)
    controller = LQRBaselineController(sim)
    camera = OnboardCameraSensor(noise_pos=0.15, noise_vel=0.02, dropout_p=0.05,
                                 rng=np.random.default_rng(seed))

    true_state = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    kf = CWKalmanFilter(controller.A, controller.B, dt, x0=true_state)

    applied = np.zeros(3)
    dropouts = 0
    print(f"{'t (s)':>6} {'true y':>9} {'est y':>9}  frame")
    for k in range(horizon):
        measured, dropped = camera.get_measurement(true_state)
        dropouts += dropped
        kf.predict(applied)
        est = kf.update(measured, dropped)

        force = controller.compute_control(est)
        true_state, applied = sim.propagate_step(true_state, force, dt)

        if k % 60 == 0:
            print(f"{k * dt:6.0f} {true_state[1]:9.2f} {est[1]:9.2f}  {'dropout' if dropped else 'ok'}")
        if is_captured(true_state):
            print(f"\nCaptured at t = {(k + 1) * dt:.0f} s with {dropouts} dropped frames")
            return
    print(f"\nNot captured within {horizon} s ({dropouts} dropped frames); "
          f"final |pos| = {np.linalg.norm(true_state[:3]):.2f} m, "
          f"|vel| = {np.linalg.norm(true_state[3:]):.3f} m/s")


if __name__ == "__main__":
    main()
