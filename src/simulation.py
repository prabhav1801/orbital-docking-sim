"""Closed-loop episode runner shared by the benchmark, dashboard and scripts."""
import numpy as np

POS_TOL = 0.5   # m
VEL_TOL = 0.05  # m/s


def is_captured(state):
    return np.linalg.norm(state[:3]) < POS_TOL and np.linalg.norm(state[3:]) < VEL_TOL


def run_episode(sim, controller, state0, horizon=600, dt=1.0, record=False):
    """Run until capture or `horizon` steps. Perfect state feedback.

    Returns a dict with success, capture_time (s, None if not captured), delta_v (m/s),
    final_state, and trajectory (rows of [t, *state]) if record=True.
    """
    state = np.asarray(state0, dtype=float).copy()
    delta_v = 0.0
    traj = []
    for k in range(horizon):
        if record:
            traj.append(np.concatenate([[k * dt], state]))
        force = controller.compute_control(state)
        state, applied = sim.propagate_step(state, force, dt)
        delta_v += np.linalg.norm(applied) / sim.mass * dt
        if is_captured(state):
            return dict(success=True, capture_time=(k + 1) * dt, delta_v=delta_v,
                        final_state=state, trajectory=np.array(traj))
    return dict(success=False, capture_time=None, delta_v=delta_v,
                final_state=state, trajectory=np.array(traj))
