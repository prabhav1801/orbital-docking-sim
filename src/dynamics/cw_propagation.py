"""Linear Clohessy-Wiltshire (Hill) relative dynamics in the LVLH frame.

State: [x, y, z, vx, vy, vz] with x radial, y along-track, z cross-track.
Control input: thrust force in N, clipped per axis.
"""
import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6_371_000.0      # m


class ClohessyWiltshireDynamics:
    def __init__(self, alt=400.0, mass=500.0, max_thrust=5.0):
        """alt in km, mass in kg, max_thrust in N (per axis)."""
        r = R_EARTH + alt * 1000.0
        self.n = np.sqrt(MU_EARTH / r**3)  # mean motion, rad/s
        self.mass = mass
        self.max_thrust = max_thrust

    def linear_matrices(self):
        """Continuous-time (A, B) for xdot = A x + B u, u in N."""
        n, m = self.n, self.mass
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n**2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -n**2, 0, 0, 0],
        ])
        B = np.vstack([np.zeros((3, 3)), np.eye(3) / m])
        return A, B

    def apply_thrust_saturation(self, force):
        return np.clip(force, -self.max_thrust, self.max_thrust)

    def state_derivative(self, state, force=None):
        if force is None:
            force = np.zeros(3)
        x, y, z, vx, vy, vz = state
        clamped = self.apply_thrust_saturation(force)
        acc = clamped / self.mass
        dvx = 3 * self.n**2 * x + 2 * self.n * vy + acc[0]
        dvy = -2 * self.n * vx + acc[1]
        dvz = -self.n**2 * z + acc[2]
        return np.array([vx, vy, vz, dvx, dvy, dvz]), clamped

    def propagate_step(self, state, force, dt):
        """One RK4 step with the thrust held constant. Returns (next_state, applied_force)."""
        def f(s):
            return self.state_derivative(s, force)[0]

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_state, self.apply_thrust_saturation(force)
