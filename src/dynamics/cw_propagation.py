import numpy as np

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
