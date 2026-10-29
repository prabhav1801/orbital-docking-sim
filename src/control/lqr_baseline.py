"""LQR controller from the continuous-time algebraic Riccati equation."""
import numpy as np
import scipy.linalg


class LQRBaselineController:
    def __init__(self, sim, q_diag=(100, 100, 100, 10, 10, 10), r_scale=1e4, gain_scale=1.0):
        """gain_scale multiplies K; 1.0 is the plain LQR gain."""
        self.sim = sim
        self.target = np.zeros(6)
        self.A, self.B = sim.linear_matrices()
        Q = np.diag(np.asarray(q_diag, dtype=float))
        R = np.eye(3) * r_scale
        P = scipy.linalg.solve_continuous_are(self.A, self.B, Q, R)
        self.K = gain_scale * (np.linalg.inv(R) @ self.B.T @ P)

    def compute_control(self, state, target=None):
        tgt = self.target if target is None else target
        return -self.K @ (state - tgt)
