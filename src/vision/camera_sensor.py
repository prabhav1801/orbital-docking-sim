"""Noisy full-state camera model and a Kalman filter using the CW dynamics."""
import numpy as np
import scipy.linalg


class OnboardCameraSensor:
    """Gaussian noise on position/velocity; with probability dropout_p the frame is lost
    and the previous measurement is returned with is_dropout=True."""

    def __init__(self, noise_pos=0.15, noise_vel=0.02, dropout_p=0.05, rng=None):
        self.sigma = np.array([noise_pos] * 3 + [noise_vel] * 3)
        self.dropout_p = dropout_p
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_valid = np.zeros(6)

    def get_measurement(self, true_state):
        if self.rng.random() < self.dropout_p:
            return self.last_valid, True
        measured = true_state + self.rng.normal(0.0, self.sigma)
        self.last_valid = measured
        return measured, False


class CWKalmanFilter:
    """Kalman filter whose process model is the zero-order-hold discretisation of the
    CW equations. The measurement is the full state (H = I)."""

    def __init__(self, A, B, dt, x0, P0=1.0, q=1e-3, r_pos=0.15, r_vel=0.02):
        n, m = A.shape[0], B.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        E = scipy.linalg.expm(M * dt)
        self.Ad, self.Bd = E[:n, :n], E[:n, n:]
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.eye(n) * P0
        self.Q = np.eye(n) * q
        self.R = np.diag([r_pos**2] * 3 + [r_vel**2] * 3)

    def predict(self, applied_force):
        self.x = self.Ad @ self.x + self.Bd @ applied_force
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

    def update(self, measurement, is_dropout):
        if not is_dropout:
            S = self.P + self.R
            K = self.P @ np.linalg.inv(S)
            self.x = self.x + K @ (measurement - self.x)
            self.P = (np.eye(len(self.x)) - K) @ self.P
        return self.x
