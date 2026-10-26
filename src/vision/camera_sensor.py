import numpy as np

class OnboardCameraSensor:
    def __init__(self, noise_pos=0.15, noise_vel=0.02, dropout_p=0.05):
        self.noise_pos = noise_pos
        self.noise_vel = noise_vel
        self.dropout_p = dropout_p
        self.last_valid = np.zeros(6)

    def get_measurement(self, true_state):
        # simulate vision dropout / occlusion occasionally
        if np.random.rand() < self.dropout_p:
            return self.last_valid, True
        
        noise = np.random.normal(0.0, [self.noise_pos]*3 + [self.noise_vel]*3, size=6)
        measured = true_state + noise
        self.last_valid = measured
        return measured, False

class SimpleKalmanFilter:
    def __init__(self):
        self.state_estimate = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * 1e-3
        self.R = np.eye(6) * 0.1

    def predict(self, A, B, state, force, dt):
        F = np.eye(6) + A * dt
        self.state_estimate = F @ self.state_estimate
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement, is_dropout):
        if is_dropout:
            return self.state_estimate # skip update if vision dropped
        
        H = np.eye(6)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = measurement - (H @ self.state_estimate)
        
        self.state_estimate += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.state_estimate
