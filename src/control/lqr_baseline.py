import numpy as np
import scipy.linalg

class LQRBaselineController:
    def __init__(self, sim):
        self.sim = sim
        n = self.sim.n
        
        # State space matrices for CW equations
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3*n**2, 0, 0, 0, 2*n, 0],
            [0, 0, 0, -2*n, 0, 0],
            [0, 0, -n**2, 0, 0, 0]
        ])
        
        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1/self.sim.mass, 0, 0],
            [0, 1/self.sim.mass, 0],
            [0, 0, 1/self.sim.mass]
        ])
        
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([1.0, 1.0, 1.0]) * 1e4
        
        # solve algebraic riccati eq for K gain
        P = scipy.linalg.solve_continuous_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R) @ self.B.T @ P

    def compute_control(self, state, target=np.zeros(6)):
        # F = -Kx
        return -self.K @ (state - target)
