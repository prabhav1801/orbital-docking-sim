import numpy as np
import scipy.linalg

class LQRBaselineController:
    def __init__(self, sim, target=np.zeros(6)):
        self.sim = sim
        self.target = target
        n = self.sim.n
        
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3*n**2, 0, 0, 0, 2*n, 0],
            [0, 0, 0, -2*n, 0, 0],
            [0, 0, -n**2, 0, 0, 0]
        ])
        
        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1/self.sim.mass, 0, 0],
            [0, 1/self.sim.mass, 0],
            [0, 0, 1/self.sim.mass]
        ])
        
        Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        R = np.diag([1.0, 1.0, 1.0]) * 1e4
        
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P

    def compute_control(self, state, target=None):
        tgt = target if target is not None else self.target
        return -self.K @ (state - tgt)
