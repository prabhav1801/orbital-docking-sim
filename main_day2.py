import numpy as np
from src.dynamics.cw_propagation import ClohessyWiltshireDynamics
from src.control.lqr_baseline import LQRBaselineController
from src.vision.camera_sensor import OnboardCameraSensor, SimpleKalmanFilter

def run_day2_simulation():
    sim = ClohessyWiltshireDynamics(orbital_altitude_km=400.0, spacecraft_mass_kg=500.0, max_thrust_n=5.0)
    controller = LQRBaselineController(sim)
    
    # Initialize camera sensor (with noise and 5% dropout probability) and Kalman Filter
    camera = OnboardCameraSensor(noise_std_pos=0.15, noise_std_vel=0.02, dropout_probability=0.05)
    kf = SimpleKalmanFilter()
    
    # Initial true state: Starting 100m behind
    true_state = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    kf.state_estimate = true_state.copy()
    
    dt = 1.0
    total_time = 600.0
    t = 0.0
    dropouts_encountered = 0
    
    print(f"{'Time(s)':<8} | {'True Y(m)':<10} | {'Est Y(m)':<10} | {'Vision Status'}")
    print("-" * 50)
    
    while t <= total_time:
        # 1. Simulate onboard camera measurement with noise and potential dropout
        measured_state, is_dropout = camera.get_measurement(true_state)
        if is_dropout:
            dropouts_encountered += 1
            
        # 2. Kalman filter predict & update cycle
        kf.predict(controller.A, controller.B, kf.state_estimate, np.zeros(3), dt)
        estimated_state = kf.update(measured_state, is_dropout)
        
        # 3. Controller computes force based on filtered state estimate
        raw_force = controller.compute_control(estimated_state, target)
        
        # 4. Propagate true physics forward
        true_state, applied_force = sim.propagate_step(true_state, raw_force, dt)
        
        if int(t) % 60 == 0:
            status = "DROPOUT" if is_dropout else "Active"
            print(f"{t:<8.1f} | {true_state[1]:<10.2f} | {estimated_state[1]:<10.2f} | {status}")
            
        t += dt
        
        if np.linalg.norm(true_state[0:3]) < 0.5 and np.linalg.norm(true_state[3:6]) < 0.05:
            print(f"\n[SUCCESS] Docking achieved under noisy vision at t = {t:.1f}s!")
            print(f"Total transient vision dropouts handled: {dropouts_encountered}")
            break

if __name__ == "__main__":
    run_day2_simulation()
