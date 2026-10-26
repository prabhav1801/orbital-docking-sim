import numpy as np
from src.dynamics.cw_propagation import ClohessyWiltshireDynamics
from src.control.lqr_baseline import LQRBaselineController

def run_day1_simulation():
    print("Initializing Orbital Docking Simulation - Day 1 Baseline...")
    sim = ClohessyWiltshireDynamics(orbital_altitude_km=400.0, spacecraft_mass_kg=500.0, max_thrust_n=5.0)
    controller = LQRBaselineController(sim)
    
    state = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    dt = 1.0
    total_time = 600.0
    t = 0.0
    total_delta_v = 0.0
    
    print(f"{'Time(s)':<8} | {'X(m)':<8} | {'Y(m)':<8} | {'Z(m)':<8} | {'Status'}")
    print("-" * 55)
    
    while t <= total_time:
        if int(t) % 60 == 0:
            print(f"{t:<8.1f} | {state[0]:<8.2f} | {state[1]:<8.2f} | {state[2]:<8.2f} | Vel: {np.linalg.norm(state[3:6]):.3f} m/s")
        
        raw_force = controller.compute_control(state, target)
        state, applied_force = sim.propagate_step(state, raw_force, dt)
        
        accel_mag = np.linalg.norm(applied_force) / sim.mass
        total_delta_v += accel_mag * dt
        t += dt
        
        if np.linalg.norm(state[0:3]) < 0.5 and np.linalg.norm(state[3:6]) < 0.05:
            print(f"\n[SUCCESS] Docking achieved at t = {t:.1f}s!")
            break
            
    print(f"\nSimulation Complete. Total Delta-V Used: {total_delta_v:.2f} m/s")

if __name__ == "__main__":
    run_day1_simulation()
