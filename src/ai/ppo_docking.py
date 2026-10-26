import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.dynamics.cw_propagation import ClohessyWiltshireDynamics

class DockingEnv:
    def __init__(self):
        self.sim = ClohessyWiltshireDynamics(orbital_altitude_km=400.0, spacecraft_mass_kg=500.0, max_thrust_n=5.0)
        self.dt = 1.0
        self.state = None
        self.max_steps = 600
        self.current_step = 0

    def reset(self, random_offset=True):
        self.current_step = 0
        if random_offset:
            y_init = -100.0 + np.random.uniform(-5.0, 5.0)
            x_init = np.random.uniform(-2.0, 2.0)
            z_init = np.random.uniform(-2.0, 2.0)
            vx_init = np.random.uniform(-0.05, 0.05)
            vy_init = np.random.uniform(-0.05, 0.05)
            vz_init = np.random.uniform(-0.05, 0.05)
            self.state = np.array([x_init, y_init, z_init, vx_init, vy_init, vz_init])
        else:
            self.state = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
        return self.state

    def step(self, action):
        force = action * self.sim.max_thrust
        next_state, applied_force = self.sim.propagate_step(self.state, force, self.dt)
        
        pos_error = np.linalg.norm(next_state[0:3])
        vel_error = np.linalg.norm(next_state[3:6])
        
        reward = -0.5 * pos_error - 2.0 * vel_error - 0.01 * np.linalg.norm(applied_force)
        
        self.state = next_state
        self.current_step += 1
        
        done = False
        if pos_error < 0.5 and vel_error < 0.05:
            reward += 500.0
            done = True
        elif self.current_step >= self.max_steps or pos_error > 150.0:
            reward -= 100.0
            done = True
            
        return self.state, reward, done

class ActorCritic(nn.Module):
    def __init__(self, state_dim=6, action_dim=3):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

def train_ppo_agent():
    print("Boosting PPO Training for robust convergence...")
    device = torch.device("cpu")
    env = DockingEnv()
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    episodes = 400
    for ep in range(episodes):
        state = env.reset(random_offset=True)
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_mean, value = model(state_tensor)
            
            action = action_mean.detach().numpy()[0] + np.random.normal(0, 0.15, size=3)
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, done = env.step(action)
            
            # Simple policy gradient update step
            advantage = reward - value.item()
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            pred_action, pred_val = model(state_tensor)
            
            loss = nn.MSELoss()(pred_val, torch.FloatTensor([[reward]])) + torch.mean((pred_action - action_tensor)**2) * (-advantage)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            
        if (ep + 1) % 100 == 0:
            print(f"Training Progress: Episode {ep+1}/{episodes} completed.")
            
    torch.save(model.state_dict(), "ppo_docking_model.pth")
    print("PPO Agent re-trained and saved successfully!")

if __name__ == "__main__":
    train_ppo_agent()
