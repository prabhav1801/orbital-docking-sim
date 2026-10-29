"""Actor-critic docking environment and a simple policy-gradient style trainer.

Prototype. The update below is a one-step advantage-weighted regression, not PPO
(no clipped surrogate, no rollout buffer, no GAE). It has not been validated against
the LQR baseline.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dynamics.cw_propagation import ClohessyWiltshireDynamics
from src.simulation import is_captured


class DockingEnv:
    def __init__(self, max_steps=600, dt=1.0, seed=None):
        self.sim = ClohessyWiltshireDynamics(alt=400.0, mass=500.0, max_thrust=5.0)
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.step_count = 0

    def reset(self, random_offset=True):
        self.step_count = 0
        if random_offset:
            self.state = np.array([
                self.rng.uniform(-2.0, 2.0),
                -100.0 + self.rng.uniform(-5.0, 5.0),
                self.rng.uniform(-2.0, 2.0),
                *self.rng.uniform(-0.05, 0.05, size=3),
            ])
        else:
            self.state = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
        return self.state

    def step(self, action):
        """action in [-1, 1]^3, scaled to the thrust limit."""
        force = action * self.sim.max_thrust
        self.state, applied = self.sim.propagate_step(self.state, force, self.dt)
        self.step_count += 1

        pos_err = np.linalg.norm(self.state[:3])
        vel_err = np.linalg.norm(self.state[3:])
        reward = -0.5 * pos_err - 2.0 * vel_err - 0.01 * np.linalg.norm(applied)

        done = False
        if is_captured(self.state):
            reward += 500.0
            done = True
        elif self.step_count >= self.max_steps or pos_err > 150.0:
            reward -= 100.0
            done = True
        return self.state, reward, done


class ActorCritic(nn.Module):
    def __init__(self, state_dim=6, action_dim=3, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)


def train(episodes=400, lr=5e-4, action_noise=0.15, save_path="docking_policy.pth", seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = DockingEnv(seed=seed)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_mean, value = model(s)
            action = np.clip(action_mean.detach().numpy()[0] + rng.normal(0, action_noise, size=3), -1.0, 1.0)

            next_state, reward, done = env.step(action)

            advantage = reward - value.item()
            a = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            value_loss = nn.functional.mse_loss(value, torch.tensor([[reward]], dtype=torch.float32))
            policy_loss = -advantage * torch.mean((action_mean - a) ** 2)
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

        if (ep + 1) % 100 == 0:
            print(f"episode {ep + 1}/{episodes}")

    torch.save(model.state_dict(), save_path)
    print(f"saved {save_path}")


if __name__ == "__main__":
    train()
