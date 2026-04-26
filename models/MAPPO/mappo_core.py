from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    update_epochs: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_horizon: int = 1024


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        x = self.backbone(obs)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class RolloutBuffer:
    def __init__(self, horizon: int, n_agents: int, obs_dim: int, state_dim: int, action_dim: int):
        self.horizon = horizon
        self.n_agents = n_agents

        self.obs = np.zeros((horizon, n_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((horizon, state_dim), dtype=np.float32)
        self.actions = np.zeros((horizon, n_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((horizon, n_agents), dtype=np.float32)
        self.rewards = np.zeros(horizon, dtype=np.float32)
        self.dones = np.zeros(horizon, dtype=np.float32)
        self.values = np.zeros(horizon, dtype=np.float32)

        self.returns = np.zeros(horizon, dtype=np.float32)
        self.advantages = np.zeros(horizon, dtype=np.float32)

        self.ptr = 0

    def add(self, obs, state, action, log_prob, reward, done, value):
        t = self.ptr
        self.obs[t] = obs
        self.states[t] = state
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self.ptr += 1

    def compute_returns_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        gae = 0.0
        for t in reversed(range(self.horizon)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.horizon - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values


class MAPPOTrainer:
    def __init__(self, env, config: MAPPOConfig, device: str = "cpu"):
        self.env = env
        self.cfg = config
        self.device = torch.device(device)

        self.n_agents = env.n_agents
        self.obs_dim = env.agent_obs_dim
        self.state_dim = env.global_state_dim
        self.action_dim = env.agent_action_dim

        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    @torch.no_grad()
    def collect_rollout(self):
        buffer = RolloutBuffer(
            horizon=self.cfg.rollout_horizon,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

        obs, info = self.env.reset()
        state = info["state"]

        episode_reward = 0.0
        episode_count = 0

        for _ in range(self.cfg.rollout_horizon):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            dist = self.actor(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            env_action = torch.clamp(action, -1.0, 1.0).cpu().numpy().astype(np.float32)

            value = self.critic(state_t).item()

            next_obs, reward, done, step_info = self.env.step(env_action)
            next_state = step_info["state"]

            buffer.add(
                obs=obs,
                state=state,
                action=env_action,
                log_prob=log_prob.cpu().numpy().astype(np.float32),
                reward=reward,
                done=float(done),
                value=value,
            )

            episode_reward += reward
            obs = next_obs
            state = next_state

            if done:
                episode_count += 1
                obs, info = self.env.reset()
                state = info["state"]

        last_state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        last_value = self.critic(last_state_t).item()
        buffer.compute_returns_advantages(last_value, self.cfg.gamma, self.cfg.gae_lambda)

        stats = {
            "mean_rollout_reward": float(np.sum(buffer.rewards)),
            "episodes_finished": int(episode_count),
            "episode_reward_acc": float(episode_reward),
        }
        return buffer, stats

    def update(self, buffer: RolloutBuffer):
        obs = torch.tensor(buffer.obs, dtype=torch.float32, device=self.device)
        states = torch.tensor(buffer.states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(buffer.returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(buffer.advantages, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_agents = advantages.unsqueeze(1).expand(-1, self.n_agents)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.cfg.update_epochs):
            # Critic update
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_optim.step()

            # Actor update (parameter sharing across agents)
            flat_obs = obs.view(-1, self.obs_dim)
            flat_actions = actions.view(-1, self.action_dim)
            flat_old_log_probs = old_log_probs.view(-1)
            flat_adv = advantages_agents.reshape(-1)

            dist = self.actor(flat_obs)
            new_log_probs = dist.log_prob(flat_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - flat_old_log_probs)
            surr1 = ratio * flat_adv
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * flat_adv
            actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_optim.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "actor_loss": total_actor_loss / self.cfg.update_epochs,
            "critic_loss": total_critic_loss / self.cfg.update_epochs,
            "adv_mean": float(buffer.advantages.mean()),
            "ret_mean": float(buffer.returns.mean()),
        }

    def save(self, save_path: str):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "config": self.cfg.__dict__,
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(payload, save_path)
