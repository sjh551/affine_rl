from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch as th
from torch import nn


class SharedIPPOActorCritic(nn.Module):
    def __init__(
        self,
        kinematics_dim: int,
        lidar_channels: int,
        lidar_length: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        lidar_feature_dim: int = 64,
    ):
        super().__init__()
        self.kinematics_dim = int(kinematics_dim)
        self.lidar_channels = int(lidar_channels)
        self.lidar_length = int(lidar_length)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.lidar_feature_dim = int(lidar_feature_dim)

        self.kinematics_encoder = nn.Sequential(
            nn.Linear(self.kinematics_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        lidar_encoder, lidar_out_dim = self._build_lidar_encoder()
        self.lidar_encoder = lidar_encoder

        fused_dim = hidden_dim + lidar_out_dim
        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, self.action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(th.zeros(self.action_dim))

    def _build_lidar_encoder(self) -> Tuple[nn.Module, int]:
        cnn = nn.Sequential(
            nn.Conv1d(self.lidar_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            dummy = th.zeros(1, self.lidar_channels, self.lidar_length)
            flat_dim = int(cnn(dummy).shape[1])

        encoder = nn.Sequential(
            cnn,
            nn.Linear(flat_dim, self.lidar_feature_dim),
            nn.ReLU(),
        )
        return encoder, self.lidar_feature_dim

    def _extract_features(self, kinematics: th.Tensor, lidar: th.Tensor) -> th.Tensor:
        kin_feat = self.kinematics_encoder(kinematics)
        lidar_feat = self.lidar_encoder(lidar)
        return self.shared_trunk(th.cat([kin_feat, lidar_feat], dim=-1))

    def forward(self, kinematics: th.Tensor, lidar: th.Tensor):
        features = self._extract_features(kinematics, lidar)
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        std = th.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return mean, std, value

    @staticmethod
    def _atanh(x: th.Tensor) -> th.Tensor:
        x = th.clamp(x, -0.999999, 0.999999)
        return 0.5 * (th.log1p(x) - th.log1p(-x))

    def _log_prob_from_pre_tanh(self, normal_dist: th.distributions.Normal, pre_tanh: th.Tensor, action: th.Tensor) -> th.Tensor:
        log_prob = normal_dist.log_prob(pre_tanh).sum(dim=-1)
        correction = th.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return log_prob - correction

    def act(self, obs_batch: Dict[str, th.Tensor], deterministic: bool = False):
        mean, std, value = self.forward(obs_batch["kinematics"], obs_batch["lidar"])
        dist = th.distributions.Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action = th.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, action)
        return action, log_prob, value

    def evaluate_actions(self, obs_batch: Dict[str, th.Tensor], actions: th.Tensor):
        mean, std, value = self.forward(obs_batch["kinematics"], obs_batch["lidar"])
        dist = th.distributions.Normal(mean, std)
        pre_tanh = self._atanh(actions)
        log_prob = self._log_prob_from_pre_tanh(dist, pre_tanh, actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def value(self, obs_batch: Dict[str, th.Tensor]):
        _, _, value = self.forward(obs_batch["kinematics"], obs_batch["lidar"])
        return value

    def save(self, file_path: str | Path, metadata: Dict | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "kinematics_dim": self.kinematics_dim,
                "lidar_channels": self.lidar_channels,
                "lidar_length": self.lidar_length,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "lidar_feature_dim": self.lidar_feature_dim,
            },
            "metadata": metadata or {},
        }
        th.save(payload, str(file_path))

    @classmethod
    def load(cls, file_path: str | Path, map_location: str | th.device = "cpu"):
        payload = th.load(str(file_path), map_location=map_location)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        model.eval()
        return model, payload


def build_obs_batch(agent_obs: Dict[str, np.ndarray], device: str | th.device):
    kinematics = th.as_tensor(agent_obs["kinematics"], dtype=th.float32, device=device).unsqueeze(0)
    lidar = th.as_tensor(agent_obs["lidar"], dtype=th.float32, device=device).unsqueeze(0)
    return {"kinematics": kinematics, "lidar": lidar}


def build_multi_agent_obs_batch(observations: Dict[str, Dict[str, np.ndarray]], agent_names: Iterable[str], device: str | th.device):
    kin_list = []
    lidar_list = []
    for name in agent_names:
        agent_obs = observations[name]
        kin_list.append(np.asarray(agent_obs["kinematics"], dtype=np.float32))
        lidar_list.append(np.asarray(agent_obs["lidar"], dtype=np.float32))

    return {
        "kinematics": th.as_tensor(np.stack(kin_list, axis=0), dtype=th.float32, device=device),
        "lidar": th.as_tensor(np.stack(lidar_list, axis=0), dtype=th.float32, device=device),
    }


def actions_to_dict(actions: th.Tensor, agent_names: Iterable[str]):
    action_np = actions.detach().cpu().numpy()
    return {name: action_np[i].astype(np.float32) for i, name in enumerate(agent_names)}