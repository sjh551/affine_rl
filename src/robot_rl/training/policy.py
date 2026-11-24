"""PPO feature extraction for the ordered three-leader observation."""

import sys

import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FormationFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        n_leaders, n_frames, n_bins = observation_space["lidar"].shape
        n_features = observation_space["kinematics"].shape[0]
        super().__init__(observation_space, features_dim=n_features + 64 * n_leaders)

        convolution = nn.Sequential(
            nn.Conv1d(n_frames, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = convolution(th.zeros(1, n_frames, n_bins)).shape[1]
        self.lidar_encoder = nn.Sequential(convolution, nn.Linear(n_flatten, 64), nn.ReLU())

    def forward(self, observations):
        lidar = observations["lidar"]
        batch, _, n_frames, n_bins = lidar.shape
        # The leaders share a CNN; their encoded features keep their node order.
        lidar_features = self.lidar_encoder(lidar.reshape(-1, n_frames, n_bins)).reshape(batch, -1)
        return th.cat((observations["kinematics"], lidar_features), dim=1)


def policy_kwargs():
    return {
        "features_extractor_class": FormationFeaturesExtractor,
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
    }


def load_policy(path, **kwargs):
    # Existing SB3 archives refer to the former top-level extractor module.
    sys.modules.setdefault("training_common", sys.modules[__name__])
    return PPO.load(path, **kwargs)
