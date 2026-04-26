import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from POMDP_env import FormationPyBulletEnv


class LeaderFollowerMAPPOEnv(gym.Env):
    """MAPPO wrapper for the existing FormationPyBulletEnv.

    It exposes:
    - per-agent local observations for two leaders
    - a centralized global state for critic
    - per-agent 2D actions mapped to original 4D action
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: bool = False, enable_obstacles: bool = True):
        super().__init__()
        self.base_env = FormationPyBulletEnv(
            render_mode=render_mode,
            enable_obstacles=enable_obstacles,
        )

        self.n_agents = 2
        self.agent_action_dim = 2

        kin_dim = self.base_env.observation_space["kinematics"].shape[0]
        lidar_dim = int(np.prod(self.base_env.observation_space["lidar_A"].shape))

        self.agent_obs_dim = kin_dim + lidar_dim
        self.global_state_dim = kin_dim + 2 * lidar_dim

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, self.agent_action_dim),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self.agent_obs_dim),
            dtype=np.float32,
        )
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.global_state_dim,),
            dtype=np.float32,
        )

    def _encode_obs(self, raw_obs: dict):
        kin = raw_obs["kinematics"].astype(np.float32)
        lidar_a = raw_obs["lidar_A"].astype(np.float32).reshape(-1)
        lidar_b = raw_obs["lidar_B"].astype(np.float32).reshape(-1)

        obs_a = np.concatenate([kin, lidar_a], axis=0)
        obs_b = np.concatenate([kin, lidar_b], axis=0)
        obs_agents = np.stack([obs_a, obs_b], axis=0)

        global_state = np.concatenate([kin, lidar_a, lidar_b], axis=0).astype(np.float32)
        return obs_agents.astype(np.float32), global_state

    def reset(self, *, seed=None, options=None):
        raw_obs, info = self.base_env.reset(seed=seed, options=options)
        obs_agents, global_state = self._encode_obs(raw_obs)
        info = dict(info)
        info["state"] = global_state
        return obs_agents, info

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n_agents, self.agent_action_dim):
            raise ValueError(
                f"Expected action shape {(self.n_agents, self.agent_action_dim)}, got {actions.shape}"
            )

        joint_action = np.concatenate([actions[0], actions[1]], axis=0)
        raw_obs, reward, terminated, truncated, info = self.base_env.step(joint_action)

        obs_agents, global_state = self._encode_obs(raw_obs)
        done = bool(terminated or truncated)

        info = dict(info)
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["state"] = global_state

        return obs_agents, float(reward), done, info

    def close(self):
        if hasattr(self.base_env, "physicsClient"):
            try:
                p.disconnect(self.base_env.physicsClient)
            except Exception:
                pass
