import numpy as np
from typing import List, Tuple, Dict, Optional


def _limit_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm == 0.0:
        return vec
    return vec / norm * max_norm


class APFController:
    """
    简单的人工势场 (APF) 控制器，用于多智能体二维平面导航。

    用法示例：
        ctrl = APFController(k_g=1.0, k_o=0.5, k_a=0.6, d0=2.0, ds=0.8, u_max=1.0, dt=0.1)
        u, new_pos = ctrl.step(positions, obstacles, goal)

    方法:
        step(positions, obstacles, goal) -> (vels, new_positions)
    """

    def __init__(
        self,
        k_g: float = 1.0,
        k_o: float = 0.8,
        k_a: float = 0.6,
        d0: float = 2.0,
        ds: float = 0.8,
        u_max: float = 1.0,
        dt: float = 0.1,
    ):
        self.k_g = k_g
        self.k_o = k_o
        self.k_a = k_a
        self.d0 = d0
        self.ds = ds
        self.u_max = u_max
        self.dt = dt

    def step(
        self,
        positions: np.ndarray,
        obstacles: List[np.ndarray],
        goal: np.ndarray,
        return_positions: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        执行一步 APF 计算。

        positions: (N,2) array
        obstacles: list of (2,) arrays (每个障碍物的 xy 坐标)
        goal: (2,) array

        返回: (vels, new_positions)
          - vels: (N,2) 每个智能体的速度向量（已限幅）
          - new_positions: (N,2) 更新后的位置 (如果 return_positions=True)
        """
        pos = np.array(positions, dtype=float)
        N = pos.shape[0]
        vels = np.zeros_like(pos)

        # 逐个智能体计算合力
        for i in range(N):
            p_i = pos[i]

            # 目标吸引力
            F_goal = self.k_g * (goal - p_i)

            # 障碍物斥力
            F_obs = np.zeros(2)
            for obs in obstacles:
                obs = np.array(obs, dtype=float)
                diff = p_i - obs
                d = np.linalg.norm(diff)
                if d <= 0.0:
                    # 距离非常近，直接施加大斥力并跳过除0
                    F_obs += self.k_o * (1.0 / 1e-3 - 1.0 / self.d0) / (1e-3**2) * (diff / (1e-3))
                    continue
                if d < self.d0:
                    coeff = self.k_o * (1.0 / d - 1.0 / self.d0) / (d * d)
                    F_obs += coeff * (diff / d)

            # 其他智能体的避碰斥力
            F_agent = np.zeros(2)
            for j in range(N):
                if j == i:
                    continue
                diff = p_i - pos[j]
                d = np.linalg.norm(diff)
                if d <= 0.0:
                    F_agent += self.k_a * (1.0 / 1e-3 - 1.0 / self.ds) / (1e-3**2) * (diff / (1e-3))
                    continue
                if d < self.ds:
                    coeff = self.k_a * (1.0 / d - 1.0 / self.ds) / (d * d)
                    F_agent += coeff * (diff / d)

            u = F_goal + F_obs + F_agent
            u = _limit_norm(u, self.u_max)
            vels[i] = u

        new_pos = pos + vels * self.dt

        if return_positions:
            return vels, new_pos
        return vels, None


if __name__ == "__main__":
    # 简单示例：3 个 agent，若干障碍物，运行几步打印位置
    ctrl = APFController(k_g=1.2, k_o=1.0, k_a=0.8, d0=2.0, ds=0.8, u_max=0.6, dt=0.1)
    positions = np.array([[-6.0, 0.0], [-6.5, 0.5], [-5.5, -0.5]])
    obstacles = [np.array([0.0, 0.0]), np.array([-2.0, 1.0])]
    goal = np.array([8.0, 0.0])

    for t in range(20):
        v, positions = ctrl.step(positions, obstacles, goal)
        print(f"step {t}: pos=\n{positions}, vel=\n{v}")
