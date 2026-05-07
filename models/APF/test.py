# 文件名: test_model.py
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p  # 必须导入 pybullet 才能获取位置
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from APF_env import FormationPyBulletEnv
from project_paths import MODEL_FINAL_NAME, model_load_path, MODEL_STAGE1_NAME
from draw.trajectory_formation_plot import plot_trajectory_and_formation
from draw.distance_statistics_plot import plot_distance_statistics

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


import numpy as np
import pybullet as p

env = FormationPyBulletEnv(render_mode=True, enable_obstacles=True)

use_apf = True
print("✅ 当前使用 APF baseline 进行测试，不加载 RL 模型")

# 2. 跑一局测试
res = env.reset()
obs = res[0] if isinstance(res, tuple) else res
done = False

print("正在运行 APF 仿真... (请看弹出的 PyBullet 窗口)")

# 用于画图的数据初始化
num_robots = int(getattr(env, 'num_robots', len(getattr(env, 'robot_ids', [])) or 0))
if num_robots <= 0:
    num_robots = 4

# 轨迹字典: key=机器人索引, value=(x序列, y序列)
all_trajectories = {i: ([], []) for i in range(num_robots)}

while not done:
    # =====================================================
    # APF baseline：不使用 RL 动作
    # env.step(None) 内部会调用 APF 控制 leader
    # =====================================================
    res = env.step(None)

    # 处理不同 gym 版本的 step() 返回值：
    # - Gym: obs, reward, done, info
    # - Gymnasium: obs, reward, terminated, truncated, info
    if isinstance(res, tuple) and len(res) == 5:
        obs, reward, terminated, truncated, info = res
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = res

    # 记录所有机器人轨迹
    for i, robot_id in enumerate(env.robot_ids):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        all_trajectories[i][0].append(pos[0])
        all_trajectories[i][1].append(pos[1])

    if done:
        print(f"回合结束。结果: {info.get('reason', 'unknown')}")
        if 'ep_formation_error_avg' in info:
            print(f"平均编队误差：{info['ep_formation_error_avg']:.3f}")
        if 'ep_formation_error_max' in info:
            print(f"最大编队误差：{info['ep_formation_error_max']:.3f}")

        break

# 在断开 PyBullet 连接前，缓存障碍物与围墙的几何信息
obstacle_footprints = []
if hasattr(env, 'obstacles'):
    for obs_id in env.obstacles:
        try:
            aabb_min, aabb_max = p.getAABB(obs_id)
            obstacle_footprints.append((aabb_min, aabb_max))
        except Exception:
            pass

wall_footprints = []
if hasattr(env, 'wall_ids'):
    for wall_id in env.wall_ids:
        try:
            aabb_min, aabb_max = p.getAABB(wall_id)
            wall_footprints.append((aabb_min, aabb_max))
        except Exception:
            pass

# 关闭物理引擎
env.close()

# 3. 独立绘制轨迹与编队形状图
print("正在绘制轨迹图...")
plot_trajectory_and_formation(
    env=env,
    all_trajectories=all_trajectories,
    obstacle_footprints=obstacle_footprints,
    wall_footprints=wall_footprints,
    output_path=None,
    show=True,
    # title="Trajectory",
)

# 4. 独立绘制距离统计图
# print("正在绘制距离统计图...")
# plot_distance_statistics(
#     all_trajectories=all_trajectories,
#     obstacle_footprints=obstacle_footprints,
#     topology_edges=getattr(env, 'edges', None),
#     show=True,
#     output_path="./outputs/figures/distance_statistics2.png",
#     # title="Formation Distance Statistics",
# )