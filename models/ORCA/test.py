# 文件名: test_model.py
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p  # 必须导入 pybullet 才能获取位置
from ORCA_env import FormationPyBulletEnv
from draw.trajectory_formation_plot import plot_trajectory_and_formation
from draw.distance_statistics_plot import plot_distance_statistics

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载纯净版 ORCA 环境 (无 RL)
print("正在初始化 ORCA 基准测试环境...")
env = FormationPyBulletEnv(render_mode=True, enable_obstacles=True)

# 2. 跑一局测试
res = env.reset()
done = False

print("正在运行 ORCA 仿真... (请看弹出的 PyBullet 窗口)")

# 用于画图的数据初始化
num_robots = int(getattr(env, 'num_robots', len(getattr(env, 'robot_ids', [])) or 0))
if num_robots <= 0:
    num_robots = 4  # 默认 4 个智能体

# 轨迹字典: key=机器人索引, value=(x序列, y序列)
all_trajectories = {i: ([], []) for i in range(num_robots)}

while not done:
    # --- 核心修改：不再需要 model.predict() ---
    # 直接传入 action=None，环境内部的 rvo2 会自动计算每个小车的速度
    res = env.step(action=None)
    
    # 处理不同 gym 版本的 step() 返回值
    if isinstance(res, tuple) and len(res) == 5:
        obs, reward, terminated, truncated, info = res
        try:
            done = bool(terminated or truncated)
        except Exception:
            done = np.logical_or(terminated, truncated)
    else:
        obs, reward, done, info = res
        
    # PyBullet 环境没有 .pos 属性，必须通过 ID 查询物理引擎
    for i, robot_id in enumerate(env.robot_ids):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        all_trajectories[i][0].append(pos[0])
        all_trajectories[i][1].append(pos[1])
        
    # 如果到了目标或者超时，就停止
    if done:
        print(f"\n✅ 回合结束！")
        print(f"最终结果: {info.get('reason', 'unknown')}")
        print(f"总耗时步数: {env.current_step}")
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