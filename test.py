# 文件名: test_model.py
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p  # 必须导入 pybullet 才能获取位置
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from POMDP_env import FormationPyBulletEnv
from project_paths import MODEL_FINAL_NAME, model_load_path, MODEL_STAGE1_NAME
from draw.trajectory_formation_plot import plot_trajectory_and_formation
from draw.distance_statistics_plot import plot_distance_statistics

plt.rcParams['font.family'] = [
    'Times New Roman',
    'Noto Sans CJK SC',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False


# 1. 加载环境和模型

# 尝试加载模型，如果没有模型文件，就随机
try:
    # model = PPO.load(model_load_path(MODEL_STAGE1_NAME))
    model = PPO.load(model_load_path(MODEL_FINAL_NAME))
    print("✅ 成功加载 RL 模型！")
    # 打印 CUDA 状态和模型所在设备
    try:
        print('torch.cuda.is_available():', torch.cuda.is_available())
    except Exception as e:
        print('无法检测 CUDA 可用性:', e)
    try:
        print('model.device:', model.device)
    except Exception as e:
        print('model.device 属性不可用:', e)
    # 统计模型参数量（policy 层）
    try:
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        print(f'Policy class: {type(model.policy).__name__}')
        print(f'Total params: {total_params:,}, Trainable params: {trainable_params:,}')
    except Exception as e:
        print('无法统计模型参数:', e)
    use_rl = True
except:
    print("⚠️ 未找到模型文件，将使用随机动作进行测试...")
    use_rl = False

env = FormationPyBulletEnv(render_mode=True, enable_obstacles=False)
# 2. 跑一局测试
# 兼容 Gym / Gymnasium：env.reset() 可能返回 obs 或 (obs, info)
res = env.reset()
obs = res[0] if isinstance(res, tuple) else res
done = False

print("正在运行仿真... (请看弹出的 PyBullet 窗口)")

# 用于画图的数据初始化
num_robots = int(getattr(env, 'num_robots', len(getattr(env, 'robot_ids', [])) or 0))
if num_robots <= 0:
    num_robots = 4

# 轨迹字典: key=机器人索引, value=(x序列, y序列)
all_trajectories = {i: ([], []) for i in range(num_robots)}


while not done:
    if use_rl:
        # 让模型预测动作 (deterministic=True 表示使用确定性策略，不随机探索)
        action, _ = model.predict(
            obs, 
            # state=lstm_states,      # <--- 传入记忆
            deterministic=True
        )
    else:
        # 只要能跑起来就行，动作随便给
        action = env.action_space.sample() 
    
    # 环境执行一步
    res = env.step(action)
    # 处理不同 gym 版本的 step() 返回值：
    # - Gym (older): obs, reward, done, info
    # - Gymnasium (new): obs, reward, terminated, truncated, info
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
        # p.getBasePositionAndOrientation 返回 ((x,y,z), (qx,qy,qz,qw))
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        all_trajectories[i][0].append(pos[0])
        all_trajectories[i][1].append(pos[1])
        
    # 如果到了目标或者超时，就停止
    if done:
        print(f"回合结束。结果: {info.get('reason', 'unknown')}")
        print(f"进度奖励：{env.ep_rew_progress:.2f}, 形状奖励：{env.ep_rew_shape:.2f}, 碰撞奖励：{env.ep_rew_collision:.2f}, 平滑奖励：{env.ep_rew_smooth:.2f}, 时间奖励：{env.ep_rew_time:.2f}, 总奖励：{env.ep_rew_progress + env.ep_rew_shape + env.ep_rew_collision + env.ep_rew_smooth + env.ep_rew_time:.2f} ")
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