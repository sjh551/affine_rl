# 文件名: test_model.py
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p  # 必须导入 pybullet 才能获取位置
import torch
from stable_baselines3 import PPO
from pybullet_env import FormationPyBulletEnv 
from leader_env import SingleRobotPyBulletEnv

# 1. 加载环境和模型
# 注意：这里开启 render_mode=True，你会看到一个 PyBullet 的 3D 窗口弹出
env = FormationPyBulletEnv(render_mode=True, enable_obstacles=True)
# 尝试加载模型，如果没有模型文件，就随机跑（方便调试代码逻辑）
try:
    # model = PPO.load("ppo2_stage1_nav", env=env)
    model = PPO.load("ppo2_final_leader", env=env)
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

# 2. 跑一局测试
obs = env.reset()
done = False

print("正在运行仿真... (请看弹出的 PyBullet 窗口)")

# 用于画图的数据初始化
all_trajectories = {0:([],[]), 1:([],[]), 2:([],[])}

while not done:
    if use_rl:
        # 让模型预测动作 (deterministic=True 表示使用确定性策略，不随机探索)
        action, _states = model.predict(obs, deterministic=True)
    else:
        # 只要能跑起来就行，动作随便给
        action = env.action_space.sample() 
    
    # 环境执行一步
    obs, reward, done, info = env.step(action)
    
    # --- [修正点] 获取 PyBullet 中的坐标 ---
    # PyBullet 环境没有 .pos 属性，必须通过 ID 查询物理引擎
    for i, robot_id in enumerate(env.robot_ids):
        # p.getBasePositionAndOrientation 返回 ((x,y,z), (qx,qy,qz,qw))
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        all_trajectories[i][0].append(pos[0])
        all_trajectories[i][1].append(pos[1])
        
    # 如果到了目标或者超时，就停止
    if done:
        print(f"回合结束。结果: {info.get('result', 'unknown')}")
        break

# 关闭物理引擎
env.close()

# 3. 画出轨迹图 (Matplotlib 后处理)
print("正在绘制轨迹图...")
plt.figure(figsize=(10, 6)) # 调整为长方形比例，适应你的地图

# 画目标点
plt.scatter(env.target_pos[0], env.target_pos[1], c='red', marker='*', s=300, label='Target', zorder=5)

# 画机器人轨迹
colors = ['r', 'b', 'g']
labels = ['Leader (RL)', 'Follower 1 (PID)', 'Follower 2 (PID)']

for i in range(3):
    # 画线
    plt.plot(all_trajectories[i][0], all_trajectories[i][1], color=colors[i], label=labels[i], linewidth=2)
    # 画起点 (空心圈)
    plt.scatter(all_trajectories[i][0][0], all_trajectories[i][1][0], c='white', edgecolors=colors[i], s=50)
    # 画终点 (实心圈)
    plt.scatter(all_trajectories[i][0][-1], all_trajectories[i][1][-1], c=colors[i], s=100)

# 画出障碍物 (方便分析避障效果)
# 注意：这里我们只能画出大概位置，因为PyBullet获取障碍物形状比较麻烦
# 假设我们知道环境里的障碍物列表 env.obstacles
if hasattr(env, 'obstacles'):
    for obs_id in env.obstacles:
        pos, _ = p.getBasePositionAndOrientation(obs_id)
        # 画一个黑色圆圈代表障碍物
        circle = plt.Circle((pos[0], pos[1]), 0.3, color='black', alpha=0.5)
        plt.gca().add_patch(circle)

# 画出围墙范围
wall_len = getattr(env, 'wall_length', 10.0)
wall_wid = getattr(env, 'wall_width', 5.0)
plt.plot([-wall_len, wall_len, wall_len, -wall_len, -wall_len], 
         [wall_wid, wall_wid, -wall_wid, -wall_wid, wall_wid], 'k--', linewidth=2, label='Walls')

plt.legend()
plt.title("Trajectory")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.axis('equal') # 保持比例一致，否则圆会变成椭圆
plt.show()