# 文件名: test_model.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from formation_env import FormationGymEnv

# 1. 加载环境和模型
env = FormationGymEnv()
model = PPO.load("ppo_leader_model")

# 2. 跑一局测试
obs = env.reset()
done = False
path_x, path_y = [], []

print("正在渲染模型表现...")

# 用于画图的数据
all_trajectories = {0:([],[]), 1:([],[]), 2:([],[])}

while not done:
    # 让模型预测动作 (deterministic=True 表示测试时不随机，要确定性策略)
    action, _states = model.predict(obs, deterministic=True)
    
    # 环境执行一步
    obs, reward, done, info = env.step(action)
    
    # 记录轨迹
    for i, robot in enumerate(env.robots):
        all_trajectories[i][0].append(robot.pos[0])
        all_trajectories[i][1].append(robot.pos[1])
        
    # 如果到了目标或者超时，就停止
    if done:
        break

# 3. 画出轨迹图
plt.figure(figsize=(8, 8))

# 画目标点
plt.scatter(env.target_pos[0], env.target_pos[1], c='red', marker='*', s=200, label='Target')

# 画机器人轨迹
colors = ['r', 'b', 'g']
labels = ['Leader (RL)', 'Follower 1 (PID)', 'Follower 2 (PID)']

for i in range(3):
    plt.plot(all_trajectories[i][0], all_trajectories[i][1], color=colors[i], label=labels[i])
    # 画最终位置
    plt.scatter(all_trajectories[i][0][-1], all_trajectories[i][1][-1], c=colors[i], s=100)

plt.legend()
plt.title("RL Leader + PID Followers Result")
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()