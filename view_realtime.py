# 文件名: view_realtime.py
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from formation_env import FormationGymEnv
from config import *

# --- 1. 加载环境和模型 ---
env = FormationGymEnv()

# ⚠️ 如果你还没跑 train.py，没有模型文件，把下面这行改为 USE_RL = False
# 这样你可以先看 PID 的实时效果
USE_RL = True 

if USE_RL:
    try:
        model = PPO.load("ppo_leader_model")
        print("已加载 RL 模型，Leader 由 AI 控制。")
    except:
        print("未找到模型文件！请先运行 train.py，或者设置 USE_RL = False")
        exit()
else:
    print("使用 PID 模式演示 (无 AI)。")

# --- 2. 准备绘图窗口 ---
plt.ion()  # 开启交互模式 (Interactive Mode) - 这就是实时的关键
fig, ax = plt.subplots(figsize=(8, 8))

# 跑 5 轮测试
for episode in range(10):
    obs = env.reset()
    done = False
    step = 0
    
    print(f"--- 第 {episode+1} 轮测试开始 ---")
    
    while not done:
        # A. 决策
        if USE_RL:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # PID 逻辑 (如果还没训练，用这个顶替)
            # 简单的向目标移动
            leader = env.robots[0]
            direction = env.target_pos - leader.pos
            action = np.clip(direction, -1, 1) # 简单的动作

        # B. 环境执行
        obs, reward, done, info = env.step(action)
        step += 1
        
        # C. 实时绘图 (每 2 帧画一次，太快了看不清)
        if step % 2 == 0:
            ax.clear()
            
            # 画目标 (红五星)
            ax.scatter(env.target_pos[0], env.target_pos[1], c='red', marker='*', s=300, label='Target')
            
            
            # 画机器人
            colors = ['red', 'blue', 'green']
            labels = ['Leader', 'F1', 'F2']
            
            for i, robot in enumerate(env.robots):
                # 画身体
                circle = plt.Circle(robot.pos, ROBOT_RADIUS, color=colors[i], fill=True, label=labels[i])
                ax.add_patch(circle)
                # 画轨迹 (最近50个点，避免太乱)
                traj_x = robot.traj_x[-50:]
                traj_y = robot.traj_y[-50:]
                ax.plot(traj_x, traj_y, color=colors[i], linestyle='--', alpha=0.5)

            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.grid(True)
            ax.set_title(f"Episode {episode+1} | Step {step} | Reward {reward:.2f}")
            ax.legend(loc='upper right')
            
            # 关键：暂停一小会儿，让窗口刷新
            plt.draw()
            plt.pause(0.01) 

    print(f"第 {episode+1} 轮结束")
    plt.pause(1.0) # 结束后停顿 1 秒

plt.ioff()
plt.show()