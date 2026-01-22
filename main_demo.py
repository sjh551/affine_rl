# 文件名: main_demo.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from robot import Robot
from formation_env import FormationGymEnv
from config import *

# --- 初始化 ---
# 1个 Leader (ID:0), 2个 Follower (ID:1, 2)
# 初始位置排成一排
robots = [
    Robot(0, [0.0, 0.0]),
    Robot(1, [-1.0, 0.0]),
    Robot(2, [1.0, 0.0])
]

env = FormationGymEnv(robots)  

# 定义队形偏移量 (相对于Leader)
# Follower 1 在左后方，Follower 2 在右后方 -> 三角形编队
offsets = {
    0: np.array([0.0, 0.0]),    # Leader 也是目标
    1: np.array([-0.8, -0.8]),  # Follower 1
    2: np.array([0.8, -0.8])    # Follower 2
}

# --- 简单的PID控制器 (代替神经网络) ---
def pid_controller(current_pos, target_pos, current_vel):
    kp = 2.0  # 比例系数
    kd = 1.0  # 微分系数
    
    error = target_pos - current_pos
    # 期望动作 = 拉向目标 + 减速阻尼
    action = kp * error - kd * current_vel
    return action

# --- 动画更新函数 ---
def update_frame(frame):
    plt.cla() # 清除上一帧
    
    # 1. 设定 Leader 的运动轨迹 (画一个圆) 轨迹规划
    time = frame * DT
    leader_target_global = np.array([
        2.0 * np.cos(time * 0.5), 
        2.0 * np.sin(time * 0.5)
    ])
    
    # 2. 更新每个机器人的动作
    for robot in robots:
        # 计算该机器人应该在的“编队目标点”
        # 逻辑：Leader去全局目标，Follower去(Leader位置 + 偏移量)
        if robot.id == 0:
            target = leader_target_global
        else:
            # 这是一个简单的层级控制：Follower 盯着 Leader
            target = robots[0].pos + offsets[robot.id]
        
        # 计算控制力 (PID)
        action = pid_controller(robot.pos, target, robot.vel)
        
        # 物理更新
        robot.update(action)
        
        # --- 核心：计算并打印奖励 (验证论文公式) ---
        reward, r_g, r_c = env.compute_reward(robot.id, target)
        if frame % 10 == 0: # 每10帧打印一次，防止刷屏
            print(f"Frame {frame} | Agent {robot.id}: Reward={reward:.2f} (Goal={r_g:.2f}, Coll={r_c:.2f})")

    # 3. 绘图
    colors = ['red', 'blue', 'green']
    labels = ['Leader', 'Follower 1', 'Follower 2']
    
    for robot, color, label in zip(robots, colors, labels):
        # 画轨迹
        plt.plot(robot.traj_x, robot.traj_y, color=color, alpha=0.3)
        # 画机器人 (圆)
        circle = plt.Circle(robot.pos, ROBOT_RADIUS, color=color, fill=True, label=label)
        plt.gca().add_patch(circle)
        # 画目标连线
        if robot.id != 0:
            target = robots[0].pos + offsets[robot.id]
            plt.plot([robot.pos[0], target[0]], [robot.pos[1], target[1]], 'k--', alpha=0.2)

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.title(f"Time: {time:.1f}s - Formation Control Simulation")
    plt.legend()
    plt.axis('equal')

# --- 运行 ---
print("开始仿真... (按窗口关闭键结束)")
fig = plt.figure(figsize=(8, 8))
ani = FuncAnimation(fig, update_frame, frames=np.arange(0, 500), interval=50)
plt.show()