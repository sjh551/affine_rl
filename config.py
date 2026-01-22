# 文件名: config.py
import numpy as np

# --- 物理参数 ---
ROBOT_RADIUS = 0.1          # 机器人半径 (论文中假设为圆形质点)
SAFE_MARGIN_FACTOR = 1.5    # 论文公式(3-8)提到的1.5倍半径安全范围 [cite: 330]
COMM_DELAY = 0.0            # 忽略通信延迟 [cite: 247]
DT = 0.1                    # 时间步长 0.1s (10Hz) [cite: 325]

# --- 训练/奖励参数 ---
COLLISION_PENALTY_BASE = 400 # 论文后续章节提到的碰撞惩罚基数 [cite: 615]
GOAL_REWARD_SCALE = 1.0      # 目标奖励缩放

# --- 地图参数 ---
MAP_SIZE = 5.0              # 地图边长 (-5 到 5)
NUM_AGENTS = 3              # 机器人数量 (1 Leader + 2 Followers)