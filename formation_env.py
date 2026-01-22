# 文件名: formation_env.py
import gym
import numpy as np
from gym import spaces
from config import *
from robot import Robot

class FormationGymEnv(gym.Env):
    def __init__(self):
        super(FormationGymEnv, self).__init__()
        
        # 初始化 3 个机器人
        self.robots = [
            Robot(0, [0.0, 0.0]),   # Leader
            Robot(1, [-1.0, 0.0]),  # Follower 1
            Robot(2, [1.0, 0.0])    # Follower 2
        ]
        
        # Follower 的目标偏移量 (三角形队形)
        self.offsets = {
            1: np.array([-0.8, -0.8]), 
            2: np.array([0.8, -0.8])
        }

        # --- 动作空间 (RL 输出 Leader 的速度) ---
        # 范围 [-1, 1]，对应 x 和 y 方向的速度比例
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- 状态空间 (输入给神经网络的数据) ---
        # [Leader位置(2), Leader速度(2), 目标相对位置(2), 两个Follower相对位置(4)] = 10维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.target_pos = np.array([0.0, 0.0])
        self.max_steps = 300  # 每一局最多跑300步
        self.current_step = 0

    def reset(self):
        """重置环境，开始新的一局"""
        self.current_step = 0
        
        # 1. 随机生成 Leader 位置和 目标点 (让 AI 适应各种情况)
        start_x = np.random.uniform(-3, 3)
        start_y = np.random.uniform(-3, 3)
        self.robots[0].pos = np.array([start_x, start_y], dtype=np.float32)
        self.robots[0].vel = np.zeros(2, dtype=np.float32)
        # 清空轨迹
        self.robots[0].traj_x = [start_x]
        self.robots[0].traj_y = [start_y]
        
        # 随机目标点 (距离起点至少 2 米)
        while True:
            tx, ty = np.random.uniform(-4, 4), np.random.uniform(-4, 4)
            self.target_pos = np.array([tx, ty], dtype=np.float32)
            if np.linalg.norm(self.target_pos - self.robots[0].pos) > 2.0:
                break
        
        # 2. 重置 Follower (直接放在 Leader 旁边)
        for i in [1, 2]:
            self.robots[i].pos = self.robots[0].pos + self.offsets[i]
            self.robots[i].vel = np.zeros(2, dtype=np.float32)
            self.robots[i].traj_x = [self.robots[i].pos[0]]
            self.robots[i].traj_y = [self.robots[i].pos[1]]

        return self._get_obs()

    def step(self, action):
        """核心循环：执行一步"""
        self.current_step += 1
        
        # --- A. Leader 移动 (由 RL 控制) ---
        # action 是神经网络输出的 [-1, 1] 之间的数，映射为速度
        leader = self.robots[0]
        leader.update(action) 
        
        # --- B. Follower 移动 (由 PID 控制 - 混合架构) ---
        # 你的任务书允许 Leader-Follower 架构，所以 Follower 用 PID 是合规且聪明的做法
        for i in [1, 2]:
            follower = self.robots[i]
            target = leader.pos + self.offsets[i] # 期望位置
            
            # 简单的 PID 公式: 动作 = P * 误差 - D * 速度
            pid_force = 2.0 * (target - follower.pos) - 1.0 * follower.vel
            follower.update(pid_force)

        # --- C. 计算奖励 ---
        reward = self._compute_reward()
        
        # --- D. 判断是否结束 ---
        dist_to_goal = np.linalg.norm(leader.pos - self.target_pos)
        done = False
        info = {}
        
        # 成功：到达目标附近
        if dist_to_goal < 0.2:
            reward += 100.0  # 给个大鸡腿
            done = True
            info['result'] = 'success'
            
        # 失败：超时
        if self.current_step >= self.max_steps:
            done = True
            info['result'] = 'timeout'

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """获取 Leader 的观测数据"""
        me = self.robots[0]
        # 归一化 (除以地图大小，让数值在 -1 到 1 之间，神经网络喜欢这样)
        p = me.pos / MAP_SIZE 
        v = me.vel 
        g = (self.target_pos - me.pos) / MAP_SIZE # 目标相对位置
        
        d_others = []
        for other in self.robots[1:]:
            d_others.extend((other.pos - me.pos) / MAP_SIZE)
            
        return np.concatenate([p, v, g, np.array(d_others)])

    def _compute_reward(self):
        """计算奖励 (复现论文公式)"""
        me = self.robots[0]
        
        # 1. 靠近目标给分
        dist = np.linalg.norm(me.pos - self.target_pos)
        r_goal = 0.5 / (dist + 0.1) 
        
        # 2. 离 Follower 太近扣分 (防止撞队友)
        r_coll = 0.0
        safe_margin = 1.5 * (ROBOT_RADIUS * 2)
        for other in self.robots[1:]:
            d = np.linalg.norm(me.pos - other.pos)
            if d < safe_margin:
                r_coll -= 0.2 / (d - 2*ROBOT_RADIUS + 0.01) # 惩罚项
                
        return r_goal + r_coll