# 文件名: robot.py
import numpy as np
from config import DT, ROBOT_RADIUS

class Robot:
    def __init__(self, agent_id, start_pos):
        self.id = agent_id
        self.pos = np.array(start_pos, dtype=np.float32) # 位置 p_i
        self.vel = np.array([0.0, 0.0], dtype=np.float32) # 速度 v_i
        self.radius = ROBOT_RADIUS
        
        # 记录轨迹用于画图
        self.traj_x = [self.pos[0]]
        self.traj_y = [self.pos[1]]

    def update(self, action_force):
        """
        根据动作更新状态
        论文公式 (3-1): dp/dt = v, dv/dt = F/m (简化设m=1)
        """
        # 限制最大加速度/力，防止飞出屏幕
        force = np.clip(action_force, -1.0, 1.0)
        
        # 更新速度 (v = v + a*t)
        self.vel += force * DT
        
        # 增加一点阻尼(摩擦力)，让运动更真实，否则会一直加速
        self.vel *= 0.9 
        
        # 更新位置 (p = p + v*t)
        self.pos += self.vel * DT
        
        # 记录轨迹
        self.traj_x.append(self.pos[0])
        self.traj_y.append(self.pos[1])

    def get_state(self):
        return self.pos, self.vel