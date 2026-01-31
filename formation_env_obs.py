# 带有障碍物信息的Formation环境，没有使用 PyBullet
import gym
import numpy as np
from gym import spaces
from config import *
from robot import Robot

class FormationGymEnv(gym.Env):
    def __init__(self):
        super(FormationGymEnv, self).__init__()
        
        # --- 1. 初始化参数 ---
        self.n_robots = 3
        self.robots = [
            Robot(0, [0.0, 0.0]),   # Leader
            Robot(1, [-1.0, 0.0]),  # Follower 1
            Robot(2, [1.0, 0.0])    # Follower 2
        ]
        
        # 队形偏移量 (Follower相对于Leader的位置)
        self.offsets = {
            1: np.array([-0.8, -0.8]), 
            2: np.array([0.8, -0.8])
        }

        # --- 2. 障碍物设置 ---
        self.obstacles = [] # 存储格式: [x, y, radius]
        self.n_obs_input = 3 # 只把“最近的3个障碍物”的位置告诉神经网络
        self.obs_dim = 3     # 障碍物特征维度 (dx, dy, radius)

        # --- 3. 定义动作空间 (RL 输出 Leader 的速度) ---
        # 范围 [-1, 1]，映射为 x 和 y 方向的速度比例
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- 4. 定义状态空间 (输入给神经网络的数据) ---
        # Leader状态(4) + 目标相对位置(2) + Follower相对位置(4) + 最近障碍物信息(3*3) = 19维
        # Leader: [px, py, vx, vy] (归一化后)
        state_dim = 4 + 2 + 4 + (self.n_obs_input * self.obs_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.target_pos = np.array([0.0, 0.0])
        self.max_steps = 400  # 稍微增加一点步数，给它绕路的时间
        self.current_step = 0

    def reset(self):
        """重置环境，开始新的一局"""
        self.current_step = 0
        
        # A. 随机生成 Leader 和 目标
        # 确保起点和终点距离较远，给 AI 表现机会
        while True:
            start_x, start_y = np.random.uniform(-4, 4), np.random.uniform(-4, 4)
            tx, ty = np.random.uniform(-4, 4), np.random.uniform(-4, 4)
            
            self.robots[0].pos = np.array([start_x, start_y], dtype=np.float32)
            self.target_pos = np.array([tx, ty], dtype=np.float32)
            
            if np.linalg.norm(self.target_pos - self.robots[0].pos) > 4.0:
                break
        
        # 重置 Leader 状态
        self.robots[0].vel = np.zeros(2, dtype=np.float32)
        self.robots[0].traj_x = [self.robots[0].pos[0]]
        self.robots[0].traj_y = [self.robots[0].pos[1]]
        
        # B. 重置 Follower (直接瞬移到 Leader 旁边)
        for i in [1, 2]:
            self.robots[i].pos = self.robots[0].pos + self.offsets[i]
            self.robots[i].vel = np.zeros(2, dtype=np.float32)
            self.robots[i].traj_x = [self.robots[i].pos[0]]
            self.robots[i].traj_y = [self.robots[i].pos[1]]

        # C. 生成障碍物 (这里对应毕设的“随机生成配置”)
        self._generate_obstacles()

        return self._get_obs()

    def step(self, action):
        """核心循环：执行一步"""
        self.current_step += 1
        
        # --- A. Leader 移动 (RL 控制) ---
        leader = self.robots[0]
        leader.update(action) 
        
        # --- B. Follower 移动 (PID 控制) ---
        for i in [1, 2]:
            follower = self.robots[i]
            target = leader.pos + self.offsets[i]
            # 简单的 PID 保持队形
            pid_force = 2.0 * (target - follower.pos) - 1.0 * follower.vel
            follower.update(pid_force)

        # --- C. 计算奖励 ---
        reward = self._compute_reward()
        
        # --- D. 判断结束条件 ---
        dist_to_goal = np.linalg.norm(leader.pos - self.target_pos)
        done = False
        info = {}
        
        # 1. 成功到达
        if dist_to_goal < 0.5:
            reward += 200.0  # 成功大奖
            done = True
            info['result'] = 'success'
        
        # 2. 发生严重碰撞 (撞障碍物)
        # 如果离障碍物太近 (小于半径之和)，直接结束，给大惩罚
        # 这能加快训练，让 AI 知道“撞墙即死”
        if self._check_collision(leader):
            reward -= 50.0 
            done = True 
            info['result'] = 'collision'

        # 3. 超时
        if self.current_step >= self.max_steps:
            done = True
            info['result'] = 'timeout'

        return self._get_obs(), reward, done, info

    def _generate_obstacles(self):
        """随机生成障碍物 (对应任务书指标1)"""
        self.obstacles = []
        num_obstacles = np.random.randint(3, 6) # 随机 3-5 个障碍物
        
        for _ in range(num_obstacles):
            # 简单的防重叠逻辑：不要生成在起点或终点附近
            while True:
                ox, oy = np.random.uniform(-4, 4), np.random.uniform(-4, 4)
                obs_pos = np.array([ox, oy])
                # 离起点 > 1.0, 离终点 > 1.0
                if (np.linalg.norm(obs_pos - self.robots[0].pos) > 1.5 and 
                    np.linalg.norm(obs_pos - self.target_pos) > 1.5):
                    break
            
            radius = np.random.uniform(0.3, 0.6) # 随机大小
            self.obstacles.append([ox, oy, radius])

    def _get_obs(self):
        """获取 Leader 的观测数据 (包含最近的障碍物)"""
        me = self.robots[0]
        
        # 1. 自身信息 (归一化)
        p = me.pos / MAP_SIZE 
        v = me.vel 
        g = (self.target_pos - me.pos) / MAP_SIZE 
        
        # 2. 队友信息
        d_followers = []
        for other in self.robots[1:]:
            d_followers.extend((other.pos - me.pos) / MAP_SIZE)
            
        # 3. 障碍物信息 (关键点：处理不定长问题)
        # 策略：找到所有障碍物，按距离排序，取最近的 3 个
        obs_list = []
        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            radius = obs[2]
            dist = np.linalg.norm(me.pos - obs_pos)
            rel_pos = (obs_pos - me.pos) / MAP_SIZE
            # 存储格式: [dist, rx, ry, radius]
            obs_list.append([dist, rel_pos[0], rel_pos[1], radius])
        
        # 按距离从小到大排序
        obs_list.sort(key=lambda x: x[0])
        
        # 取前 n_obs_input 个
        final_obs_feat = []
        for i in range(self.n_obs_input):
            if i < len(obs_list):
                # 如果有障碍物，放入 [rx, ry, radius]
                final_obs_feat.extend(obs_list[i][1:])
            else:
                # 如果障碍物不够3个，补0 (Padding)
                final_obs_feat.extend([0.0, 0.0, 0.0])
                
        return np.concatenate([p, v, g, np.array(d_followers), np.array(final_obs_feat)])

    def _check_collision(self, robot):
        """检查是否撞到障碍物 (物理碰撞)"""
        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            radius = obs[2]
            dist = np.linalg.norm(robot.pos - obs_pos)
            if dist < (ROBOT_RADIUS + radius):
                return True
        return False

    def _compute_reward(self):
        """计算奖励 (复现论文公式)"""
        me = self.robots[0]
        
        # 1. 目标奖励
        dist = np.linalg.norm(me.pos - self.target_pos)
        r_goal = 0.5 / (dist + 0.1) 
        
        # 2. 避障惩罚 (针对静态障碍物)
        r_obs = 0.0
        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            radius = obs[2]
            dist = np.linalg.norm(me.pos - obs_pos)
            
            # 安全距离 = 半径和 + 缓冲 (0.3)
            safe_threshold = ROBOT_RADIUS + radius + 0.3
            
            if dist < safe_threshold:
                # 论文中的反比例惩罚函数
                penalty = 0.5 / (dist - (ROBOT_RADIUS + radius) + 0.01)
                r_obs -= penalty
                
        return r_goal + r_obs