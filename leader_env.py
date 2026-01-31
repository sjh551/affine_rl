import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

class SingleRobotPyBulletEnv(gym.Env):
    def __init__(self, render_mode=False, enable_obstacles=True):
        super(SingleRobotPyBulletEnv, self).__init__()
        
        # --- 1. PyBullet 初始化配置 ---
        self.render_mode = render_mode
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 2. 机器人参数 ---
        self.num_robots = 1  # <--- 修改点：改为 1 个机器人
        self.robot_ids = []
        self.robot_radius = 0.25
        
        # 删除了 self.offsets (不需要跟随偏移量了)
        
        # --- 3. 定义空间 ---
        # 动作: [vx, vy]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 观测空间: 
        # Leader状态(4) + 相对目标(2) + 3个最近障碍物(3*3=9) = 15维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.target_pos = np.array([0, 0, 0])
        self.obstacles = []
        self.max_steps = 1500

        self.wall_length = 10.0  
        self.wall_width = 5.0   
        self.enable_obstacles = enable_obstacles
        self.last_dist = None
        
        # Lidar 参数 (保留你的定义，尽管当前obs没用到)
        self.lidar_range = 3.0
        self.lidar_fov = 180
        self.lidar_rays = 9

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # --- 创建围墙 ---
        self._create_walls()
        
        # --- A. 创建机器人 (只创建一个 Leader) ---
        self.robot_ids = []
        # 起点设置在左侧
        start_pos = [-self.wall_length + 3.0, 0, 0.25]
        # 创建唯一的机器人 (红色)
        leader_id = self._create_box_robot(start_pos, [1, 0, 0, 1])
        self.robot_ids.append(leader_id) 
        
        # <--- 修改点：删除了原本在这里生成 Follower 的循环 --->

        # --- B. 随机生成目标点 ---
        # tx = np.random.uniform(self.wall_length-3.0, self.wall_length-1.0)
        # ty = np.random.uniform(-self.wall_width+2.0, self.wall_width-2.0)
        self.target_pos = np.array([8.5, 0, 0]) # 固定目标点便于测试
        
        self.target_visual = self._create_visual_sphere(self.target_pos, [0, 1, 0, 0.5])
        
        # --- C. 生成障碍物 ---
        if self.enable_obstacles:
            self._generate_obstacles()
        else:
            self.obstacles = []
        
        self.current_step = 0

        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        self.last_dist = np.linalg.norm(np.array(leader_pos[:2]) - self.target_pos[:2])

        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        
        # --- 1. 机器人控制 (唯一的机器人) ---
        # 限制速度，防止动作过大
        vel_cmd = np.array([action[0], action[1], 0]) * 2.0
        p.resetBaseVelocity(self.robot_ids[0], linearVelocity=vel_cmd)
        
        # <--- 修改点：删除了 Follower 控制循环 --->
            
        # --- 2. 物理步进 ---
        for _ in range(10):
            p.stepSimulation()

        if self.render_mode:
            time.sleep(0.01)
            
        # --- 3. 计算奖励和状态 ---
        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        leader_pos = np.array(leader_pos)

        obs = self._get_obs()
        reward = self._compute_reward(leader_pos)
        
        # --- 4. 结束条件判断 ---
        done = False
        dist_to_goal = np.linalg.norm(leader_pos[:2] - self.target_pos[:2])
        
        # a. 成功到达
        if dist_to_goal < 0.5:
            reward += 100
            done = True
        
        # b. 超时
        elif self.current_step >= self.max_steps:
            reward -= 10 
            done = True
            
        # c. 出界检测 (X轴 或 Y轴)
        elif abs(leader_pos[0]) > self.wall_length - 0.5 or abs(leader_pos[1]) > self.wall_width - 0.5:
            reward -= 100 
            done = True
            
        # d. 碰撞检测 (障碍物)
        else:
            contact_points = p.getContactPoints(bodyA=self.robot_ids[0])
            for contact in contact_points:
                # contact[2] 是 bodyB 的 ID。
                # 如果撞到的不是地面(0) 且 不是自己(其实只有自己了)，那就是墙或者障碍物
                if contact[2] != 0 and contact[2] not in self.robot_ids: 
                    reward -= 50
                    done = True
                    break

        # --- 打印 Debug 信息 ---
        if done:
            if dist_to_goal < 0.5: 
                print("原因：到达终点")
            elif self.current_step >= self.max_steps: 
                print("原因：步数耗尽（超时）")
            # 修正了之前的打印逻辑，包含Y轴判断
            elif abs(leader_pos[0]) > self.wall_length - 0.5 or abs(leader_pos[1]) > self.wall_width - 0.5: 
                print("原因：越界/撞墙 (X或Y轴)")
            else: 
                print("原因：检测到物理碰撞 (障碍物)") 

        return obs, reward, done, {}

    def _get_obs(self):
        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        leader_vel, _ = p.getBaseVelocity(self.robot_ids[0])
        
        # 1. 自身状态 [x, y, vx, vy] (归一化位置)
        obs = [leader_pos[0]/self.wall_length, leader_pos[1]/self.wall_width, leader_vel[0], leader_vel[1]]
        
        # 2. 相对目标向量 [dx, dy]
        rel_goal = self.target_pos - np.array(leader_pos)
        obs.extend([rel_goal[0]/self.wall_length, rel_goal[1]/self.wall_width])
        
        # <--- 修改点：彻底删除了 Follower 的观测代码 --->
            
        # 3. 最近的3个障碍物信息 [dist, dx, dy]
        dists = []
        for obs_id in self.obstacles:
            o_pos, _ = p.getBasePositionAndOrientation(obs_id)
            d = np.linalg.norm(np.array(o_pos[:2]) - np.array(leader_pos[:2]))
            rel = np.array(o_pos[:2]) - np.array(leader_pos[:2])
            dists.append((d, rel[0], rel[1]))
        
        dists.sort(key=lambda x: x[0])
        
        for i in range(3):
            if i < len(dists):
                # 如果有障碍物，加入相对位置
                obs.extend([dists[i][1]/self.wall_length, dists[i][2]/self.wall_width, 0.3])
            else:
                # 如果障碍物不足3个（或者没有），填充默认安全值
                obs.extend([1.0, 1.0, 0]) 
                
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, leader_pos):
        # 1. 目标奖励
        current_dist = np.linalg.norm(np.array(leader_pos[:2]) - self.target_pos[:2])

        if not hasattr(self, 'last_dist'):
            self.last_dist = current_dist

        # 2. 进步奖励
        reward_progress = (self.last_dist - current_dist) * 100.0
        self.last_dist = current_dist
        
        # 3. 时间惩罚
        reward_step = -0.1 
        
        # 4. 避障惩罚
        reward_obs = 0
        if self.enable_obstacles:
            for obs_id in self.obstacles:
                o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                d = np.linalg.norm(np.array(o_pos[:2]) - np.array(leader_pos[:2]))
                
                safe_threshold = 0.9 
                
                if d < safe_threshold:
                    penalty = (safe_threshold - d) * 2.0
                    reward_obs -= penalty

        # 5. 离墙太近惩罚
        limit_x = self.wall_length - 0.5 
        limit_y = self.wall_width - 0.5
        if abs(leader_pos[0]) > limit_x or abs(leader_pos[1]) > limit_y:
            reward_obs -= 0.5
            
        return reward_progress + reward_step + reward_obs

    # --- 辅助函数保持不变 ---
    def _create_walls(self):
        """创建四周的围墙"""
        self.wall_ids = []
        wall_thickness = 0.5
        wall_height = 1.0
        wall_length = self.wall_length 
        wall_width = self.wall_width   
        
        walls_config = [
            ([wall_length, 0, 0.5], [wall_thickness, wall_width, wall_height]), 
            ([-wall_length, 0, 0.5], [wall_thickness, wall_width, wall_height]),
            ([0, wall_width, 0.5], [wall_length, wall_thickness, wall_height]), 
            ([0, -wall_width, 0.5], [wall_length, wall_thickness, wall_height]) 
        ]
        
        for pos, extents in walls_config:
            v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.8, 0.8, 0.8, 1])
            c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                        baseVisualShapeIndex=v_shape, basePosition=pos)
            self.wall_ids.append(wall_id)

    def _create_box_robot(self, pos, color):
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=color)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision_shape, 
                                    baseVisualShapeIndex=visual_shape, basePosition=pos)
        return body_id

    def _create_visual_sphere(self, pos, color):
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color)
        body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=pos)
        return body_id

    def _generate_obstacles(self):
        self.obstacles = []
        config_id = 4#np.random.randint(0, 5)
        
        v_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=1, rgbaColor=[0.1, 0.1, 0.1, 1])
        c_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=1)

        pos_list = []
        mid_x_min, mid_x_max = -4.0, 4.0
        mid_y_range = self.wall_width - 1.0 

        if config_id == 0:
            for x in [-3, 0, 3]:
                pos_list.append([x, 1.5, 0.5])
                pos_list.append([x, -1.5, 0.5])
        elif config_id == 1:
            for y in np.linspace(-self.wall_width, self.wall_width, 6):
                if abs(y) > 1.2:
                    pos_list.append([0, y, 0.5])
        elif config_id == 2:
            for x in np.linspace(-4, 4, 3):
                for y in [-2.5, 2.5]:
                    jitter_x = np.random.uniform(-0.5, 0.5)
                    jitter_y = np.random.uniform(-0.5, 0.5)
                    pos_list.append([x + jitter_x, y + jitter_y, 0.5])
        elif config_id == 3:
            pos_list = [
                [-3, 2, 0.5], [-3, 0, 0.5], 
                [0, 0, 0.5], [0, -2, 0.5], 
                [3, 2, 0.5], [3, 0, 0.5] 
            ]
        else:
            while len(pos_list) < 6:
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                is_far_enough = True
                for p_exist in pos_list:
                    if np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) < 2.0:
                        is_far_enough = False
                        break
                if is_far_enough:
                    pos_list.append([ox, oy, 0.5])

        for pos in pos_list:
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                        baseVisualShapeIndex=v_shape, basePosition=pos)
            self.obstacles.append(obs_id)
    
    # 保留了 _get_lidar 函数，虽然暂时没用，但如果你以后想切换成 Lidar 观测可以用
    def _get_lidar(self, pos):
       # ... (保持原样) ...
       pass