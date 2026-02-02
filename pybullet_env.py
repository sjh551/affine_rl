import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

class FormationPyBulletEnv(gym.Env):
    def __init__(self, render_mode=False, enable_obstacles=True):
        super(FormationPyBulletEnv, self).__init__()
        
        # --- 1. PyBullet 初始化配置 ---
        self.render_mode = render_mode
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 参数配置 ---
        self.num_robots = 2 # 只训练两个 Leader
        self.robot_ids = []
        self.robot_radius = 0.25
        
        # 期望队形距离 (两个Leader之间的理想距离)
        self.desired_formation_dist = 1.5 
        
        # --- 动作空间 (4维) ---
        # [LeaderA_Vx, LeaderA_Vy, LeaderB_Vx, LeaderB_Vy]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # --- 观测空间 (26维) ---
        # 1. 编队中心相对目标的向量 (2)
        # 2. Leader A 相对编队中心的偏移 (2) + 速度 (2)
        # 3. Leader B 相对编队中心的偏移 (2) + 速度 (2)
        # 4. Leader A 雷达 (9)
        # 5. Leader B 雷达 (9)
        # Total = 2 + 4 + 4 + 9 + 9 = 28 (为了方便我精简一下)
        # 实际我们用: [Rel_Goal_X, Rel_Goal_Y, A_Vx, A_Vy, B_Vx, B_Vy, Rel_A_B_X, Rel_A_B_Y, LidarA(9), LidarB(9)]
        # Total = 8 + 18 = 26
        self.n_lidar_rays = 9
        self.lidar_range = 3.0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        
        self.target_pos = np.array([8.5, 0, 0])
        self.obstacles = []
        self.max_steps = 1500
        self.wall_length = 10.0
        self.wall_width = 5.0
        self.enable_obstacles = enable_obstacles
        self.last_center_dist = None
        
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # --- 创建围墙 ---
        self._create_walls()
        
        # --- 创建两个 Leader ---
        self.robot_ids = []
        # Leader A (左路)
        start_pos_A = [-self.wall_length + 2.0, -0.75, 0.25] 
        self.robot_ids.append(self._create_box_robot(start_pos_A, [1, 0, 0, 1])) # 红
        
        # Leader B (右路)
        start_pos_B = [-self.wall_length + 2.0, 0.75, 0.25]
        self.robot_ids.append(self._create_box_robot(start_pos_B, [0, 0, 1, 1])) # 蓝

        # --- 随机生成目标点 ---
        tx = np.random.uniform(self.wall_length-3.0, self.wall_length-1.0)
        ty = np.random.uniform(-self.wall_width+2.0, self.wall_width-2.0)
        self.target_pos = np.array([tx, ty, 0])
        self._create_visual_sphere(self.target_pos, [0, 1, 0, 0.5])
        
        # --- C. 生成障碍物 (根据参数决定) ---
        if self.enable_obstacles:
            self._generate_obstacles()
        else:
            self.obstacles = [] # 确保列表为空
        
        self.current_step = 0

        # --- 初始化上一帧的速度记录 ---
        # 记录为 [vx, vy, vz]，虽然我们只用 xy，但 pybullet 返回 3 维
        self.last_vel_A = np.zeros(3)
        self.last_vel_B = np.zeros(3)

        # 初始化距离记录 (用两车中点计算)
        center_pos = self._get_center_pos()
        self.last_center_dist = np.linalg.norm(center_pos[:2] - self.target_pos[:2])
        
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        
        # --- 1. 双重控制 ---
        # 解析动作 [Ax, Ay, Bx, By]
        vel_A = np.array([action[0], action[1], 0]) * 1.0 # 限制最大速度
        vel_B = np.array([action[2], action[3], 0]) * 1.0

        p.resetBaseVelocity(self.robot_ids[0], linearVelocity=vel_A)
        p.resetBaseVelocity(self.robot_ids[1], linearVelocity=vel_B)

        # --- 2. 物理仿真 (高频率) ---
        for _ in range(10): 
            p.stepSimulation()
            
        if self.render_mode: time.sleep(0.01)
            
        # --- 3. 获取状态与计算奖励 ---
        center_pos = self._get_center_pos()
        obs = self._get_obs()
        reward, done_info = self._compute_reward(center_pos)
        done = done_info['done']

        return obs, reward, done, {}
    
    def _get_center_pos(self):
        """计算双车中心点"""
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        center = (np.array(pos_A) + np.array(pos_B)) / 2.0
        return center

    def _create_walls(self):
        """创建四周的围墙"""
        self.wall_ids = []
        wall_thickness = 0.5
        wall_height = 1.0
        wall_length = self.wall_length   # 围墙长
        wall_width = self.wall_width    # 围墙宽
        
        # 墙的参数: (中心位置, 长宽高)
        walls_config = [
            ([wall_length, 0, 0.5], [wall_thickness, wall_width, wall_height]),  # 右墙
            ([-wall_length, 0, 0.5], [wall_thickness, wall_width, wall_height]), # 左墙
            ([0, wall_width, 0.5], [wall_length, wall_thickness, wall_height]),  # 上墙
            ([0, -wall_width, 0.5], [wall_length, wall_thickness, wall_height])  # 下墙
        ]
        
        for pos, extents in walls_config:
            v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.8, 0.8, 0.8, 1])
            c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                        baseVisualShapeIndex=v_shape, basePosition=pos)
            self.wall_ids.append(wall_id)

    def _create_box_robot(self, pos, color):
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=color)
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        return p.createMultiBody(1, c, v, basePosition=pos)

    def _create_visual_sphere(self, pos, color):
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color)
        body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=pos)
        return body_id

    def _generate_obstacles(self):
        self.obstacles = []
        config_id = np.random.randint(0, 5)
        
        v_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=1, rgbaColor=[0.1, 0.1, 0.1, 1])
        c_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=1)

        pos_list = []
        
        # 确定中间障碍区的范围 (X轴中心地带)
        # 起点在 -7 左右，终点在 +7 左右，中间地带定为 [-4, 4]
        mid_x_min, mid_x_max = -4.0, 4.0
        mid_y_range = self.wall_width - 1.0  # Y轴摆放范围

        if config_id == 0:
            # --- 配置 1：双排“之”字形（强迫编队 S 型走位） ---
            for x in [-3, 0, 3]:
                pos_list.append([x, 1.5, 0.5])
                pos_list.append([x, -1.5, 0.5])

        elif config_id == 1:
            # --- 配置 2：窄门（考验编队收缩/精确通过） ---
            # 在 X=0 处留一个上下两个大障碍物组成的中间缝隙
            for y in np.linspace(-self.wall_width, self.wall_width, 6):
                if abs(y) > 1.2: # 只有中间 2.4 米宽的缝隙可以通过
                    pos_list.append([0, y, 0.5])

        elif config_id == 2:
            # --- 配置 3：随机森林（均匀分布） ---
            # 使用简单的网格抖动算法，确保障碍物分布均匀不堆叠
            for x in np.linspace(-4, 4, 3):
                for y in [-2.5, 2.5]:
                    jitter_x = np.random.uniform(-0.5, 0.5)
                    jitter_y = np.random.uniform(-0.5, 0.5)
                    pos_list.append([x + jitter_x, y + jitter_y, 0.5])

        elif config_id == 3:
            # --- 配置 4：横向交错墙 ---
            pos_list = [
                [-3, 2, 0.5], [-3, 0, 0.5],  # 上方半堵墙
                [0, 0, 0.5], [0, -2, 0.5],   # 下方半堵墙
                [3, 2, 0.5], [3, 0, 0.5]     # 上方半堵墙
            ]

        else:
            # --- 配置 5：动态泊松盘采样的简化版（完全随机但均匀） ---
            while len(pos_list) < 6:
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                
                # 检查与已生成障碍物的距离，确保不挤在一起
                is_far_enough = True
                for p_exist in pos_list:
                    if np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) < 2.0:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    pos_list.append([ox, oy, 0.5])

        # 统一生成
        for pos in pos_list:
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                       baseVisualShapeIndex=v_shape, basePosition=pos)
            self.obstacles.append(obs_id)

    def _get_obs(self):
        # 获取机器人状态
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        vel_A, _ = p.getBaseVelocity(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        vel_B, _ = p.getBaseVelocity(self.robot_ids[1])
        
        # 1. 目标相对信息的 (用中心点看目标)
        center_pos = (np.array(pos_A) + np.array(pos_B)) / 2.0
        rel_goal = self.target_pos - center_pos
        
        # 2. 编队向量 (A 看 B) -> 让网络知道队友在哪
        rel_AB = np.array(pos_B) - np.array(pos_A)
        
        # 3. 雷达数据 (两人都要看)
        lidar_A = self._get_lidar(self.robot_ids[0])
        lidar_B = self._get_lidar(self.robot_ids[1])
        
        # 归一化/组装 Observation
        # 坐标除以墙长做简单的归一化
        obs = [
            rel_goal[0]/self.wall_length, rel_goal[1]/self.wall_width, # 目标向量 (2)
            vel_A[0], vel_A[1],  # A速度 (2)
            vel_B[0], vel_B[1],  # B速度 (2)
            rel_AB[0]/self.wall_length, rel_AB[1]/self.wall_width # 队友相对位置 (2)
        ]
        obs.extend(lidar_A) # (9)
        obs.extend(lidar_B) # (9)
        
        return np.array(obs, dtype=np.float32)

    def _get_lidar(self, robot_id):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        start_pos = [pos[0], pos[1], 0.25]
        
        ray_froms = []
        ray_tos = []
        # 180度扫描前方 (从 -90 到 +90)
        for angle in np.linspace(-np.pi/2, np.pi/2, self.n_lidar_rays):
            # 这里简单处理：假设雷达坐标系随世界坐标系 (为了学会转向，建议结合速度方向，但此处先简化)
            end_x = start_pos[0] + self.lidar_range * np.cos(angle)
            end_y = start_pos[1] + self.lidar_range * np.sin(angle)
            ray_froms.append(start_pos)
            ray_tos.append([end_x, end_y, 0.25])
            
        results = p.rayTestBatch(ray_froms, ray_tos)
        readings = []
        for res in results:
            readings.append(res[2]) # hit fraction
        return readings
    
    
    def _compute_reward(self, center_pos):
        reward = 0
        done = False
        reason = ""
        
        # --- 1. 进度奖励 ---
        dist_to_goal = np.linalg.norm(center_pos[:2] - self.target_pos[:2])
        if not hasattr(self, 'last_center_dist'): self.last_center_dist = dist_to_goal
        
        # 稍微降低进度奖励权重，防止它为了赶路忽略队形
        # 改为 50.0 (原 100.0)
        reward += (self.last_center_dist - dist_to_goal) * 50.0 
        self.last_center_dist = dist_to_goal
        
        reward -= 0.1 # 时间消耗

        # --- 平滑性惩罚 (Smoothness Penalty) ---
        # 获取当前速度
        vel_A, _ = p.getBaseVelocity(self.robot_ids[0])
        vel_B, _ = p.getBaseVelocity(self.robot_ids[1])
        vel_A = np.array(vel_A)
        vel_B = np.array(vel_B)

        # 计算加速度（即速度的变化量），只看 XY 平面即可
        # jerk (急动度) = || v_t - v_{t-1} ||
        # 使用 L2 范数
        diff_A = np.linalg.norm(vel_A[:2] - self.last_vel_A[:2])
        diff_B = np.linalg.norm(vel_B[:2] - self.last_vel_B[:2])

        # 系数建议：0.1 ~ 0.5。如果太大，机器人会变得反应迟钝（懒得加速）
        # 如果太小，压不住抖动。先试 0.2
        smooth_penalty = (diff_A + diff_B) * 0.5
        reward -= smooth_penalty

        # [重要] 更新历史速度，供下一步使用
        self.last_vel_A = vel_A
        self.last_vel_B = vel_B
        
        # --- 2. 队形保持 (关键修改) ---
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        curr_dist = np.linalg.norm(np.array(pos_A[:2]) - np.array(pos_B[:2]))
        
        dist_error = abs(curr_dist - self.desired_formation_dist)
        
        # 如果误差 > 0.5，开始惩罚。
        # [修改] 提高权重：由 0.2 提升到 1.0，让它重视队形
        if dist_error > 1.0: 
             reward -= dist_error * 1.0
             
        # --- 3. 碰撞与出界检测 (逻辑修复) ---
        
        # A. 互撞检测
        if curr_dist < 0.5: # 机器人直径0.4，0.5意味着已经蹭上了
            reward -= 50
            done = True
            reason = "Crash_Teammate"
            
        # B. 撞墙检测 (必须分开查！)
        limit_x = self.wall_length - 0.5
        limit_y = self.wall_width - 0.5
        # 只要有一个出界，就结束
        if (abs(pos_A[0]) > limit_x or abs(pos_A[1]) > limit_y or 
            abs(pos_B[0]) > limit_x or abs(pos_B[1]) > limit_y):
            reward -= 50
            done = True
            reason = "Wall_Out"

        # C. 障碍物检测
        if self.enable_obstacles and not done: # 如果还没挂，才查障碍物
            for rid in self.robot_ids:
                r_pos, _ = p.getBasePositionAndOrientation(rid)
                for obs_id in self.obstacles:
                    o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                    d = np.linalg.norm(np.array(o_pos[:2]) - np.array(r_pos[:2]))
                    
                    # 势场避障 (给予微小负反馈，提示它绕路)
                    if d < 1.0:
                        reward -= (1.0 - d) * 0.5
                    
                    # 物理碰撞
                    if d < 0.55:
                        reward -= 100
                        done = True
                        reason = "Crash_Obstacle"
                        break
                if done: break

        # --- 4. 成功判定 ---
        if not done and dist_to_goal < 0.5:
            reward += 100
            done = True
            reason = "Success"
        
        # --- 5. 超时判定 ---
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"

        if done: 
            print(f"Episode End: {reason}")
            # [可视化小技巧] 如果碰了，可以让最后一步停顿久一点方便观察
            # if self.render_mode and reason != "Success" and reason != "Timeout":
            #     time.sleep(1.0) 
        
        return reward, {'done': done}