import pybullet as p
import pybullet_data
import os

import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

class DualLeaderEnv(gym.Env):
    def __init__(self, render_mode=False, enable_obstacles=True):
        super(DualLeaderEnv, self).__init__()
        
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
        
        self._create_walls()
        
        # --- 创建两个 Leader ---
        self.robot_ids = []
        # Leader A (左路)
        start_pos_A = [-self.wall_length + 2.0, -0.75, 0.25] 
        self.robot_ids.append(self._create_box_robot(start_pos_A, [1, 0, 0, 1])) # 红
        
        # Leader B (右路)
        start_pos_B = [-self.wall_length + 2.0, 0.75, 0.25]
        self.robot_ids.append(self._create_box_robot(start_pos_B, [0, 0, 1, 1])) # 蓝
        
        # --- 随机目标 ---
        tx = np.random.uniform(self.wall_length-3.0, self.wall_length-1.0)
        ty = np.random.uniform(-3.0, 3.0)
        self.target_pos = np.array([tx, ty, 0])
        self._create_visual_sphere(self.target_pos, [0, 1, 0, 0.5])
        
        # --- 障碍物 ---
        if self.enable_obstacles:
            self._generate_obstacles()
        else:
            self.obstacles = []
            
        self.current_step = 0
        
        # 初始化距离记录 (用两车中点计算)
        center_pos = self._get_center_pos()
        self.last_center_dist = np.linalg.norm(center_pos[:2] - self.target_pos[:2])
        
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        
        # --- 1. 双重控制 ---
        # 解析动作 [Ax, Ay, Bx, By]
        vel_A = np.array([action[0], action[1], 0]) * 1.5 # 限制最大速度
        vel_B = np.array([action[2], action[3], 0]) * 1.5
        
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
            rel_AB[0]/5.0, rel_AB[1]/5.0 # 队友相对位置 (2)
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
        
        # --- A. 进度奖励 (驱动力) ---
        # 既然是双车，用“中心点”是否接近目标来给分
        dist_to_goal = np.linalg.norm(center_pos[:2] - self.target_pos[:2])
        if not hasattr(self, 'last_center_dist'): self.last_center_dist = dist_to_goal
        
        reward += (self.last_center_dist - dist_to_goal) * 100.0
        self.last_center_dist = dist_to_goal
        
        reward -= 0.1 # 时间惩罚
        
        # --- B. 队形保持奖励 (弹簧约束) ---
        # 这一步是实现“自适应变换”的关键
        # 1. 计算两车实际距离
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        curr_dist = np.linalg.norm(np.array(pos_A[:2]) - np.array(pos_B[:2]))
        
        # 2. 计算误差：我们希望默认保持 desired_formation_dist (1.5m)
        # 但是！如果遇到障碍物，我们允许它变小。
        # 这是一个软约束：dist_error = abs(curr_dist - 1.5)
        # 惩罚系数给小一点，允许它为了避障而违反队形
        dist_error = abs(curr_dist - self.desired_formation_dist)
        
        if dist_error > 0.5: # 只有偏差超过 0.5米 才开始惩罚，给它变形的空间
             reward -= dist_error * 0.2
             
        # 防止两车相撞 (太近了要重罚)
        if curr_dist < 0.6:
            reward -= 1.0
            
        # 防止两车走散 (太远了说明没配合)
        if curr_dist > 4.0:
            reward -= 1.0

        # --- C. 安全检测 (双车都要检测) ---
        robots_crashed = False
        if self.enable_obstacles:
            for rid in self.robot_ids:
                r_pos, _ = p.getBasePositionAndOrientation(rid)
                # 遍历障碍物
                for obs_id in self.obstacles:
                    o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                    d = np.linalg.norm(np.array(o_pos[:2]) - np.array(r_pos[:2]))
                    
                    # 势场惩罚
                    if d < 1.0:
                        reward -= (1.0 - d) * 2.0
                    
                    # 物理碰撞检测
                    if d < 0.55: # 撞上了
                        robots_crashed = True
        
        # --- D. 终止条件 ---
        # 1. 到达
        if dist_to_goal < 0.5:
            reward += 100
            done = True
            reason = "Success"
        # 2. 撞墙/出界
        elif abs(center_pos[0]) > self.wall_length - 0.5:
            reward -= 50
            done = True
            reason = "Wall"
        # 3. 撞障碍物
        elif robots_crashed:
            reward -= 100
            done = True
            reason = "Crash"
        # 4. 超时
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"

        if done: print(f"Episode End: {reason}")
        
        return reward, {'done': done}

    # --- 辅助函数 (创建墙、机器人、球、障碍物) 保持不变 ---
    # 为了节省篇幅，这里复用你之前的 _create_walls, _create_box_robot, 
    # _create_visual_sphere, _generate_obstacles 方法即可。
    # 只需确保 _create_box_robot 里不要固定 pos，而是传入参数。
    
    def _create_walls(self):
        # ... (直接复制之前的代码) ...
        self.wall_ids = []
        walls_config = [
            ([self.wall_length, 0, 0.5], [0.5, self.wall_width, 1]), 
            ([-self.wall_length, 0, 0.5], [0.5, self.wall_width, 1]),
            ([0, self.wall_width, 0.5], [self.wall_length, 0.5, 1]), 
            ([0, -self.wall_width, 0.5], [self.wall_length, 0.5, 1])
        ]
        for pos, extents in walls_config:
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.8,0.8,0.8,1])
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            p.createMultiBody(0, c, v, basePosition=pos)

    def _create_box_robot(self, pos, color):
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=color)
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        return p.createMultiBody(1, c, v, basePosition=pos)

    def _create_visual_sphere(self, pos, color):
        v = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color)
        return p.createMultiBody(baseVisualShapeIndex=v, basePosition=pos)

    def _generate_obstacles(self):
        # ... (逻辑保持不变，确保 obstacles 列表被填充) ...
        self.obstacles = []
        # 简单起见，先生成一个配置
        obs_positions = [[0, 1.5, 0.5], [0, -1.5, 0.5]] # 窄门
        v = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=1, rgbaColor=[0.1,0.1,0.1,1])
        c = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=1)
        for pos in obs_positions:
            self.obstacles.append(p.createMultiBody(0, c, v, basePosition=pos))


# 1. 连接物理引擎
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 添加内置资源路径
p.setGravity(0, 0, -9.8)
max_v = 1.0  # 最大线速度 (m/s) [cite: 185]
max_w = 1.0  # 最大角速度 (rad/s) [cite: 185]
wheel_indices = [0, 1, 2, 3] # 假设 0,1 是左, 2,3 是右，请根据你的打印结果修改
# 2. 加载地面
planeId = p.loadURDF("plane.urdf")

# 3. 设置模型路径 (请替换为你本地的实际路径)
# 建议使用绝对路径，避免找不到 mesh 文件的报错
urdf_root = "./jackal_description/urdf"
urdf_file = os.path.join(urdf_root, "jackal.urdf")

# 4. 加载机器人
# flags=p.URDF_USE_INERTIA_FROM_FILE 是为了确保物理参数准确
robotId = p.loadURDF(urdf_file, [0, 0, 0.5], flags=p.URDF_USE_INERTIA_FROM_FILE)
# robotId = p.loadURDF("racecar/racecar_differential.urdf", [0, 0, 0.05])

# 打印关节信息，确认哪些是驱动轮
for i in range(p.getNumJoints(robotId)):
    info = p.getJointInfo(robotId, i)
    print(f"Joint Index: {info[0]}, Name: {info[1].decode('utf-8')}")

while True:
    # 1. 获取键盘事件
    keys = p.getKeyboardEvents()
    
    linear_v = 0
    angular_w = 0

    # 2. 解析按键 (使用 Pybullet 内置键码)
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        linear_v = max_v
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        linear_v = -max_v
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        angular_w = max_w
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        angular_w = -max_w

    # 3. 差速运动学转换 (简易版)
    # 这里的 0.4 是假设的车宽（轮距），你可以根据 Jackal 实际参数调整
    wheel_sep = 0.4 
    v_left = linear_v - angular_w * wheel_sep / 2
    v_right = linear_v + angular_w * wheel_sep / 2

    # 4. 设置电机控制
    # 假设索引 0, 1 是左轮
    for i in [0, 1]:
        p.setJointMotorControl2(robotId, i, p.VELOCITY_CONTROL, targetVelocity=v_left, force=10)
    # 假设索引 2, 3 是右轮
    for i in [2, 3]:
        p.setJointMotorControl2(robotId, i, p.VELOCITY_CONTROL, targetVelocity=v_right, force=10)

    p.stepSimulation()



# train.py里面的代码片段:

    # def _compute_reward(self, leader_pos):
    #     # 1. 目标奖励
    #     dist = np.linalg.norm(np.array(leader_pos[:2]) - self.target_pos[:2])
    #     r_goal = 2.0 / (dist + 0.1)
        
    #     # 2. 避障引导 (距离惩罚)
    #     r_coll = 0
    #     if self.enable_obstacles:
    #         for obs_id in self.obstacles:
    #             o_pos, _ = p.getBasePositionAndOrientation(obs_id)
    #             d = np.linalg.norm(np.array(o_pos[:2]) - np.array(leader_pos[:2]))
    #             if d < 0.8:
    #                 r_coll -= 1.0 / (d + 0.1)
        
    #     # 3. 边界引导
    #     # 如果离墙太近 (< 0.5m)，也给点负分，教它不要贴墙走
    #     limit_x = self.wall_length - 0.5 
    #     limit_y = self.wall_width - 0.5

    #     if abs(leader_pos[0]) > limit_x or abs(leader_pos[1]) > limit_y:
    #         r_coll -= 0.5
                
    #     return r_goal + r_coll



