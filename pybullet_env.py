import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from math_utils import compute_W_matrix, get_3_agent_nominal_config

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
        self.n_leaders = 2
        self.n_followers = 1
        self.num_robots = self.n_leaders + self.n_followers # 训练两个 Leader + 1 Follower
        self.robot_ids = []
        self.robot_radius = 0.25
        
        # --- [整合核心 1] 初始化标称构型和权重 ---
        self.g_nom = get_3_agent_nominal_config()
        self.W_02, self.W_12, self.W_22_inv = self._init_formation_weights()


        # --- 动作空间 (4维) ---
        # [LeaderA_Vx, LeaderA_Vy, LeaderB_Vx, LeaderB_Vy]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.n_nearest_obs = 5 # 最近的5个障碍物
        # --- 观测空间 (38维) ---
        # 1. 目标相对向量 (2)
        # 2. A速度(2) + B速度(2)
        # 3. AB相对位置(2)
        # 4. A的最近5个障碍物 (5 * 2 = 10)
        # 5. B的最近5个障碍物 (5 * 2 = 10)
        # 6. C的最近5个障碍物 (5 * 2 = 10)
        # 2 + 4 + 2 + 10 + 10 + 10 = 38 维
        self.n_lidar_rays = 9
        self.lidar_range = 3.0

        obs_dim = 2 + 4 + 2 + (self.n_nearest_obs * 2 * 3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.target_pos = np.array([8.5, 0, 0])
        self.obstacles = []
        self.max_steps = 1500
        self.wall_length = 10.0
        self.wall_width = 5.0
        self.enable_obstacles = enable_obstacles
        self.last_center_dist = None
        # 用于可视化的线段ID
        self.debug_line_ids = []

    def _init_formation_weights(self):
        """
        计算 Follower 的权重矩阵
        [关键修改]：只取前两维 (X, Y)，扔掉第三维 (Angle)，避免奇异性。
        """
        theta = 0 
        g0 = self.g_nom[0] # Leader A
        g1 = self.g_nom[1] # Leader B
        g2 = self.g_nom[2] # Follower
        
        # 1. 计算原始 3x3 矩阵
        W_12_3x3, W_02_3x3 = compute_W_matrix(g0, g1, g2, theta)
        
        # 2. [新增步骤] 矩阵切片：只保留前 2 行前 2 列
        # 变成 2x2 矩阵
        self.W_02 = W_02_3x3[:2, :2]
        self.W_12 = W_12_3x3[:2, :2]
        
        # 3. 计算 M_22 (2x2)
        M_22 = self.W_02 + self.W_12
        
        # 4. 求逆 (现在 M_22 是 [[-0.4, 0], [0, -2]]，完全可逆！)
        self.W_22_inv = np.linalg.inv(M_22)
        
        return self.W_02, self.W_12, self.W_22_inv


    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # --- 创建围墙 ---
        self._create_walls()
        
        # --- [修改] 基于标称构型生成初始位置 ---
        # 1. 确定编队生成的中心点 (比如在左侧起跑线)
        start_center = np.array([-self.wall_length + 4.0, 0.0, 0.25])
        
        # 2. 读取标称构型 (我们在 __init__ 里定义的 g_nom)
        # 假设 g_nom 是以 (0,0) 为中心的
        # 格式: {0: [0, 1], 1: [0, -1], 2: [-1, 0]}


        # --- 创建两个 Leader ---
        self.robot_ids = []

        # 遍历三个智能体 (0:L1, 1:L2, 2:F)
        for i in range(3):
            # 取出标称偏移量 (2D)
            offset = self.g_nom[i] 
            
            # 计算实际生成位置: 中心 + 偏移
            spawn_pos = [
                start_center[0] + offset[0], # x
                start_center[1] + offset[1], # y
                0.25                         # z (高度固定)
            ]
            
            # 颜色区分: Leader(红), Follower(绿)
            color = [1, 0, 0, 1] if i < 2 else [0, 1, 0, 1]
            
            # 创建机器人
            rid = self._create_box_robot(spawn_pos, color)
            self.robot_ids.append(rid)  

        # --- 随机生成目标点 ---
        tx = np.random.uniform(self.wall_length-3.0, self.wall_length-2.0)
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
        # 记录上一帧速度用于平滑惩罚
        self.last_vels = [np.zeros(3) for _ in range(self.num_robots)]
        # [关键] 重置上一帧动作 (用于计算动作平滑度)
        self.last_action = np.zeros(self.action_space.shape)

        # 初始化距离记录 (用两车中点计算)
        center_pos = self._get_center_pos()
        # print(f"初始中心位置: {center_pos}, 目标位置: {self.target_pos}")
        self.last_center_dist = np.linalg.norm(center_pos[:2] - self.target_pos[:2])

        # 清除旧的线
        p.removeAllUserDebugItems()
        self.debug_line_ids = []
        
        return self._get_obs()

    def step(self, action):
        self.current_step += 1

        # --- 1. 获取动作的坐标系朝向 ---
        center_pos = self._get_center_pos()
        vec_to_goal = self.target_pos[:2] - center_pos[:2]
        angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        
        # 构建逆旋转矩阵 (Body -> World) 
        # 注意！这里是正的 angle_to_goal，和 obs 里的负号正好相反
        c, s = np.cos(angle_to_goal), np.sin(angle_to_goal)
        R_action = np.array([[c, -s], [s, c]])
        
        # --- 2. 解析网络输出的体坐标系动作 ---
        action_A_body = np.array([action[0], action[1]]) * 1.5
        action_B_body = np.array([action[2], action[3]]) * 1.5
        
        # --- 3. [关键修复] 转换到世界坐标系 ---
        vel_A_world = R_action @ action_A_body
        vel_B_world = R_action @ action_B_body

        # 拼上 Z 轴速度 (0.0)
        target_vel_A = np.array([vel_A_world[0], vel_A_world[1], 0.0])
        target_vel_B = np.array([vel_B_world[0], vel_B_world[1], 0.0])
        
        # 假设最大允许的单步速度变化量 (相当于最大加速度 * dt)
        max_delta_v = 0.5
        
        # 限制加速度：当前速度不能突变，只能在上一帧速度的基础上增减 max_delta_v
        vel_A = np.clip(target_vel_A, self.last_vels[0] - max_delta_v, self.last_vels[0] + max_delta_v)
        vel_B = np.clip(target_vel_B, self.last_vels[1] - max_delta_v, self.last_vels[1] + max_delta_v)

        
        p.resetBaseVelocity(self.robot_ids[0], linearVelocity=vel_A)
        p.resetBaseVelocity(self.robot_ids[1], linearVelocity=vel_B)

        # --- 2. Follower 控制 (外部算法接口) ---
        # 获取当前所有机器人的状态传给算法
        states = self._get_all_robot_states()
        follower_vel = self._control_follower(states)
        p.resetBaseVelocity(self.robot_ids[2], linearVelocity=follower_vel)

        # --- 3. 物理仿真 (高频率) ---
        for _ in range(20): 
            p.stepSimulation()

        # --- 4. 可视化队形连线 (画三角形) ---
        if self.render_mode:
            self._draw_formation_lines()
            time.sleep(0.01)
            
            
        # --- 5. 获取状态与计算奖励 ---
        center_pos = self._get_center_pos()
        obs = self._get_obs()
        reward, done_info = self._compute_reward(center_pos,action)
        done = done_info['done']

        # 更新上一帧速度
        for i, rid in enumerate(self.robot_ids):
            v, _ = p.getBaseVelocity(rid)
            self.last_vels[i] = np.array(v)

        return obs, reward, done, {}
    
    
    # ============================================================
    #   [核心接口] Follower 控制算法
    # ============================================================
    def _control_follower(self, states):
        """
        使用 2D 矩阵进行控制计算
        """
        # 1. 获取实时状态 (只取前 2 维: x, y)
        p0 = states[0]['pos'][:2] # Leader A pos (2D)
        v0 = states[0]['vel'][:2] # Leader A vel (2D)
        
        p1 = states[1]['pos'][:2] # Leader B pos (2D)
        v1 = states[1]['vel'][:2] # Leader B vel (2D)
        
        p2 = states[2]['pos'][:2] # Follower pos (2D)
        # v2 不是必须的，因为我们要算的是目标速度
        
        # 2. 计算 2D 相对误差
        g_02 = p0 - p2 # 2D vector
        g_12 = p1 - p2 # 2D vector
        
        k_f = 2.0 
        
        # 3. 计算控制律 (全是 2x2 矩阵 @ 2D 向量)
        # term_L0: (2x2) @ (2,) -> (2,)

        term_L1 = self.W_02 @ (k_f * g_12 + v1)
        term_L0 = self.W_12 @ (k_f * g_02 + v0)
        
        target_vel_2d = self.W_22_inv @ (term_L0 + term_L1)
        
        # 4. 转换回 PyBullet 格式 (拼上 Z轴速度 0)
        control_vel = np.array([target_vel_2d[0], target_vel_2d[1], 0.0])
        
        # 5. 安全限幅
        speed = np.linalg.norm(control_vel)
        if speed > 1.5:
            control_vel = control_vel / speed * 1.5
            
        return control_vel
    
    def _get_obs(self):
        # --- 1. 获取原始世界坐标系数据 ---
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        vel_A, _ = p.getBaseVelocity(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        vel_B, _ = p.getBaseVelocity(self.robot_ids[1])
        # states of followers 
        pos_C, _ = p.getBasePositionAndOrientation(self.robot_ids[2])
        vel_C, _ = p.getBaseVelocity(self.robot_ids[2])
        
        # 转换为 numpy 数组 (只取 x, y)
        pA = np.array(pos_A[:2])
        vA = np.array(vel_A[:2])
        pB = np.array(pos_B[:2])
        vB = np.array(vel_B[:2])
        pC = np.array(pos_C[:2])  # 提取 Follower 的 2D 坐标

        # the centroid of the whole formation
        center_pos = self._get_center_pos()[:2] # 只取 x, y
        
        # --- 2. 构建旋转矩阵 (World -> Body/Target Frame) ---
        # 以"指向目标的方向"作为 Body Frame 的 X 轴
        vec_to_goal = self.target_pos[:2] - center_pos
        angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        
        # 旋转矩阵
        c, s = np.cos(-angle_to_goal), np.sin(-angle_to_goal)
        R = np.array([[c, -s], [s, c]])
        
        # --- 3. [关键修改] 统一坐标变换 ---
        
        # A. 目标向量 (转到 Body Frame)
        rel_goal_world = self.target_pos[:2] - center_pos
        rel_goal_body = R @ rel_goal_world # 结果通常是 [dist, 0]
        
        # B. 速度向量 (也必须转！否则网络不知道是侧身走还是直走)
        vA_body = R @ vA
        vB_body = R @ vB
        
        # C. 队友相对位置 (也必须转！否则旋转后队友位置就乱了)
        rel_AB_world = pB - pA
        rel_AB_body = R @ rel_AB_world
        
        # --- 4. [核心重构] 分别计算障碍物相对于 A, B, C 的位置 ---
        # 为了避免写三次重复代码，定义一个局部辅助函数
        def get_agent_obs_features(agent_pos):
            obs_rel_list = []
            if self.enable_obstacles:
                for obs_id in self.obstacles:
                    o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                    # 计算障碍物相对于当前智能体的世界坐标
                    rel_o_world = np.array(o_pos[:2]) - agent_pos
                    # 转到编队的统一 Body Frame
                    rel_o_body = R @ rel_o_world
                    dist = np.linalg.norm(rel_o_world)
                    
                    # 过滤掉身后的障碍物 (x < -2.0)
                    if rel_o_body[0] > -2.0: 
                        obs_rel_list.append((dist, rel_o_body[0], rel_o_body[1]))
            
            # 按距离排序
            obs_rel_list.sort(key=lambda x: x[0])
            
            features = []
            for i in range(self.n_nearest_obs):
                if i < len(obs_rel_list):
                    features.extend([
                        obs_rel_list[i][1] / self.wall_length, 
                        obs_rel_list[i][2] / self.wall_width
                    ])
                else:
                    # 视野内无障碍物时，填远处安全默认值
                    features.extend([1.0, 1.0])
            return features

        # 分别获取 A, B, C 身边的障碍物特征
        obs_features_A = get_agent_obs_features(pA)
        obs_features_B = get_agent_obs_features(pB)
        obs_features_C = get_agent_obs_features(pC)

        # --- 5. 组装最终的 Observation ---
        obs = [
            # 目标 (2维)
            rel_goal_body[0] / self.wall_length, 
            rel_goal_body[1] / self.wall_width, 
            
            # 速度 (4维)
            vA_body[0], vA_body[1],
            vB_body[0], vB_body[1],
            
            # Leader 相对位置 (2维)
            rel_AB_body[0] / self.wall_length, 
            rel_AB_body[1] / self.wall_width
        ]
        
        # 拼接三个智能体的障碍物信息
        obs.extend(obs_features_A)
        obs.extend(obs_features_B)
        obs.extend(obs_features_C)

        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self, center_pos, action):
        reward = 0
        done = False
        reason = ""
        
        # --- 1. 进度奖励 ---
        dist_to_goal = np.linalg.norm(center_pos[:2] - self.target_pos[:2])
        if not hasattr(self, 'last_center_dist'): self.last_center_dist = dist_to_goal
        
        
        reward += (self.last_center_dist - dist_to_goal) * 50.0 
        self.last_center_dist = dist_to_goal

        # --- 2. 动作平滑惩罚 (Action Smoothness) ---
        # 惩罚动作的突变：||a_t - a_{t-1}||
        # 这对 Follower 能够跟上至关重要！
        # action_diff = np.linalg.norm(action - self.last_action)
        # reward -= action_diff * 0.05  # 系数根据训练情况调，0.5 是经验值

        # self.last_action = action.copy() # 更新记录

        # --- 3. 奇异性避免 (Singularity Avoidance) ---
        # 防止两个 Leader 靠得太近 (距离 < 0.3) 或太远 (距离 > 5.0)
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])

        pA_2d = np.array(pos_A[:2])
        pB_2d = np.array(pos_B[:2])
        
        # 实际的相对向量
        actual_vec = pB_2d - pA_2d

        # A. 奇异性避免 (靠得太近极度危险)
        dist_AB = np.linalg.norm(actual_vec)
        if dist_AB < 0.8: 
            reward -= 10.0 # 危险！接近穿模
            
        # 极端通信断连惩罚
        if dist_AB > 5.0:
            reward -= (dist_AB - 5.0) * 1.0

        # B. 向量误差约束 (Vector Error Constraint)
        # 获取标称构型下的期望向量 (假设 g_nom 里存的是世界坐标系下的偏移)
        # desired_vec 就是 Leader B 应该在 Leader A 的什么相对位置
        if dist_to_goal < 2.5: # 只有在接近目标时才严格要求队形，否则允许更大误差
            desired_vec = np.array(self.g_nom[1][:2]) - np.array(self.g_nom[0][:2])
            # 计算向量差异 (同时约束了距离放缩和过度旋转)
            vec_error = np.linalg.norm(actual_vec - desired_vec)
            
            # 柔性约束：允许一定程度的变形避障 (比如误差在 0.5 米以内不怎么罚)
            if vec_error > 0.5:
                # 误差越大，惩罚越重。乘 2.0 是一个经验系数
                reward -= (vec_error - 0.5) * 10.0 


        # 轻微的时间惩罚，鼓励快点到达
        reward -= 0.05 # 时间消耗
        
        # --- 4. 全局避障 (Leader要对Follower负责) ---
        # 任何一个机器人(包括Follower)撞了，都算Leader的责任
        
        # A. 撞墙检测
        limit_x = self.wall_length - 0.5 - 0.3  # wall_thicknes is 0.5m + robot_radius is 0.2m 留出0.3来
        limit_y = self.wall_width - 0.5 - 0.3
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            if abs(pos[0]) > limit_x or abs(pos[1]) > limit_y:
                reward -= 50
                done = True
                reason = "Wall_Out"
                break
             
        # B. 撞障碍物检测 & 互撞检测
        if not done:
            # 获取所有位置
            all_pos = []
            for rid in self.robot_ids:
                p_val, _ = p.getBasePositionAndOrientation(rid)
                all_pos.append(np.array(p_val[:2]))
            
            # 1. 检测互撞 (Leader-Leader, Leader-Follower)
            # 简单的两两比对
            import itertools
            for p1, p2 in itertools.combinations(all_pos, 2):
                if np.linalg.norm(p1 - p2) < 0.5:
                    reward -= 50 # 发生互撞
                    done = True
                    reason = "Crash_Team"
                    break
            
            # 2. 检测障碍物
            if self.enable_obstacles and not done:
                for i, r_pos in enumerate(all_pos):
                    for obs_id in self.obstacles:
                        o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                        d = np.linalg.norm(np.array(o_pos[:2]) - r_pos)
                        if d < 1.0: # 势场
                            reward -= (1.0 - d) * 0.5
                        if d < 0.6: # 碰撞
                            reward -= 50
                            done = True
                            reason = f"Crash_Obstacle (Bot {i})"
                            break
                    if done: break

        # --- 5. 成功与超时 ---
        if not done and dist_to_goal < 1.0:
            reward += 200
            done = True
            reason = "Success"
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"

    
        # if done: print(f"Episode End: {reason}")
        # if done and reason == "Success":
        #     print(f"Episode End: {reason}")
        
        return reward, {'done': done}
    
    def _draw_formation_lines(self):
        # 画出三角形连接线：A-B, B-F, F-A
        points = []
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            points.append(pos)
        
        # 线条颜色 [R, G, B]
        line_color = [1, 1, 0] # 黄色
        
        # 移除上一帧的线 (如果不移除，画面会变卡且全是线)
        # 注意：p.removeAllUserDebugItems() 会移除所有 debug 物体。
        # 如果你想保留其他 debug 物体，应该保存 line_id 并用 removeUserDebugItem
        # 这里为了简单，我们用 addUserDebugLine 的 replaceItemArgument (需要维护ID)
        # 或者简单的 removeAll (会把 target sphere 的字也清掉，但不影响 sphere 本身)
        
        # 更好的方法：只在第一次创建，后面使用 replace
        # 这里使用 lifetime 方案，让它自动消失
        p.addUserDebugLine(points[0], points[1], line_color, lineWidth=2, lifeTime=0.1) # A-B
        p.addUserDebugLine(points[1], points[2], line_color, lineWidth=2, lifeTime=0.1) # B-F
        p.addUserDebugLine(points[2], points[0], line_color, lineWidth=2, lifeTime=0.1) # F-A

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

    def _get_all_robot_states(self):
        """辅助函数：获取所有机器人的位置和速度，供 Follower 算法使用"""
        states = {}
        for i, rid in enumerate(self.robot_ids):
            pos, _ = p.getBasePositionAndOrientation(rid)
            vel, _ = p.getBaseVelocity(rid)
            states[i] = {'pos': np.array(pos), 'vel': np.array(vel)}
        return states

    def _get_center_pos(self):
        """计算双车中心点"""
        pos_list = []
        for i, robot_id in enumerate(self.robot_ids):
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            pos_list.append(np.array(pos))
        center = np.mean(pos_list, axis=0)
        return center

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