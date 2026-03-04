import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from formation_graph import FormationGraph
from formation_utils import get_nominal_configuration

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
        self.n_followers = 7
        self.num_robots = self.n_leaders + self.n_followers 
        self.robot_ids = []
        self.robot_radius = 0.25
        
        # --- [整合核心 1] 初始化标称构型和权重 ---
        self.g_nom = get_nominal_configuration()

        # 2. 构建全局大矩阵并进行 2D 降维分块
        self._init_global_matrices()

        # --- 动作空间 (4维) ---
        # [LeaderA_Vx, LeaderA_Vy, LeaderB_Vx, LeaderB_Vy]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.n_nearest_obs = 10 # 最近的6个障碍物 + 四面墙 = 10 个观测点
        # --- 观测空间 (44维) ---
        # 1. 目标相对向量 (2)
        # 2. A速度(2) + B速度(2)
        # 3. AB相对位置(2)
        # 4. A的最近6个障碍物 (6 * 2 = 12)
        # 5. B的最近6个障碍物 (6 * 2 = 12)
        # 6. 两个leader的位置(4)
        # 2 + 4 + 2 + 12 + 12 = 44 维
        self.n_lidar_rays = 9
        self.lidar_range = 3.0

        obs_dim = 2 + 4 + 2 + (self.n_nearest_obs * 2 * 2) + 4 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # print(f"Observation space dimension: {obs_dim}")
        
        self.target_pos = np.array([8.5, 0, 0])
        self.obstacles = []
        self.max_steps = 1500
        self.wall_length = 15.0
        self.wall_width = 10.0
        self.enable_obstacles = enable_obstacles
        self.last_center_dist = None
        # 用于可视化的线段ID
        self.debug_line_ids = []

    def _init_global_matrices(self):
        """
        构建 9 个智能体的全局拉普拉斯矩阵，并剔除角度维度，进行分块
        """
        # 1. 构建 27x27 的矩阵 M
        fg = FormationGraph(self.g_nom)
        M_3d = fg.build_laplacian(theta=0)
        
        # 2. 提取 2D 维度 (只保留 x 和 y 的行与列)
        # 生成保留的索引: 0,1, 3,4, 6,7, ..., 24,25
        idx_2d = [i * 3 + j for i in range(9) for j in range(2)]
        
        # 使用 numpy 的高级索引抽取 18x18 矩阵
        self.M_2d = M_3d[np.ix_(idx_2d, idx_2d)]
        
        # 3. 提取 2D 标称构型 (转成 1D 数组方便矩阵相乘)
        g_nom_2d_list = [g[:2] for g in self.g_nom]
        self.g_nom_flat = np.concatenate(g_nom_2d_list) # shape: (18,)
        
        # 4. 矩阵分块 (Leaders: 前 4 行/列; Followers: 后 14 行/列)
        split_idx = self.n_leaders * 2 # 2 个 leader * 2D = 4
        
        self.M_ff = self.M_2d[split_idx:, split_idx:] # (14, 14)
        self.M_fl = self.M_2d[split_idx:, :split_idx] # (14, 4)
        
        # 5. 提前求逆 (极大提升仿真速度)
        self.M_ff_inv = np.linalg.inv(self.M_ff)

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # --- 创建围墙 ---
        self._create_walls()


        # --- 创建两个 Leader ---
        self.robot_ids = []
        spawned_positions = []
        
        # 1. 随机生成一个队伍的起跑中心点
        start_center = np.array([
            np.random.uniform(-self.wall_length + 5.0, -self.wall_length + 5.2),
            np.random.uniform(-self.wall_width + 6.0, -self.wall_width + 6.2)
        ])
        
        # 2. 带有剧烈噪声的标称排布
        for i in range(self.num_robots):
            while True:
                # 获取标称偏移，并加入大幅度的随机噪声 (-0.8m 到 +0.8m)
                # 这样阵型一开始是极度扭曲、松散且不规则的，满足了控制论文"初始位置随机"的要求
                noise_x = np.random.uniform(-0.8, 0.8)
                noise_y = np.random.uniform(-0.8, 0.8)
                
                rx = start_center[0] + self.g_nom[i][0] + noise_x
                ry = start_center[1] + self.g_nom[i][1] + noise_y
                
                # 查重，防止直接重叠穿模
                is_safe = True
                for p_exist in spawned_positions:
                    if np.linalg.norm(np.array([rx, ry]) - np.array(p_exist)) < 0.6:
                        is_safe = False
                        break
                
                if is_safe:
                    spawned_positions.append([rx, ry])
                    spawn_pos = [rx, ry, 0.25]
                    color = [1, 0, 0, 1] if i < 2 else [0, 1, 0, 1]
                    rid = self._create_box_robot(spawn_pos, color)
                    self.robot_ids.append(rid)
                    break


        # --- 随机生成目标点 ---
        tx = np.random.uniform(self.wall_length-5.0, self.wall_length-4.0)
        ty = np.random.uniform(-self.wall_width+4.0, self.wall_width-4.0)
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

        obs = self._get_obs()
        
        return obs, {}

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
        # 获取速度列表
        follower_vels = self._control_follower(states)
        # 遍历 7 个跟随者 (ID 从 2 到 8)
        for idx, f_vel in enumerate(follower_vels):
            robot_id = self.robot_ids[idx + 2] 
            p.resetBaseVelocity(robot_id, linearVelocity=f_vel)

        # --- 3. 物理仿真 (高频率) ---
        for _ in range(24): 
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

        # 如果是因为超过最大步数结束的，属于 truncated
        # 如果是因为撞墙、撞障碍物或到达终点结束的，属于 terminated
        if done and self.current_step >= self.max_steps:
            terminated = False
            truncated = True
        else:
            terminated = done
            truncated = False

        return obs, reward, terminated, truncated, {}
    
    
    # ============================================================
    #   [核心接口] Follower 控制算法
    # ============================================================
    def _control_follower(self, states):
        """
        使用全局矩阵一次性计算 7 个跟随者的目标速度
        """
        # 1. 提取实时的全局状态 (只取 x, y)
        # p_all shape: (18,), v_l shape: (4,)
        p_all = []
        for i in range(9):
            p_all.extend(states[i]['pos'][:2])
        p_flat = np.array(p_all)
        
        v_l = []
        for i in range(2):
            v_l.extend(states[i]['vel'][:2])
        v_l_flat = np.array(v_l)
        
        k_f = 2.0 # 跟随者增益
        
        # 2. 核心数学公式：v_f = M_ff_inv @ ( -k_f * (M @ p)_f - M_fl @ v_l )
        # 计算全局误差
        error_global = -k_f * (self.M_2d @ p_flat)
        
        # 截取属于 Followers 的部分 (后 14 维)
        error_f = error_global[4:]
        
        # 计算 Leader 运动带来的耦合项
        coupling_f = -self.M_fl @ v_l_flat
        
        # 求解 7 个 Follower 的 2D 速度 (shape: 14,)
        v_f_flat = self.M_ff_inv @ (error_f + coupling_f)
        
        # 3. 解析回各个机器人的 3D 速度并安全限幅
        follower_vels = []
        for i in range(7):
            vx = v_f_flat[i*2]
            vy = v_f_flat[i*2 + 1]
            vel_3d = np.array([vx, vy, 0.0])
            
            # 限幅
            speed = np.linalg.norm(vel_3d)
            if speed > 1.5:
                vel_3d = vel_3d / speed * 1.5
                
            follower_vels.append(vel_3d)
            
        return follower_vels
    
    def _get_obs(self):
        # --- 1. 获取原始世界坐标系数据 ---
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        vel_A, _ = p.getBaseVelocity(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        vel_B, _ = p.getBaseVelocity(self.robot_ids[1])
        # states of followers 
        # pos_C, _ = p.getBasePositionAndOrientation(self.robot_ids[2])
        # vel_C, _ = p.getBaseVelocity(self.robot_ids[2])
        
        # 转换为 numpy 数组 (只取 x, y)
        pA = np.array(pos_A[:2])
        vA = np.array(vel_A[:2])
        pB = np.array(pos_B[:2])
        vB = np.array(vel_B[:2])
        # pC = np.array(pos_C[:2])  # 提取 Follower 的 2D 坐标

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

        # D. Leader A 和 B 的位置 (转到 Body Frame)
        pos_A_body = R @ pA
        pos_B_body = R @ pB 
        
        # --- 4. [核心重构] 分别计算障碍物相对于 A, B, C 的位置 ---
        # 为了避免写三次重复代码，定义一个局部辅助函数
        def get_agent_obs_features(agent_pos):
            obs_rel_list = []
            rx, ry = agent_pos[0], agent_pos[1]
            
            # 计算到四面墙的最近点向量 (世界坐标系)
            # 格式: (距离, 世界向量x, 世界向量y)
            walls_info = [
                # 右墙 (X正向)
                (self.wall_length - 0.5 - rx, np.array([self.wall_length - 0.5 - rx, 0])),
                # 左墙 (X负向)
                (rx - (-self.wall_length + 0.5), np.array([- (rx - (-self.wall_length + 0.5)), 0])),
                # 上墙 (Y正向)
                (self.wall_width - 0.5 - ry, np.array([0, self.wall_width - 0.5 - ry])),
                # 下墙 (Y负向)
                (ry - (-self.wall_width + 0.5), np.array([0, - (ry - (-self.wall_width + 0.5))]))
            ]

            for dist, rel_wall_world in walls_info:
                # 旋转到 Body Frame
                rel_wall_body = R @ rel_wall_world
                # 过滤掉身后的墙壁 (比如 x < -2.0)，只关注前方和侧面的威胁
                if rel_wall_body[0] > -2.0:
                    obs_rel_list.append((dist, rel_wall_body[0], rel_wall_body[1]))

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

        # 分别获取 A, B 身边的障碍物特征
        obs_features_A = get_agent_obs_features(pA)
        obs_features_B = get_agent_obs_features(pB)

        # --- 5. 组装最终的 Observation ---
        obs = [
            # 目标 (2维)
            rel_goal_body[0] / self.wall_length, 
            rel_goal_body[1] / self.wall_width, 

            # 位置 （4维，转到 Body Frame）
            pos_A_body[0] / self.wall_length,
            pos_A_body[1] / self.wall_width,
            pos_B_body[0] / self.wall_length,
            pos_B_body[1] / self.wall_width,
            
            # 速度 (4维)
            vA_body[0], vA_body[1],
            vB_body[0], vB_body[1],
            
            # Leader 相对位置 (2维)
            rel_AB_body[0] / self.wall_length, 
            rel_AB_body[1] / self.wall_width
        ]
        
        # 拼接智能体的障碍物信息
        obs.extend(obs_features_A)
        obs.extend(obs_features_B)

        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self, center_pos, action):
        reward = 0
        done = False
        reason = ""
        
        # --- 1. 进度奖励 ---
        dist_to_goal = np.linalg.norm(center_pos[:2] - self.target_pos[:2])
        if not hasattr(self, 'last_center_dist'): self.last_center_dist = dist_to_goal
        
        
        reward += (self.last_center_dist - dist_to_goal) * 20.0
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
            reward -= 2.0 # 危险！接近穿模

        # 极端通信断连惩罚
        if dist_AB > 6.0:
            reward -= (dist_AB - 6.0) * 0.2
            

        # B. 向量误差约束 (Vector Error Constraint)
        # 获取标称构型下的期望向量 (假设 g_nom 里存的是世界坐标系下的偏移)
        # desired_vec 就是 Leader B 应该在 Leader A 的什么相对位置
        nom_vec = np.array(self.g_nom[1][:2]) - np.array(self.g_nom[0][:2]) # 原始 [6, 6]
        # 使用等比例缩放，避免队伍长宽比严重失调导致碰撞
        if 2.0 < dist_to_goal < 5.0:
            # 第一阶段收缩：缩小到 60% (目标长度约 5.0)
            scale = 0.8
            desired_vec = nom_vec * scale
            vec_error = np.linalg.norm(actual_vec - desired_vec)
            if vec_error > 0.5:
                reward -= (vec_error - 0.5) * 1.0
                
        elif dist_to_goal <= 2.0:
            # 第二阶段极限收缩：缩小到 35% (目标长度约 3.0，挤在一起过终点)
            scale = 0.6
            desired_vec = nom_vec * scale
            vec_error = np.linalg.norm(actual_vec - desired_vec)
            if vec_error > 0.5:
                reward -= (vec_error - 0.5) * 1.5  # 离终点越近，阵型要求越严
        
        reward -= 0.01  # 每步固定的时间惩罚，鼓励更快完成任务

        # --- 4. 全局避障 (Leader要对Follower负责) ---
        # 任何一个机器人(包括Follower)撞了，都算Leader的责任
        
        # A. 撞墙检测
        limit_x = self.wall_length - 0.5 - 0.3  # wall_thicknes is 0.5m + robot_radius is 0.2m 留出0.3来
        limit_y = self.wall_width - 0.5 - 0.3
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            if abs(pos[0]) > limit_x or abs(pos[1]) > limit_y:
                reward -= 10
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
                if np.linalg.norm(p1 - p2) < 0.6:
                    reward -= 10 # 发生互撞
                    done = True
                    reason = "Crash_Team"
                    break
            
            # ==========================================
            # [核心新增] 2. 领航者与跟随者之间的“倒车雷达”势场
            # 逼迫 Leader 主动与 Follower 保持安全距离
            # ==========================================
            if not done:
                for l_idx in [0, 1]:  # 遍历两个 Leader (索引 0 和 1)
                    p_leader = all_pos[l_idx]
                    for f_idx in range(2, len(all_pos)): # 遍历所有 Follower (索引 2 到最后)
                        p_follower = all_pos[f_idx]
                        dist_lf = np.linalg.norm(p_leader - p_follower)
                        
                        safe_margin = 0.9 # 设定的警戒距离
                        if dist_lf < safe_margin:
                            # 距离越近，扣分越狠 (类似障碍物势场)
                            # 系数设为 2.0，不要太大，免得盖过找目标的奖励
                            team_repulsion = 2.0 / (dist_lf + 0.1) - 2.0 / (safe_margin + 0.1)
                            reward -= team_repulsion
            # ==========================================

            # 2. 检测障碍物
            if self.enable_obstacles and not done:
                for i, r_pos in enumerate(all_pos):
                    for obs_id in self.obstacles:
                        o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                        d = np.linalg.norm(np.array(o_pos[:2]) - r_pos)
                        if 0.6 <= d < 1.2: # 势场
                            repulsion = 3.0 / (d + 0.1) - 3.0 / (1.2 + 0.1) 
                            reward -= repulsion
                        if d < 0.6: # 碰撞
                            reward -= 10
                            done = True
                            reason = f"Crash_Obstacle (Bot {i})"
                            break
                    if done: break

        # --- 5. 成功与超时 ---
        if not done and dist_to_goal < 1.0:
            reward += 40
            done = True
            reason = "Success"
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"

    
        # if done: print(f"Episode End: {reason}")
        if done and reason == "Success":
            print(f"Episode End: {reason}")
        
        return reward, {'done': done}
    
    def _draw_formation_lines(self):
        """
        可视化 9 智能体编队的拓扑连线
        """
        # 1. 获取所有 9 个机器人的当前位置
        points = []
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            points.append(pos)
        
        # 线条颜色 [R, G, B] - 这里用黄色
        line_color = [1, 1, 0] 
        
        # 2. 定义 9 智能体的图拓扑边 (根据 formation_graph.py 中的 DEP 路径)
        # 注意：索引 0 是 Leader A(g1)，1 是 Leader B(g2)
        edges = [
            # Path 1 (中间主干路)
            (0, 4), (4, 3), (3, 2), (2, 1), 
            # Path 2 (上方支路)
            (0, 6), (6, 5), (5, 4),         
            # Path 3 (下方支路)
            (2, 7), (7, 8), (8, 1)          
        ]
        
        # 3. 遍历所有的拓扑边，并在 PyBullet 中画线
        for u, v in edges:
            # 加入越界保护：确保在 robot_ids 还没完全生成时不会报错
            if u < len(points) and v < len(points):
                p.addUserDebugLine(
                    lineFromXYZ=points[u], 
                    lineToXYZ=points[v], 
                    lineColorRGB=line_color, 
                    lineWidth=2, 
                    lifeTime=0.1 # 0.1秒后自动消失，避免画面被线段填满
                )

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
        """计算编队中心点"""
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