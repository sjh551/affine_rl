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
        # --- 观测空间 (26维) ---
        # 1. 编队中心相对目标的向量 (2)
        # 2. A 速度 (2)
        # 3. AB之间的距离(2) + B 速度 (2)
        # 4. 最近的5个障碍物相对编队中心的位置 (5*2=10)
        self.n_lidar_rays = 9
        self.lidar_range = 3.0

        obs_dim = 2 + 2 + 2 + 2 + (self.n_nearest_obs * 2)
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
        
        # --- [关键修改] 动作平滑处理 ---
        # 混合系数 alpha: 0.2 表示 20% 新动作，80% 旧动作 (惯性很大，轨迹会非常平滑)
        # 训练初期可以设大一点(0.5)，后期为了丝滑可以设小。建议 0.3 ~ 0.5
        alpha = 0.4 
        smooth_action = alpha * action + (1 - alpha) * self.last_action
        
        # 解析平滑后的动作
        vel_A = np.array([smooth_action[0], smooth_action[1], 0]) * 1.5 
        vel_B = np.array([smooth_action[2], smooth_action[3], 0]) * 1.5

        # 更新 last_action (注意：要更新为 smooth_action 还是 raw action? 
        # 通常更新为 raw action 记录用于计算 diff 惩罚，但为了物理平滑，这里我们控制的是电机)
        # 为了计算 reward 里的 action_diff，我们还是保留原始 action 的记录
        # 但物理上执行 smooth_action
        
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
    
    def _get_center_pos(self):
        """计算双车中心点"""
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        center = (np.array(pos_A) + np.array(pos_B)) / 2.0
        return center
    
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
    
    def _get_all_robot_states(self):
        """辅助函数：获取所有机器人的位置和速度，供 Follower 算法使用"""
        states = {}
        for i, rid in enumerate(self.robot_ids):
            pos, _ = p.getBasePositionAndOrientation(rid)
            vel, _ = p.getBaseVelocity(rid)
            states[i] = {'pos': np.array(pos), 'vel': np.array(vel)}
        return states

    def _generate_obstacles(self):
        self.obstacles = []
        config_id = np.random.randint(0, 1)
        
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
        # --- 1. 获取原始世界坐标系数据 ---
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        vel_A, _ = p.getBaseVelocity(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        vel_B, _ = p.getBaseVelocity(self.robot_ids[1])
        
        # 转换为 numpy 数组 (只取 x, y)
        pA = np.array(pos_A[:2])
        vA = np.array(vel_A[:2])
        pB = np.array(pos_B[:2])
        vB = np.array(vel_B[:2])
        
        center_pos = (pA + pB) / 2.0
        
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
        
        # D. 障碍物信息 (保持你写的逻辑，但加上坐标转换)
        obs_rel_list = []
        if self.enable_obstacles:
            for obs_id in self.obstacles:
                o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                rel_o_world = np.array(o_pos[:2]) - center_pos
                
                # 转到 Body Frame
                rel_o_body = R @ rel_o_world
                
                dist = np.linalg.norm(rel_o_world)
                # 过滤掉身后的障碍物 (x < -2.0)
                if rel_o_body[0] > -2.0: 
                    # 记录：距离, Body系x, Body系y
                    obs_rel_list.append((dist, rel_o_body[0], rel_o_body[1]))
        
        # 按距离排序
        obs_rel_list.sort(key=lambda x: x[0])

        # --- 4. 组装 Observation ---
        obs = [
            # 目标 (归一化)
            rel_goal_body[0] / self.wall_length, 
            rel_goal_body[1] / self.wall_width, 
            
            # 速度 (使用转换后的 Body 系速度)
            vA_body[0], vA_body[1],
            vB_body[0], vB_body[1],
            
            # 队友位置 (使用转换后的 Body 系偏移)
            # 注意：队友距离一般很近(1.5m)，除以 wall_length(10m) 会变得太小(0.15)
            # 建议除以一个更小的常数，比如 5.0，让特征更明显
            rel_AB_body[0] / 5.0, 
            rel_AB_body[1] / 5.0 
        ]
        
        # 填充障碍物
        for i in range(self.n_nearest_obs):
            if i < len(obs_rel_list):
                obs.extend([
                    obs_rel_list[i][1] / self.wall_length, 
                    obs_rel_list[i][2] / self.wall_width
                ])
            else:
                # 无障碍物填远处的默认值
                obs.extend([1.0, 1.0])

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

        # --- 2. [关键] 动作平滑惩罚 (Action Smoothness) ---
        # 惩罚动作的突变：||a_t - a_{t-1}||
        # 这对 Follower 能够跟上至关重要！
        action_diff = np.linalg.norm(action - self.last_action)
        reward -= action_diff * 0.05  # 系数根据训练情况调，0.5 是经验值
        # [新增] 存活奖励 (Living Reward)
        # 抵消掉时间惩罚的一部分，鼓励它在困难环境下活着
        reward += 0.05

        self.last_action = action.copy() # 更新记录

        # --- 3. [关键] 奇异性避免 (Singularity Avoidance) ---
        # 防止两个 Leader 靠得太近 (距离 < 0.3) 或太远 (距离 > 5.0)
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        dist_AB = np.linalg.norm(np.array(pos_A[:2]) - np.array(pos_B[:2]))
        
        # 窄缝通过时允许靠得很近，但不能重合(奇异)
        if dist_AB < 0.3: 
            reward -= 10.0 # 危险！接近奇异点
            # 甚至可以直接 done，因为物理上可能已经穿模了
            # done = True 
            # info['done_reason'] = 'singularity'
        elif dist_AB > 5.0:
            reward -= (dist_AB - 5.0) * 1.0 # 通信距离限制


        # 轻微的时间惩罚，鼓励快点到达
        reward -= 0.05 # 时间消耗

        # --- 2. 平滑性惩罚 (所有机器人) 后面再加入 ---
        # total_smooth_penalty = 0
        # for i, rid in enumerate(self.robot_ids):
        #     curr_vel, _ = p.getBaseVelocity(rid)
        #     diff = np.linalg.norm(np.array(curr_vel)[:2] - self.last_vels[i][:2])
        #     total_smooth_penalty += diff
        # reward -= total_smooth_penalty * 0.1


        # --- 3. [关键修改] 队形柔性约束 ---
        # 不再强制固定距离，允许放缩。
        # 我们只给一个“最大/最小距离”的松弛约束，防止太离谱。
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        dist_AB = np.linalg.norm(np.array(pos_A[:2]) - np.array(pos_B[:2]))
        
        # 假设我们希望他们在 0.8米 到 4.0米 之间自由调整
        # 只有超出这个范围才惩罚
        if dist_AB < 0.8:
            reward -= (0.8 - dist_AB) * 2.0 # 太近了，惩罚重一点
        elif dist_AB > 4.0:
            reward -= (dist_AB - 4.0) * 1.0 # 太远了，稍微惩罚
        
        # --- 3. [关键修改] 全局避障 (Leader要对Follower负责) ---
        # 任何一个机器人(包括Follower)撞了，都算Leader的责任
        
        # A. 撞墙检测
        limit_x = self.wall_length - 0.5 - 0.3  # wall_thicknes is 0.5m + robot_radius is 0.2m 留出0.3来
        limit_y = self.wall_width - 0.5 - 0.3
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            if abs(pos[0]) > limit_x or abs(pos[1]) > limit_y:
                reward -= 20
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
                    reward -= 20 # 发生互撞
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
                        if d < 0.55: # 碰撞
                            reward -= 20
                            done = True
                            reason = f"Crash_Obstacle (Bot {i})"
                            break
                    if done: break

        # --- 4. 成功与超时 ---
        if not done and dist_to_goal < 1.0:
            reward += 200
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