import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import pyrvo as rvo2  # 引入 ORCA 核心库
import sys
'''
每个小车独立导航，没有一致性协议
'''

CURRENNT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENNT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.formation_utils import get_square_configuration, get_square_topology

RANGER_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "ranger_mini_v3", "urdf", "ranger_mini.urdf").replace("\\", "/")

class FormationPyBulletEnv(gym.Env):
    """
    纯正的 ORCA 多智能体导航基准环境 (Baseline)
    无需强化学习控制，所有智能体由 pyrvo 独立进行避障和速度解算。
    """
    def __init__(self, render_mode=False, enable_obstacles=True):
        super(FormationPyBulletEnv, self).__init__()
        
        # --- 1. PyBullet 初始化配置 ---
        self.render_mode = render_mode
        self.enable_obstacles = enable_obstacles
        
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 2. 基础参数 ---
        self.n_leaders = 2
        self.n_followers = 2
        self.num_robots = self.n_leaders + self.n_followers 

        self.robot_ids = []
        self.max_vel = 1.0      # 最大线速度
        self.safe_margin = 0.6  # 机器人间的安全距离
        self.max_steps = 1500
        self.wall_length = 20.0
        self.wall_width = 10.0
        
        # 定义 9 个智能体在终点附近的阵型偏移（防止所有车挤在同一个绝对坐标上）
        # 这里用一个简单的 3x3 方阵偏移
        self.g_nom, self.edges = get_square_configuration(), get_square_topology()

        self.wheel_indices = [3, 1, 5, 7] 
        self.steer_indices = [2, 0, 4, 6] 

        # --- 3. 占位用的 Gym 空间 ---
        # 因为评估脚本可能会读取这些属性，所以保留最简单的占位符
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        self.obstacles = []
        self._create_walls()
        
        # 随机生成目标中心点
        tx = np.random.uniform(self.wall_length - 4.5, self.wall_length - 4.0)
        ty = np.random.uniform(-self.wall_width + 6.0, self.wall_width - 6.0)
        self.target_pos = np.array([tx, ty, 0])
        self._create_visual_sphere(self.target_pos, [1, 0, 0, 0.8])

        # 随机生成起点中心
        self.start_center = np.array([
            np.random.uniform(-self.wall_length + 6.0, -self.wall_length + 6.2),
            np.random.uniform(-2.0, 2.0)
        ])

        # --- 4. 生成小车 ---
        self.robot_ids = []
        for i in range(self.num_robots):
            # 在起点附近加上偏移量
            rx = self.start_center[0] + self.g_nom[i][0]
            ry = self.start_center[1] + self.g_nom[i][1]
            
            spawn_pos = [rx, ry, 0.15]
            spawn_quat = p.getQuaternionFromEuler([0, 0, 0])

            rid = p.loadURDF(RANGER_URDF_PATH, basePosition=spawn_pos, baseOrientation=spawn_quat)
            
            # 统一改个颜色 (区分 Baseline)
            cyan = [0.2, 0.8, 0.8, 1.0]
            p.changeVisualShape(objectUniqueId=rid, linkIndex=-1, rgbaColor=cyan)
            
            for joint_idx in range(p.getNumJoints(rid)):
                if joint_idx in self.steer_indices:
                    p.changeDynamics(rid, joint_idx, lateralFriction=0.1)
                    p.setJointMotorControl2(rid, joint_idx, p.POSITION_CONTROL, targetPosition=0)
                elif joint_idx in self.wheel_indices:
                    p.changeDynamics(rid, joint_idx, lateralFriction=0.5, rollingFriction=0.02)
                    p.setJointMotorControl2(rid, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

            p.changeDynamics(rid, -1, lateralFriction=0.5, rollingFriction=0.01)
            self.robot_ids.append(rid)

        if self.enable_obstacles:
            self._generate_obstacles()

        # 让物理引擎稳定一下
        for _ in range(50):
            p.stepSimulation()

        # --- 5. 初始化 ORCA (RVO2) 引擎 ---
        # 参数: dt, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
        self.rvo_sim = rvo2.RVOSimulator(0.1, 6.0, 10, 3.0, 2.0, self.safe_margin / 2.0, self.max_vel)
        self.rvo_agent_ids = []
        for i in range(self.num_robots):
            pos, _ = p.getBasePositionAndOrientation(self.robot_ids[i])
            self.rvo_agent_ids.append(self.rvo_sim.add_agent((pos[0], pos[1])))

        self.current_step = 0
        return np.zeros(1), {}  # 返回 Dummy 观测

    def step(self, action=None):
        """
        Baseline 的 step 忽略外部传入的 action，完全由 ORCA 独立接管
        """
        self.current_step += 1
        
        all_pos = []
        all_vel = []
        all_yaw = []

        # 1. 采集所有机器人的当前状态并同步给 ORCA
        for i, rid in enumerate(self.robot_ids):
            pos, quat = p.getBasePositionAndOrientation(rid)
            vel, _ = p.getBaseVelocity(rid)
            yaw = p.getEulerFromQuaternion(quat)[2]
            
            all_pos.append(np.array(pos[:2]))
            all_vel.append(np.array(vel[:2]))
            all_yaw.append(yaw)

            self.rvo_sim.set_agent_position(self.rvo_agent_ids[i], (pos[0], pos[1]))
            self.rvo_sim.set_agent_velocity(self.rvo_agent_ids[i], (vel[0], vel[1]))

        # 2. 计算每个小车的期望速度 (目标吸引力 + 静态环境排斥)
        for i in range(self.num_robots):
            # 每个小车的专属目标点
            goal_pos = self.target_pos[:2] + self.g_nom[i][:2]
            
            # 目标引力
            vec_to_goal = goal_pos - all_pos[i]
            dist_to_goal = np.linalg.norm(vec_to_goal)
            
            if dist_to_goal > 0.1:
                v_pref = (vec_to_goal / dist_to_goal) * self.max_vel
                # 接近目标时减速
                if dist_to_goal < 1.0:
                    v_pref *= dist_to_goal
            else:
                v_pref = np.zeros(2)

            # 加入环境静态障碍物的斥力 (防撞墙/柱子)
            _, APF_vel = self._get_obstacle_repulsion(all_pos[i])
            v_pref_combined = v_pref + APF_vel

            self.rvo_sim.set_agent_pref_velocity(self.rvo_agent_ids[i], tuple(v_pref_combined))

        # 3. 运行 ORCA 避障核心算法
        self.rvo_sim.do_step()

        # 4. 获取安全速度并驱动物理引擎
        for i, rid in enumerate(self.robot_ids):
            safe_v_world = self.rvo_sim.get_agent_velocity(self.rvo_agent_ids[i])
            yaw = all_yaw[i]

            vx_w = safe_v_world.x
            vy_w = safe_v_world.y
            
            # 世界坐标系转车体坐标系
            vx_body = vx_w * np.cos(yaw) + vy_w * np.sin(yaw)
            vy_body = -vx_w * np.sin(yaw) + vy_w * np.cos(yaw)
            
            # 让车头倾向于朝向运动方向
            w = 0.0
            speed = np.hypot(vx_w, vy_w)
            if speed > 0.1:
                target_yaw = np.arctan2(vy_w, vx_w)
                yaw_diff = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
                w = np.clip(yaw_diff * 2.0, -1.0, 1.0) # P控制器转车头

            self._drive_robot(rid, vx_body, vy_body, w)

        # 5. 步进物理引擎
        for _ in range(24):
            p.stepSimulation()
            # if self.render_mode:
            #     time.sleep(0.0000001)

        # 6. 状态检查 (是否碰撞 / 是否到达)
        done = False
        reason = "Unknown"
        stats = self._check_collisions(all_pos)
        
        # 检查是否全部到达
        all_arrived = True
        for i in range(self.num_robots):
            goal_pos = self.target_pos[:2] + self.g_nom[i][:2]
            if np.linalg.norm(goal_pos - all_pos[i]) > 0.5:
                all_arrived = False
                break
                
        if all_arrived:
            done = True
            reason = "Success"
        elif stats["crash"]:
            done = True
            reason = stats["reason"]
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"

        info = {'reason': reason}
        return np.zeros(1), 0.0, done, False, info

    def _drive_robot(self, robot_id, v_x, v_y, w):
        R = 0.09  
        wheel_positions = [(0.25, 0.19), (0.25, -0.19), (-0.25, 0.19), (-0.25, -0.19)]
        target_steering_angles, target_wheel_speeds = [], []
        current_joint_states = p.getJointStates(robot_id, self.steer_indices)
        
        for i, (wx, wy) in enumerate(wheel_positions):
            wheel_vx = v_x - w * wy
            wheel_vy = v_y + w * wx
            speed_2d = np.hypot(wheel_vx, wheel_vy)
            
            if speed_2d < 2e-2:
                target_steering_angles.append(current_joint_states[i][0]) 
                target_wheel_speeds.append(0.0)
                continue
                
            ideal_angle = np.arctan2(wheel_vy, wheel_vx)
            cmd_angle = -ideal_angle 
            ideal_speed = speed_2d / R
            current_angle = current_joint_states[i][0]
            
            angle1, speed1 = cmd_angle, ideal_speed
            angle2 = angle1 - np.pi if angle1 > 0 else angle1 + np.pi
            speed2 = -ideal_speed
            
            valid_configs = []
            if -2.1 <= angle1 <= 2.1: valid_configs.append((angle1, speed1))
            if -2.1 <= angle2 <= 2.1: valid_configs.append((angle2, speed2))
                
            if len(valid_configs) > 0:
                best_angle, best_speed = min(valid_configs, key=lambda x: abs(x[0] - current_angle))
            else:
                best_angle, best_speed = current_angle, 0.0
                
            angle_diff = abs(best_angle - current_angle)
            final_speed = best_speed * max(0, np.cos(angle_diff))
            
            target_steering_angles.append(best_angle)
            target_wheel_speeds.append(final_speed)
            
        p.setJointMotorControlArray(
            robot_id, self.steer_indices, p.POSITION_CONTROL,
            targetPositions=target_steering_angles, forces=[80.0] * 4
        )
        p.setJointMotorControlArray(
            robot_id, self.wheel_indices, p.VELOCITY_CONTROL,
            targetVelocities=target_wheel_speeds, forces=[20.0] * 4
        )

    def _get_obstacle_repulsion(self, pos_2d, safe_margin=1.0, num_rays=90):
        """保留 APF 用于处理复杂的静态墙壁和柱子"""
        ray_from, ray_to, ray_dirs = [], [], []
        z_height = 0.23
        
        for i in range(num_rays):
            angle = i * (2 * np.pi / num_rays)
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            ray_dirs.append(dir_vec)
            ray_from.append([pos_2d[0], pos_2d[1], z_height])
            ray_to.append([pos_2d[0] + safe_margin * dir_vec[0], pos_2d[1] + safe_margin * dir_vec[1], z_height])

        results = p.rayTestBatch(ray_from, ray_to)
        min_dist = safe_margin
        repulsion_vec = np.zeros(2) 
        k_repulsion = 2.0 
        
        for i, res in enumerate(results):
            if res[2] < 1.0 and res[0] not in self.robot_ids:
                dist = res[2] * safe_margin
                if dist < min_dist: min_dist = dist
                force = k_repulsion * (safe_margin - dist)**2 
                repulsion_vec -= force * ray_dirs[i] 
                    
        return min_dist, repulsion_vec

    def _check_collisions(self, all_pos):
        """简单的碰撞检测逻辑"""
        limit_x, limit_y = self.wall_length - 1.0, self.wall_width - 1.0
        
        # 撞墙
        for i, pos in enumerate(all_pos):
            if abs(pos[0]) > limit_x or abs(pos[1]) > limit_y:
                return {"crash": True, "reason": "Wall_Out"}
                
        # 团队互撞
        import itertools
        for p1, p2 in itertools.combinations(all_pos, 2):
            if np.linalg.norm(p1 - p2) < 0.35: # 物理硬碰撞距离
                return {"crash": True, "reason": "Crash_Team"}
                
        # 撞静物
        if self.enable_obstacles:
            for rid in self.robot_ids:
                for obs_id in self.obstacles:
                    pts = p.getClosestPoints(bodyA=rid, bodyB=obs_id, distance=0.05)
                    if len(pts) > 0:
                        return {"crash": True, "reason": "Crash_Obstacle"}
                        
        return {"crash": False}

    def _create_walls(self):
        """保留原样"""
        self.wall_ids = []
        walls_config = [
            ([self.wall_length, 0, 0.5], [0.5, self.wall_width, 1.0]), 
            ([-self.wall_length, 0, 0.5], [0.5, self.wall_width, 1.0]), 
            ([0, self.wall_width, 0.5], [self.wall_length, 0.5, 1.0]), 
            ([0, -self.wall_width, 0.5], [self.wall_length, 0.5, 1.0]) 
        ]
        for pos, extents in walls_config:
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.8, 0.8, 0.8, 1])
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            self.wall_ids.append(p.createMultiBody(0, c, v, basePosition=pos))

    def _generate_obstacles(self):
        self.obstacles = []
        config_id =  np.random.randint(0, 5) 
        
        v_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.5, rgbaColor=[0.2, 0.2, 0.2, 1])
        c_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.5)

        pos_list = []
        
        mid_x_min, mid_x_max = -self.wall_length / 3, self.wall_length / 3
        mid_y_range = self.wall_width - 1.0

        if config_id == 0:  # 均匀4排
            for x in [-4, -1.33333333, 1.33333333, 4]:
                pos_list.append([x, 0, 0.5])
                pos_list.append([x, 0.5 * self.wall_width, 0.5])
                pos_list.append([x, -0.5 * self.wall_width, 0.5])


        elif config_id == 1:
            # 目标：在 x=0 处建一堵墙，支持多开几个门降低探索难度
            
            # --- 在这里自由定义门的位置和大小 ---
            doors_y = [-5.0, 0.0, 5.0]  # 定义三个门的中心 Y 坐标
            gap_half_width = 0.8        # 每个门留 1.6 米的宽缝
            
            wall_thickness = 1.0
            wall_height = 1.5
            wall_x = 0.0                # 墙在 X 轴的位置
            
            # 墙体的起始 Y 坐标 (从场地的最右侧边界开始)
            current_y = -self.wall_width + 0.5
            
            # 遍历所有的门，在门与门之间“砌墙”
            for door_y in doors_y:
                # 计算当前墙段的终点 (门的下边缘)
                segment_end_y = door_y - gap_half_width 
                segment_length = segment_end_y - current_y
                
                # 如果墙段长度大于0，则生成这块墙
                if segment_length > 0:
                    center_y = current_y + segment_length / 2.0
                    half_extents = [wall_thickness / 2.0, segment_length / 2.0, wall_height / 2.0]
                    
                    v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 1])
                    c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                    
                    obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                               baseVisualShapeIndex=v_shape, 
                                               basePosition=[wall_x, center_y, wall_height / 2.0])
                    self.obstacles.append(obs_id)
                
                # 更新 current_y 到门的另一侧上边缘
                current_y = door_y + gap_half_width
                
            # --- 收尾：最后一道门到左侧边界的墙段 ---
            segment_end_y = self.wall_width - 0.5
            segment_length = segment_end_y - current_y
            if segment_length > 0:
                center_y = current_y + segment_length / 2.0
                half_extents = [wall_thickness / 2.0, segment_length / 2.0, wall_height / 2.0]
                
                v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 1])
                c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                
                obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                           baseVisualShapeIndex=v_shape, 
                                           basePosition=[wall_x, center_y, wall_height / 2.0])
                self.obstacles.append(obs_id)
            

        elif config_id == 2:    
            # ==========================================
            # 八字形逐渐缩小的袋口 (Funnel / Pocket)
            # ==========================================
            # 1. 定义袋口的 X 轴跨度 (假设小车从 X 负半轴往正半轴开)
            x_start = -5.0       # 袋口的最外端 (宽口)
            x_end = 3.0          # 袋口的最里端 (窄口)
            
            # 2. 定义开口大小 (Y轴的正负一半)
            y_gap_start = 8.0    # 宽口一半宽度 (总宽 12 米)
            y_gap_end = 1.2      # 窄口一半宽度 (总宽 2.4 米，极度考验编队收缩)
            
            # 3. 定义单侧墙壁的柱子密集程度
            num_pillars = 10    
            
            # 4. 利用线性插值 (Linear Interpolation) 生成两条斜线
            for i in range(num_pillars):
                # t 是插值系数，从 0.0 逐渐变到 1.0
                t = i / (num_pillars - 1)
                
                # 计算当前的 X 坐标
                cx = x_start + t * (x_end - x_start)
                
                # 计算当前 X 对应的 Y 边界
                cy = y_gap_start + t * (y_gap_end - y_gap_start)
                
                # 上半边的斜墙 (y > 0)
                pos_list.append([cx, cy, 0.5])
                
                # 下半边的斜墙 (y < 0，对称)
                pos_list.append([cx, -cy, 0.5])

        elif config_id == 3:
            # --- 配置 5：动态泊松盘（稀疏森林） ---
            while len(pos_list) < 6: # 生成 6 根柱子
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                
                is_far_enough = True
                for p_exist in pos_list:
                    if np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) < 4.5:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    pos_list.append([ox, oy, 0.5])

        else:
            # --- 配置 5：动态泊松盘（稀疏森林） ---
            while len(pos_list) < 10: # 生成 10 根柱子
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                
                is_far_enough = True
                for p_exist in pos_list:
                    if np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) < 4.0:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    pos_list.append([ox, oy, 0.5])

        # 统一生成障碍物
        for pos in pos_list:
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                       baseVisualShapeIndex=v_shape, basePosition=pos)
            self.obstacles.append(obs_id)

    def _create_visual_sphere(self, pos, color):
        v = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=color)
        p.createMultiBody(baseVisualShapeIndex=v, basePosition=[pos[0], pos[1], 0.2])