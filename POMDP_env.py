import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from utils.formation_graph import FormationGraph
from utils.formation_utils import get_square_configuration, get_square_topology, get_nominal_configuration, get_topology, get_five_agent_configuration, get_five_agent_topology
import itertools
from collections import deque
import os
import shutil


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RANGER_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "ranger_mini_v3", "urdf", "ranger_mini.urdf").replace("\\", "/")

class FormationPyBulletEnv(gym.Env):
    def __init__(self, render_mode=False, enable_obstacles=True, hide_robots=False, draw_line=True, k_accel_angular=0.2, k_vel_align=1.0, k_progress = 10.0,
                 k_shape_err = 0.01, k_disconnect = 0.2, k_smooth_linear = 0.05, k_omega_limit = 0.2, k_path_cost = 0.2):
        super(FormationPyBulletEnv, self).__init__()
        
        # self.k_accel_angular = k_accel_angular
        self.k_vel_align = k_vel_align
        self.k_progress = k_progress
        self.k_shape_err = k_shape_err 
        self.k_disconnect = k_disconnect
        self.k_smooth_linear = k_smooth_linear
        # self.k_omega_limit = k_omega_limit
        self.k_path_cost = k_path_cost

        # --- 1. PyBullet 初始化配置 ---
        self.render_mode = render_mode
        self.hide_robots = hide_robots  # 是否隐藏小车模型
        self.draw_line = draw_line  # 是否绘制队形连线

        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 参数配置 ---
        self.n_leaders = 2
        self.n_followers = 2
        self.num_robots = self.n_leaders + self.n_followers 
        self.robot_ids = []
        

        # --- 初始化标称构型和权重 ---
        self.g_nom, self.edges = get_square_configuration(), get_square_topology()
        # self.g_nom, self.edges = get_five_agent_configuration(), get_five_agent_topology()
        # 2. 构建全局大矩阵并进行 2D 降维分块
        self._init_global_matrices()

        # --- 动作空间 (6维) ---
        # [LeaderA_Vx, LeaderA_Vy, LeaderA_omega, LeaderB_Vx, LeaderB_Vy, LeaderB_omega]  不需要角速度，角速度可以直接为0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.n_nearest_obs = 16 # 最近的12个障碍物 + 四面墙 = 16 个观测点
        self.n_lidar_rays = 180 # 雷达一圈多少线
        self.lidar_range = 7.0 # 最大长度


        self.frame_len = 20  # 20帧的历史数据
        self.in_channels = 5 # 每3条雷达线合成一个输入通道，减少输入维度
        self.lidar_buffer_A = deque(maxlen=self.frame_len)
        self.lidar_buffer_B = deque(maxlen=self.frame_len)

        # 运动学特征 (距离, 相对向量, 速度, 姿态) 
        kinematics_dim = 2 + 4 + 4 + 6 + 2 + 4 # 6 表示sin(),cos(),以及当前的角速度w x 2，-1 表示我们把目标的 Y 轴坐标去掉了 + 2 表示期望向量 + 上一时刻的动作

        self.observation_space = spaces.Dict({
            # 基础运动学状态 (1D 向量)
            "kinematics": spaces.Box(low=-np.inf, high=np.inf, shape=(kinematics_dim,), dtype=np.float32),
            # 雷达 A 的状态: 3 个通道 (历史帧), 每个通道 180 个数值 
            "lidar_A": spaces.Box(low=0.0, high=1.0, shape=(self.in_channels, 60), dtype=np.float32),
            # 雷达 B 的状态
            "lidar_B": spaces.Box(low=0.0, high=1.0, shape=(self.in_channels, 60), dtype=np.float32)
        })
        # print(f"Observation space dimension: {obs_dim}")

        self.target_pos = np.array([8.5, 0, 0])
        # 定义 Leader A 相对于 Leader B 的完美相对向量 [-4, -2, np.pi/8],   # g1 (Leader) [ 2,  4, -np.pi/4],  # g2 (Leader)，A在B的下方，、
        self.nominal_vec = np.array([(self.g_nom[0] - self.g_nom[1])[0], (self.g_nom[0] - self.g_nom[1])[1], 3 * np.pi/8])
        # print(self.nominal_vec)
        self.wheel_indices = [3,1,5,7] # 小车轮子关节索引
        self.steer_indices = [2,0,4,6] # 小车转向关节索引

        self.max_vel = 1.0  # 最大线速度 1.5 m/s
        self.max_omega = 1.0  # 最大角速度 1.0 rad/s
        self.wheel_radius = 0.09  # 轮半径，m
        self.half_horizon = 0.25 # 水平的一半长
        self.half_vertical = 0.19 # 垂直的一半

        self.safe_margin = 1.0 # 设定的agent之间的警戒距离
        self.obstacles = []
        self.max_steps = 1500
        self.wall_length = 20.0 # 墙的长度的一半
        self.wall_width = 10.0 # 墙的宽度的一半
        self.enable_obstacles = enable_obstacles
        self.last_center_dist = None
        # 用于可视化的线段ID
        self.debug_line_ids = []

        self.followerA_err = []
        self.followerB_err = []

        if not hasattr(self, 'last_follower_vels'):
            self.last_follower_vels = [np.zeros(3) for _ in range(self.n_followers)]

    def _init_global_matrices(self):
        """
        构建 9 个智能体的全局拉普拉斯矩阵，并剔除角度维度，进行分块
        """
        # 1. 构建矩阵 M
        fg = FormationGraph(self.g_nom)
        self.M_3d = fg.build_laplacian(theta=0)
        
        # 转成 1D 数组方便矩阵相乘
        self.g_nom_flat = np.concatenate(self.g_nom) # shape: (18,)
        
        # 矩阵分块 (Leaders: 前 6 行/列; Followers: 后 21 行/列)
        split_idx = self.n_leaders * 3 # 2 个 leader * 3D = 6
        
        self.M_ff = self.M_3d[split_idx:, split_idx:] # (21, 21)
        self.M_fl = self.M_3d[split_idx:, :split_idx] # (21, 6)
        
        # 5. 提前求逆 (极大提升仿真速度)
        self.M_ff_inv = np.linalg.inv(self.M_ff)

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # --- 创建围墙 ---
        self._create_walls()

        # --- 随机生成目标点 ---
        tx = np.random.uniform(self.wall_length-4.5, self.wall_length-4.0)
        ty = np.random.uniform(-self.wall_width+6.0, self.wall_width-6.0)
        self.target_pos = np.array([tx, ty, 0])
        self._create_visual_sphere(self.target_pos, [0, 1, 0, 0.5])

        # --- create robots ---
        self.robot_ids = []
        
        # 1. 随机生成一个队伍的起跑中心点
        self.start_center = np.array([
            np.random.uniform(-self.wall_length + 6.0, -self.wall_length + 6.2),
            np.random.uniform(-2.0, 2.0)
        ])

        self.L = self.target_pos[:2] - self.start_center[:2]
        self.L_norm = np.linalg.norm(self.L)
        dir_start_to_goal = self.L / self.L_norm
        virtual_extension_dist = 4.0 
        self.virtual_target_pos = self.target_pos[:2] + dir_start_to_goal * virtual_extension_dist  # 把目标点后移


        # 2. 标称构型加随机位置
        for i in range(self.num_robots):
            rx = self.start_center[0] + self.g_nom[i][0]
            ry = self.start_center[1] + self.g_nom[i][1]
            yaw = self.g_nom[i][2]
            
            spawn_pos = [rx, ry, 0.15]
            spawn_quat = p.getQuaternionFromEuler([0, 0, yaw])

            rid = p.loadURDF(RANGER_URDF_PATH, basePosition=spawn_pos, baseOrientation=spawn_quat)

            # 前两台 Leader 车身设为浅绿色，后续机器人保持模型默认颜色
            if i < self.n_leaders:
                light_green = [0.65, 0.9, 0.65, 1.0]
                p.changeVisualShape(objectUniqueId=rid, linkIndex=-1, rgbaColor=light_green)
                for link_idx in range(p.getNumJoints(rid)):
                    # 轮子和转向机构保留原色，只改车身相关链接
                    if link_idx in self.wheel_indices or link_idx in self.steer_indices:
                        continue
                    p.changeVisualShape(objectUniqueId=rid, linkIndex=link_idx, rgbaColor=light_green)
            
            # 如果隐藏小车，设置其透明度为 0
            if self.hide_robots:
                p.changeVisualShape(objectUniqueId=rid, 
                        linkIndex=-1, 
                        rgbaColor=[1.0, 0.0, 0.0, 0.0])  # Alpha=0 隐藏
                # 隐藏所有链接
                for link_idx in range(p.getNumJoints(rid)):
                    p.changeVisualShape(objectUniqueId=rid, 
                            linkIndex=link_idx, 
                            rgbaColor=[1.0, 0.0, 0.0, 0.0])
            
            for joint_idx in range(p.getNumJoints(rid)):
                if joint_idx in self.steer_indices:
                    p.changeDynamics(rid, joint_idx, lateralFriction=0.1)
                                    # 解除默认的电机锁定，将其设为 0 力度的速度控制
                    p.setJointMotorControl2(rid, joint_idx, p.POSITION_CONTROL, targetPosition=0)

                elif joint_idx in self.wheel_indices:
                    p.changeDynamics(rid, joint_idx, lateralFriction=0.5, rollingFriction=0.02)
                    p.setJointMotorControl2(rid, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

            p.changeDynamics(rid, -1, lateralFriction=0.5, rollingFriction=0.01)

            self.robot_ids.append(rid)


        # --- C. 生成障碍物 ---
        if self.enable_obstacles:
            self._generate_obstacles()
        else:
            self.obstacles = [] # 确保列表为空
        for _ in range(100):
            for rid in self.robot_ids:
                # 强制方向盘回正
                p.setJointMotorControlArray(
                    rid, self.steer_indices, p.POSITION_CONTROL, 
                    targetPositions=[0]*4, forces=[80.0]*4
                )
                # 强制轮子刹车锁死
                p.setJointMotorControlArray(
                    rid, self.wheel_indices, p.VELOCITY_CONTROL, 
                    targetVelocities=[0]*4, forces=[20.0]*4
                )
            p.stepSimulation()

        self.current_step = 0

        # --- 初始化上一帧的速度记录 ---
        self.last_vels = [np.zeros(3) for _ in range(self.num_robots)]
        # 重置上一帧动作 (用于计算动作平滑度)
        self.last_action = np.zeros(self.action_space.shape)
        # 初始化距离记录 (用两车中点计算)
        center_pos = self._get_center_pos()

        # print(f"初始中心位置: {center_pos}, 目标位置: {self.target_pos}")
        vec_to_goal = self.target_pos[:2] - center_pos[:2]
        target_yaw = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        self.last_center_dist = np.linalg.norm(vec_to_goal)

        # 获取第 0 帧的雷达数据
        pos_A, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        pos_B, _ = p.getBasePositionAndOrientation(self.robot_ids[1])
        init_lidar_A = self._get_lidar(pos_A[:2], target_yaw)
        init_lidar_B = self._get_lidar(pos_B[:2], target_yaw)

        # 清空之前的残留，并将初始帧复制 20 份填入
        self.lidar_buffer_A.clear()
        self.lidar_buffer_B.clear()
        for _ in range(self.frame_len):
            self.lidar_buffer_A.append(init_lidar_A)
            self.lidar_buffer_B.append(init_lidar_B)

        # 清除旧的线
        p.removeAllUserDebugItems()
        self.debug_line_ids = []
        self.ep_rew_progress = 0.0   # 进度得分
        self.ep_rew_shape = 0.0      # 队形弹簧扣分
        self.ep_rew_collision = 0.0  # 碰撞扣分
        self.ep_rew_time = 0.0       # 时间扣分
        self.ep_rew_smooth = 0.0     # 平滑度得分
        self.ep_formation_error_sum = 0.0   # 每步编队误差之和的累计
        self.ep_formation_error_max = 0.0    # 回合内最大的每步编队误差
        obs = self._get_obs(center_pos)

        self.is_frame_locked = False
        self.locked_target_yaw = 0.0
        
        return obs, {}

    def step(self, action):
        # print(action)
        # 当前动作为小车目标坐标系下的动作，不是小车自身坐标系的动作
        self.current_step += 1

        # old_center_pos = self._get_center_pos()
        # vec_to_goal = self.target_pos[:2] - center_pos[:2]
        # dist_to_goal = np.linalg.norm(vec_to_goal)

        action_diff = action - self.last_action
        self.last_action = action.copy()
        
        # 提取 Leader A 的线速度和角速度
        vAx = action[0] * self.max_vel
        vAy = action[1] * self.max_vel
        wA = 0.0

        # 提取 Leader B 的线速度和角速度
        vBx = action[2] * self.max_vel
        vBy = action[3] * self.max_vel
        wB = 0.0

        # print(abs(wA) + abs(wB))

        vAx_body = vAx * self._cosA - vAy * self._sinA
        vAy_body = vAx * self._sinA + vAy * self._cosA
        vBx_body = vBx * self._cosB - vBy * self._sinB
        vBy_body = vBx * self._sinB + vBy * self._cosB

        # 驱动 leaders
        self._drive_robot(self.robot_ids[0], vAx_body, vAy_body, wA)
        self._drive_robot(self.robot_ids[1], vBx_body, vBy_body, wB)

        # --- 2. Follower 控制 (外部算法接口) ---
        # 获取当前所有机器人的状态传给算法
        if self.n_followers > 0:
            states = self._get_all_robot_states()
            # print(f'states: {states}')
            # 驱动 followers
            self._control_follower(states)

        # --- 3. 物理仿真 (高频率) ---
        for _ in range(24): 
            p.stepSimulation()

        # --- 4. 可视化队形连线 (画三角形) ---
        if self.render_mode and self.draw_line:
            self._draw_formation_lines()
            time.sleep(0.01)
            
        # --- 5. 获取状态与计算奖励 ---
        new_center_pos = self._get_center_pos()
        obs = self._get_obs(new_center_pos)
        reward, reward_info = self._compute_reward(new_center_pos,action,action_diff)

        # 累加这一步的各项得分
        self.ep_rew_progress += reward_info['r_progress']
        self.ep_rew_shape += reward_info['r_shape']
        self.ep_rew_collision += reward_info['r_collision']
        self.ep_rew_time += reward_info['r_time']
        self.ep_rew_smooth += reward_info['r_smooth']
        self.ep_formation_error_sum += reward_info['formation_error_step']
        self.ep_formation_error_max = max(self.ep_formation_error_max, reward_info['formation_error_step'])
        done = reward_info['done']
        # print(f"Step: {self.current_step}, Reward: {reward:.3f}, Progress: {reward_info['r_progress']:.3f}, Shape: {reward_info['r_shape']:.3f}, Collision: {reward_info['r_collision']:.3f}, Time: {reward_info['r_time']:.3f}, Smooth: {reward_info['r_smooth']:.3f}")
        info = {}
        if done:
            info['ep_rew_progress'] = self.ep_rew_progress
            info['ep_rew_shape'] = self.ep_rew_shape
            info['ep_rew_collision'] = self.ep_rew_collision
            info['ep_rew_smooth'] = self.ep_rew_smooth
            info['ep_rew_time'] = self.ep_rew_time
            info['ep_formation_error_avg'] = self.ep_formation_error_sum / max(self.current_step, 1)
            info['ep_formation_error_max'] = self.ep_formation_error_max
            info['reason'] = reward_info.get('reason', 'Unknown')

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

        return obs, reward, terminated, truncated, info
    
    # ============================================================
    #   [核心接口] Follower 控制算法
    # ============================================================
    def _control_follower(self, states):
        """
        使用全局 3D 矩阵一次性计算跟随者的目标速度
        """
        # 1. 提取实时的全局状态 (x, y, yaw)
        # p_flat shape: (27,), v_l_flat shape: (6,)
        p_all = []
        for i in range(self.num_robots):  
            p_all.extend(states[i]['pos'][:2])
            p_all.extend([states[i]['yaw']])
        p_flat = np.array(p_all)
        
        v_l = []
        for i in range(2):
            v_l.extend(states[i]['vel'][:2])
            # 只提取 Z 轴角速度，保证 leader_vel 的维度正好是 6 (2 * 3)
            v_l.extend([states[i]['angular_vel']]) 
        v_l_flat = np.array(v_l)
        
        k_f = 2.0 # 跟随者增益
        
        # 2. 核心数学公式：v_f = M_ff_inv @ ( -k_f * (M @ p)_f - M_fl @ v_l )
        # 计算全局误差 (shape: 27,)
        error_global = -k_f * (self.M_3d @ p_flat)
        
        # 截取属于 Followers 的部分 (后 21 维)
        error_f = error_global[6:]
        
        # 计算 Leader 运动带来的耦合项 (21x6 @ 6, = 21,)
        coupling_f = -self.M_fl @ v_l_flat
        
        # 求解 7 个 Follower 的 3D 速度 (世界系, shape: 21,)
        v_f_flat = self.M_ff_inv @ (error_f + coupling_f)
        
        # 3. 解析回各个机器人的 3D 速度并安全限幅
        follower_vels = []
        for i in range(self.n_followers):
            vel_3d = np.array(v_f_flat[i*3 : i*3+3]) # shape: (3,)
            follower_vels.append(vel_3d)

        # 4. 世界坐标系 -> 体坐标系 转换与驱动
        for idx in range(self.n_followers):
            robot_id = self.robot_ids[idx + 2]

            _, APF_vel = self._get_obstacle_repulsion(states[idx + 2]['pos'][:2])
            # print(f"Follower {idx} APF_vel: ({APF_vel[0]:.2f}, {APF_vel[1]:.2f})")

            world_vx = follower_vels[idx][0] + APF_vel[0]  # 期望绝对 X 速度
            world_vy = follower_vels[idx][1] + APF_vel[1]  # 期望绝对 Y 速度
            # w = follower_vels[idx][2]        # 期望角速度
            
            # 直接从 states 字典获取 yaw
            yaw = states[idx + 2]['yaw']
            
            # 将世界系平移速度转化为全向小车体坐标系的 (vx, vy)
            v_body_x = world_vx * np.cos(yaw) + world_vy * np.sin(yaw)
            v_body_y = -world_vx * np.sin(yaw) + world_vy * np.cos(yaw)
            # if idx == 0:
            #     print(f"angle:{np.arctan2(v_body_y,v_body_x)}")
            
            # 调用底层舵轮解算函数
            # print(f"Follower {idx} world_v: ({world_vx:.2f}, {world_vy:.2f}), body_v: ({v_body_x:.2f}, {v_body_y:.2f})")
            self._drive_robot(robot_id, v_body_x, v_body_y, 0)
    
    def _drive_robot(self, robot_id, v_x, v_y, w):
        """
        接收输入为小车体坐标系下的速度
        """
        R = 0.09  
        wheel_positions = [
            (0.25, 0.19),    # FL 
            (0.25, -0.19),   # FR 
            (-0.25, 0.19),   # RL 
            (-0.25, -0.19)   # RR 
        ]
        
        target_steering_angles = []
        target_wheel_speeds = []
        current_joint_states = p.getJointStates(robot_id, self.steer_indices)
        
        for i, (wx, wy) in enumerate(wheel_positions):
            wheel_vx = v_x - w * wy
            wheel_vy = v_y + w * wx
            
            speed_2d = np.hypot(wheel_vx, wheel_vy)
            if speed_2d < 2e-2:
                # 速度太小时保持当前角度，防止原地抽搐
                current_angle = current_joint_states[i][0]  
                target_steering_angles.append(current_angle) 
                target_wheel_speeds.append(0.0)
                continue
                
            ideal_angle = np.arctan2(wheel_vy, wheel_vx)
            cmd_angle = -ideal_angle # 修正 URDF Z 轴反向
            ideal_speed = speed_2d / R
            
            current_angle = current_joint_states[i][0]
            
            # =================================================
            # 核心优化：全向轮的双态物理等效计算
            # =================================================
            # 状态1：正向驱动
            angle1 = cmd_angle
            speed1 = ideal_speed
            
            # 状态2：反向驱动 (轮子掉转 180 度，速度给负数)
            if angle1 > 0:
                angle2 = angle1 - np.pi
            else:
                angle2 = angle1 + np.pi
            speed2 = -ideal_speed
            
            # 过滤：只保留满足 URDF 物理限位 [-2.1, 2.1] 的合法状态
            valid_configs = []
            limit = 2.1 
            if -limit <= angle1 <= limit:
                valid_configs.append((angle1, speed1))
            if -limit <= angle2 <= limit:
                valid_configs.append((angle2, speed2))
                
            # 决策：选择离当前轮子物理角度最近的那个状态 (完美杀死抖动！)
            if len(valid_configs) > 0:
                best_config = min(valid_configs, key=lambda x: abs(x[0] - current_angle))
                best_angle, best_speed = best_config
            else:
                # 数学托底：因为 angle1 和 angle2 必定相差 3.14，总有一个在 [-2.1, 2.1] 的 4.2 跨度内。
                best_angle, best_speed = current_angle, 0.0
                
            # 离合器防抖保护
            angle_diff = abs(best_angle - current_angle)
            speed_penalty = max(0, np.cos(angle_diff))
            final_speed = best_speed * speed_penalty
            
            target_steering_angles.append(best_angle)
            target_wheel_speeds.append(final_speed)
            
        # 3. 发送指令
        p.setJointMotorControlArray(
            robot_id,
            self.steer_indices,
            p.POSITION_CONTROL,
            targetPositions=target_steering_angles,
            forces=[80.0] * 4  # <--- 将扭矩从 5 提升到 80，彻底击碎侧向摩擦力死锁
        )
        
        p.setJointMotorControlArray(
            robot_id,
            self.wheel_indices,
            p.VELOCITY_CONTROL,
            targetVelocities=target_wheel_speeds,
            forces=[20.0] * 4  # 速度控制不施加额外力，完全由 POSITION_CONTROL 的转向来调节
        )
    
    def _get_obs(self, center_pos):
        # the centroid of the whole formation
        center_pos = center_pos[:2]
        vec_to_goal = self.target_pos[:2] - center_pos
        vec_to_virtual_goal = self.virtual_target_pos - center_pos

        # --- 1. 获取原始世界坐标系数据 ---
        pos_A, quat_A = p.getBasePositionAndOrientation(self.robot_ids[0])
        yaw_A = p.getEulerFromQuaternion(quat_A)[2] # A的航向角
        vel_A, angular_vel_A = p.getBaseVelocity(self.robot_ids[0])
        w_A = angular_vel_A[2]  # A的角速度
        pA = np.array(pos_A[:2])
        vA = np.array(vel_A[:2])

        target_yaw = np.arctan2(vec_to_virtual_goal[1], vec_to_virtual_goal[0])
        delta_yaw_A = target_yaw - yaw_A
        

        pos_B, quat_B = p.getBasePositionAndOrientation(self.robot_ids[1])
        yaw_B = p.getEulerFromQuaternion(quat_B)[2] # B的航向角
        vel_B, angular_vel_B = p.getBaseVelocity(self.robot_ids[1])
        w_B = angular_vel_B[2]  # B的角速度
        pB = np.array(pos_B[:2]) # 转换为 numpy 数组 (只取 x, y)
        vB = np.array(vel_B[:2])
        delta_yaw_B = target_yaw - yaw_B

        self._cosA = np.cos(delta_yaw_A)
        self._sinA = np.sin(delta_yaw_A)
        self._cosB = np.cos(delta_yaw_B)
        self._sinB = np.sin(delta_yaw_B)
        
        # --- 2. 构建旋转矩阵 (World -> Body/Target Frame) ---

        c, s = np.cos(-target_yaw), np.sin(-target_yaw)
        R = np.array([[c, -s], [s, c]])
        self.last_target_yaw = target_yaw
        
        # --- 3. [关键修改] 统一坐标变换 ---
        
        # A. 目标向量 (转到 Body Frame)
        rel_goal_body = R @ vec_to_goal 
        
        # B. 速度向量 (也必须转！否则网络不知道是侧身走还是直走)
        vA_body = R @ vA
        vB_body = R @ vB

        # D. Leader A 和 B 的位置 (转到 Body Frame)
        pos_A_body = R @ (pA - center_pos)
        pos_AB_body = R @ (pA - pB)  # A 相对于 B 的位置

        desired_AB_body = R @ (self.nominal_vec[:2])  # 标称构型中 A 相对于 B 的位置
        
        # --- 4. [核心重构] 分别计算障碍物相对于 A, B, C 的位置 ---
        

        # 分别获取 A, B 身边的障碍物特征
        lidar_A = self._get_lidar(pA, target_yaw, self.n_lidar_rays, self.lidar_range)
        lidar_B = self._get_lidar(pB, target_yaw, self.n_lidar_rays, self.lidar_range)

        # append 会自动把最老的一帧挤出去
        self.lidar_buffer_A.append(lidar_A)
        self.lidar_buffer_B.append(lidar_B)

        sparse_frames_A = [
            self.lidar_buffer_A[0], 
            self.lidar_buffer_A[4], 
            self.lidar_buffer_A[9], 
            self.lidar_buffer_A[14], 
            self.lidar_buffer_A[-1]
        ]
        sparse_frames_B = [
            self.lidar_buffer_B[0], 
            self.lidar_buffer_B[4], 
            self.lidar_buffer_B[9], 
            self.lidar_buffer_B[14], 
            self.lidar_buffer_B[-1]
        ]
        stacked_lidar_A = np.array(sparse_frames_A, dtype=np.float32)
        stacked_lidar_B = np.array(sparse_frames_B, dtype=np.float32)
        # print(f"lidar_A shape: {stacked_lidar_A.shape}, lidar_B shape: {stacked_lidar_B.shape}")

        # --- 5. 组装最终的 Observation ---
        kinematics_obs = np.array([
            # 目标 (2维)
            rel_goal_body[0] / self.wall_length,
            rel_goal_body[1] / self.wall_width,

            # 位置 （4维，转到 Body Frame）
            pos_A_body[0] / self.wall_length,
            pos_A_body[1] / self.wall_width,
            pos_AB_body[0] / self.wall_length,
            pos_AB_body[1] / self.wall_width,
            desired_AB_body[0] / self.wall_length,
            desired_AB_body[1] / self.wall_width,

            # 速度 (4维)
            vA_body[0], vA_body[1],
            vB_body[0], vB_body[1],

            # 姿态相关（6维）
            self._sinA, self._cosA, w_A,
            self._sinB, self._cosB, w_B,

            # 让网络拥有肌肉记忆
            self.last_action[0], self.last_action[1], 
            self.last_action[2], self.last_action[3]
        
        ],dtype=np.float32)
        # 返回字典结构
        obs = {
            "kinematics": kinematics_obs,
            "lidar_A": stacked_lidar_A,
            "lidar_B": stacked_lidar_B
        }
        return obs
    
    def _compute_reward(self, center_pos, action, action_diff):
        reward = 0
        done = False
        reason = ""
        formation_error_step = 0.0

        # --- 1. 进度奖励 ---
        vec_to_goal = self.target_pos[:2] - center_pos[:2]
        dist_to_goal = np.linalg.norm(vec_to_goal)
        dir_to_goal = vec_to_goal / (dist_to_goal + 1e-8) # 加 1e-8 防止除以 0

        if not hasattr(self, 'last_center_dist'): self.last_center_dist = dist_to_goal

        pos_A, quat_A = p.getBasePositionAndOrientation(self.robot_ids[0])
        _,_,yaw_A = p.getEulerFromQuaternion(quat_A)
        pos_B, quat_B = p.getBasePositionAndOrientation(self.robot_ids[1])
        _,_,yaw_B = p.getEulerFromQuaternion(quat_B)
        pA_3d = np.array([pos_A[0], pos_A[1], yaw_A])
        pB_3d = np.array([pos_B[0], pos_B[1], yaw_B])

        vel_A, _ = p.getBaseVelocity(self.robot_ids[0])
        vA_vec = np.array(vel_A[:2])
        speed_A = np.linalg.norm(vA_vec)

        vel_B, _ = p.getBaseVelocity(self.robot_ids[1])
        vB_vec = np.array(vel_B[:2])
        speed_B = np.linalg.norm(vB_vec)

        # 3. 编队误差状态
        actual_vec = pA_3d[:2] - pB_3d[:2]
        # dist_AB = np.linalg.norm(actual_vec)
        shape_error = np.linalg.norm(actual_vec - self.nominal_vec[:2])

        # 4. 获取所有机器人的 2D 位置
        all_pos = [np.array(p.getBasePositionAndOrientation(rid)[0][:2]) for rid in self.robot_ids]


        # =========================================================
        # [1] 时间与进度奖励 (Time & Progress)
        # =========================================================
        r_time = -0.1  # 每步固定的时间惩罚

        k_progress = 10.0  # 进度奖励系数

        r_progress = (self.last_center_dist - dist_to_goal) * k_progress
        self.last_center_dist = dist_to_goal


        # =========================================================
        # [2] 编队形状与朝向奖励 (Shape & Alignment)
        # =========================================================
        r_shape = 0.0

        # C. 目标点附近形状约束
        if dist_to_goal < 5.0 and shape_error > 2.0:    # 我发现用这种if条件判断反而有利于训练，如果只管平滑的奖励函数反而效果不好，因为这样在可以保证在初期不管这个惩罚
            r_shape -= self.k_shape_err * shape_error
        # print(r_shape)


        # =========================================================
        # [3] 动作平滑与路径跟踪惩罚 (Smoothness & Path)    去掉这个惩罚好像没有什么影响
        # =========================================================
        r_smooth = 0.0

        # A. 速度朝向一致性 (距离目标较远时鼓励朝向目标)

        # if speed_A > 0.1:
        #     dot_A = np.dot(vA_vec / (speed_A + 1e-8), dir_to_goal)
        #     # 使用 e^(1 - dot) - 1.0
        #     # dot=1 (正向)  -> e^0 - 1 = 0 
        #     # dot=0 (侧向)  -> e^1 - 1 ≈ 1.71
        #     # dot=-1 (背向) -> e^2 - 1 ≈ 6.38 (暴击惩罚！)
        #     r_smooth -= self.k_vel_align * 1/(1 + np.exp(-5*(dist_to_goal-2.0))) * (np.exp(1.0 - dot_A) - 1.0)
            
        # if speed_B > 0.1:
        #     dot_B = np.dot(vB_vec / (speed_B + 1e-8), dir_to_goal)
        #     r_smooth -= self.k_vel_align * 1/(1 + np.exp(-5*(dist_to_goal-2.0))) * (np.exp(1.0 - dot_B) - 1.0)

        if dist_to_goal > 2.0:
            if speed_A > 0.1:
                r_smooth -= self.k_vel_align * (1.0 - np.dot(vA_vec / (speed_A + 1e-8), dir_to_goal))
            if speed_B > 0.1:
                r_smooth -= self.k_vel_align * (1.0 - np.dot(vB_vec / (speed_B + 1e-8), dir_to_goal))
        
        # A. 动作变化率惩罚 (防抽搐)
        # dvA_linear = np.linalg.norm(action_diff[0:2])
        # dvB_linear = np.linalg.norm(action_diff[2:4])
        # r_smooth -= self.k_smooth_linear * (dvA_linear + dvB_linear)


        # B. 路径横向偏差惩罚 (Cross Track Error)
        R_vec = center_pos[:2] - self.start_center[:2]
        cross_product = abs(self.L[0] * R_vec[1] - self.L[1] * R_vec[0])
        cte = cross_product / self.L_norm
        # print(cte)
        r_smooth -= self.k_path_cost * cte


        # =========================================================
        # [4] 碰撞与安全惩罚 (Collision & Safety)
        # =========================================================
        r_collision = 0.0
        
        k_wall_crash = 50.0      # 撞墙惩罚
        k_team_crash = 50.0      # 互撞惩罚
        k_obs_crash = 50.0       # 撞障碍物惩罚
        k_repulsion_base = 0.1   # 势场排斥力基数
        
        # 预先推算 Followers 估计位置 (用于避障预警)
        estimated_team_positions = []
        if self.n_followers > 0 and self.enable_obstacles:
            pre_A = self._forward_motion(all_pos[0], vA_vec, yaw_A) # 推算领导者下一步的位置
            pre_B = self._forward_motion(all_pos[1], vB_vec, yaw_B)
            p_leader_pre_flat = np.concatenate([pre_A, pre_B])

            p_leader_current_flat = np.concatenate([[all_pos[0][0], all_pos[0][1], yaw_A], [all_pos[1][0], all_pos[1][1], yaw_B]])
    
            p_follower_est_flat = -self.M_ff_inv @ self.M_fl @ p_leader_pre_flat
            p_follower_expected = -self.M_ff_inv @ self.M_fl @ p_leader_current_flat    # calculates the current expected follower' positions

            for i in range(self.n_followers):
                est_pos = np.array([p_follower_est_flat[i*3], p_follower_est_flat[i*3 + 1]])    # 取出预测的位置
                expected_pos = np.array([p_follower_expected[i*3], p_follower_expected[i*3 + 1]])
                actual_pos = all_pos[i + 2]
                error_dist = np.linalg.norm(actual_pos - expected_pos)
                formation_error_step += error_dist
                tolerance = 0.5 # 允许一定的误差范围，超过这个范围才扣分
                if error_dist > tolerance:
                    r_collision -= 1.0 * (error_dist - tolerance)
                    # print(f"Follower {i} deformation error: {error_dist:.2f}, r_collision: {r_collision:.2f}")

                min_dist, _  = self._get_obstacle_repulsion(est_pos)   # 计算离最近的位置
                if min_dist < self.safe_margin:
                    cost  = 3.0 *(self.safe_margin - min_dist)   # 离得越近，惩罚越大
                    # print(cost)
                    r_collision -=  cost

        # print(r_collision)


        # A. 撞墙检测
        limit_x = self.wall_length - 0.5 - 0.6
        limit_y = self.wall_width - 0.5 - 0.6
        for i, pos_2d in enumerate(all_pos):
            if abs(pos_2d[0]) > limit_x or abs(pos_2d[1]) > limit_y:
                r_collision -= k_wall_crash
                done = True
                reason = f"Wall_Out (Bot {i})" 
                break

        # B. 团队互撞检测
        if not done:
            for p1, p2 in itertools.combinations(all_pos, 2):
                if np.linalg.norm(p1 - p2) < self.safe_margin:
                    r_collision -= k_team_crash
                    done = True
                    reason = "Crash_Team"
                    break

        # C. 障碍物碰撞与斥力势场
        if self.enable_obstacles and not done:
            crash_threshold = 0.05  # 表面距离小于 5cm 认为撞毁
            warn_threshold = 0.35   # 表面距离小于 35cm 开始施加斥力       
            
            # 遍历所有机器人
            for i, robot_id in enumerate(self.robot_ids):
                # 遍历所有障碍物
                for obs_id in self.obstacles:
                    
                    # 查询机器人与该障碍物之间，距离在 warn_threshold 以内的最近点信息
                    # 如果距离大于 warn_threshold，直接返回空列表
                    closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=obs_id, distance=warn_threshold)
                    
                    if len(closest_points) > 0:
                        # closest_points 是一个列表，里面可能包含多个接近点的信息
                        # 返回结果的第 8 个元素 (index 8) 是这两个点之间的直线物理距离
                        # 我们取出其中最小的距离
                        min_surface_dist = min([pt[8] for pt in closest_points])
                        
                        # --- 1. 致命碰撞判定 ---
                        if min_surface_dist <= crash_threshold:
                            r_collision -= k_obs_crash
                            done = True
                            reason = f"Crash_Obstacle (Bot {i} hit wall)"
                            break  # 撞死了就跳出障碍物循环
                            
                        # --- 2. 预警斥力判定 ---
                        elif min_surface_dist < warn_threshold:
                            # 距离越小（越接近 crash_threshold），斥力越大
                            r_collision -= k_repulsion_base * (1.0 / max(min_surface_dist, 0.01) - 1.0 / warn_threshold)
                        # print(min_surface_dist, r_collision)
                if done: 
                    break # 如果其中一台车撞死了，结束整个回合
       

        # =========================================================
        # [5] 任务终止判定 (Success / Timeout)
        # =========================================================
        k_success_bonus = 150.0  # 成功奖励

        if not done and dist_to_goal < 2.0 and shape_error <= 2.0:
            r_progress += k_success_bonus
            done = True
            reason = "Success"
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"


        # =========================================================
        # [6] 汇总与返回
        # =========================================================
        r_smooth = 0.1 * r_smooth
        reward = r_progress + r_shape + r_collision + r_time + r_smooth

        if done and reason == "Success":
            print(f"Episode End: {reason}")

        reward_info = {
            'done': done,
            'reason': reason,
            'r_progress': r_progress,
            'r_shape': r_shape,
            'r_collision': r_collision,
            'r_time': r_time,
            'r_smooth': r_smooth,
            'formation_error_step': formation_error_step
        }
        return reward, reward_info
    
    def _draw_formation_lines(self):
        """
        可视化连线
        """
        if len(self.robot_ids) < 2: return
        
        points = []
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            points.append(pos)
            
        line_color = [1, 1, 0] 
        
        # 如果只有两个 Leader，就只画它们俩之间的连线
        if self.n_followers == 0:
            edges = [(0, 1)]
        else:
            # 原有的 9 智能体连线
            edges = self.edges
            
        for u, v in edges:
            if u < len(points) and v < len(points):
                p.addUserDebugLine(
                    lineFromXYZ=points[u], 
                    lineToXYZ=points[v], 
                    lineColorRGB=line_color, 
                    lineWidth=2, 
                    lifeTime=0.1 
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
        body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[pos[0]-2.0, pos[1], 0]) #减去2.0是为了显示的时候靠近编队中心一点，因为目前不是编队中心靠近目标而是领导者中心
        return body_id

    def _generate_obstacles(self):
        self.obstacles = []
        config_id = np.random.randint(1, 3) #np.random.choice([0,3,4])
        
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

    def _get_all_robot_states(self):
        """辅助函数：获取所有机器人的位置和速度，供 Follower 算法使用"""
        states = {}
        for i, rid in enumerate(self.robot_ids):
            pos, quat = p.getBasePositionAndOrientation(rid)
            vel, angular_vel = p.getBaseVelocity(rid)
            euler = p.getEulerFromQuaternion(quat)
            yaw = euler[2]
            states[i] = {
                'pos': np.array(pos), 
                'yaw': yaw,  
                'vel': np.array(vel),
                'angular_vel': np.array(angular_vel[2]) # 只取 Z 轴角速度
            }
        return states

    def _get_center_pos(self):
        """
        计算编队中心点，只算了领导者，感觉算了跟随者不好估计
        """
        pos_list = []
        for robot_id in self.robot_ids[:self.num_robots]:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            pos_list.append(np.array(pos))
        center = np.mean(pos_list, axis=0)
        return center

    def _get_lidar(self, pos_2d, reference_yaw, num_rays=180, ray_length=7.0):
        """
        使用 PyBullet 的批量射线检测模拟 2D 激光雷达。
        reference_yaw: 雷达 0度 射线的基准朝向
        """
        ray_from = []
        ray_to = []
        # 射线发射高度稍微抬高一点点，防止扫到地面,车高0.2
        z_height = 0.23 
        
        # 构建射线束
        for i in range(num_rays):
            angle = reference_yaw + (i * 2 * np.pi / num_rays)
            ray_from.append([pos_2d[0], pos_2d[1], z_height])
            ray_to.append([pos_2d[0] + ray_length * np.cos(angle),
                           pos_2d[1] + ray_length * np.sin(angle),
                           z_height])

        # 执行批量射线检测 (极其高效，底层用 C++ 并行计算)
        results = p.rayTestBatch(ray_from, ray_to)
        
        # 解析结果：results[i][2] 是射线的命中比例 (Hit Fraction)，范围 0.0 ~ 1.0
        # 如果是 1.0，说明没扫到任何东西。
        # 我们把它反转一下：1.0 - fraction，这样越接近 1.0 代表障碍物越贴脸，0.0 代表前方畅通
        features = [1.0 - res[2] for res in results]
        features = np.array(features).reshape(60, 3).max(axis=1)
        
        return features
    
    
    def _forward_motion(self, pos, action, yaw):
        """
        计算经过时间dt后, what is the next state
        state: 位置
        return: 预测的下一状态, 2D位置
        """
        dt = 0.5    # 一个step为0.1s, 因为动作是线性的, 直接预测5步之后的状态
        s = pos.copy()
        s[0] += action[0] * dt  # x 變化
        s[1] += action[1] * dt  # y 變化

        return np.array([s[0], s[1], yaw])
    
    def _get_obstacle_repulsion(self, pos_2d, safe_margin=1.0, num_rays=90):
        """
        全向势场探测器：向四周发射射线，算出一个总的避障斥力向量
        返回: min_dist (最短距离), repulsion_vec (斥力向量 [vx, vy])
        """
        
        ray_from = []
        ray_to = []
        z_height = 0.23
        
        # 记录每根射线的方向向量
        ray_dirs = []
        
        for i in range(num_rays):
            angle = i * (2 * np.pi / num_rays)
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            ray_dirs.append(dir_vec)
            
            ray_from.append([pos_2d[0], pos_2d[1], z_height])
            ray_to.append([pos_2d[0] + safe_margin * dir_vec[0],
                           pos_2d[1] + safe_margin * dir_vec[1],
                           z_height])

        results = p.rayTestBatch(ray_from, ray_to)
        
        min_dist = safe_margin
        repulsion_vec = np.zeros(2) # 初始化斥力向量
        
        # APF 斥力系数
        k_repulsion = 1.5 
        
        for i, res in enumerate(results):
            hit_object_id = res[0]
            hit_fraction = res[2]
            
            # 只有打中非机器人实体时，才产生斥力
            if hit_fraction < 1.0 and hit_object_id not in self.robot_ids:
                dist = hit_fraction * safe_margin
                if dist < min_dist:
                    min_dist = dist
                
                # 斥力计算公式：距离越近，斥力呈指数/反比增加
                # (safe_margin - dist) 确保在边缘时斥力为 0，平滑过渡
                force_magnitude = k_repulsion * (safe_margin - dist)**2 
                
                # 斥力方向 = 射线的反方向！
                repulsion_vec -= force_magnitude * ray_dirs[i] 
                    
        return min_dist, repulsion_vec