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
        
        # --- 2. 机器人参数 ---
        self.num_robots = 3
        self.robot_ids = []
        self.robot_radius = 0.25
        
        # 队形偏移量
        self.offsets = {
            1: np.array([-1.0, -1.0, 0]), 
            2: np.array([-1.0, 1.0, 0])
        }
        
        # --- 3. 定义空间 ---
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.n_obs_obstacles = 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        
        self.target_pos = np.array([0, 0, 0])
        self.obstacles = []
        self.max_steps = 1500

        self.wall_length = 10.0  # 围墙的一半长度
        self.wall_width = 5.0   # 围墙的一半宽度
        self.enable_obstacles = enable_obstacles
        self.last_dist = None
        
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # --- 创建围墙 ---
        self._create_walls()
        
        # --- A. 创建机器人 ---
        self.robot_ids = []
        start_pos = [-self.wall_length + 3.0, 0, 0.25]
        self.robot_ids.append(self._create_box_robot(start_pos, [1, 0, 0, 1])) 
        
        for i in range(1, self.num_robots):
            f_pos = [start_pos[0]+self.offsets[i][0], start_pos[1]+self.offsets[i][1], 0.25]
            self.robot_ids.append(self._create_box_robot(f_pos, [0, 0, 1, 1]))

        # --- B. 随机生成目标点 ---
        tx = np.random.uniform(self.wall_length-3.0, self.wall_length-1.0)
        ty = np.random.uniform(-self.wall_width+2.0, self.wall_width-2.0)
        self.target_pos = np.array([8.5, 0, 0])
        
        self.target_visual = self._create_visual_sphere(self.target_pos, [0, 1, 0, 0.5])
        
        # --- C. 生成障碍物 (根据参数决定) ---
        if self.enable_obstacles:
            self._generate_obstacles()
        else:
            self.obstacles = [] # 确保列表为空
        
        self.current_step = 0

        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        self.last_dist = np.linalg.norm(np.array(leader_pos[:2]) - self.target_pos[:2])

        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        
        # --- 1. Leader 控制 ---
        leader_vel = np.array([action[0], action[1], 0]) * 2.0
        p.resetBaseVelocity(self.robot_ids[0], linearVelocity=leader_vel)
        
        # --- 2. Follower 控制 ---
        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        leader_pos = np.array(leader_pos)
        
        for i in range(1, self.num_robots):
            follower_id = self.robot_ids[i]
            current_pos, _ = p.getBasePositionAndOrientation(follower_id)
            current_pos = np.array(current_pos)
            desired_pos = leader_pos + self.offsets[i]
            error = desired_pos - current_pos
            control_vel = error * 2.0
            control_vel[2] = 0
            p.resetBaseVelocity(follower_id, linearVelocity=control_vel)
            
        # --- 3. 物理步进 ---
        for _ in range(10):
            p.stepSimulation()

        if self.render_mode:
            time.sleep(0.01)
            
        # --- 4. 计算奖励和状态 ---
        obs = self._get_obs()
        reward = self._compute_reward(leader_pos)
        
        # --- 5. [优化] 结束条件判断 ---
        done = False
        dist_to_goal = np.linalg.norm(leader_pos[:2] - self.target_pos[:2])
        
        # a. 成功到达
        if dist_to_goal < 0.5:
            reward += 100
            done = True
        
        # b. 超时
        elif self.current_step >= self.max_steps:
            reward -= 10 # 超时给点小惩罚，逼迫它快点走
            done = True
            
        # c. 出界检测
        # 如果 Leader 跑出了围墙范围
        elif abs(leader_pos[0]) > self.wall_length - 0.5 or abs(leader_pos[1]) > self.wall_width - 0.5:
            reward -= 100 
            done = True
            
        # d. 碰撞检测 (障碍物)
        else:
            contact_points = p.getContactPoints(bodyA=self.robot_ids[0])
            for contact in contact_points:
                # contact[2] 是 bodyB 的 ID。0 是地面，self.wall_ids 是墙
                # 如果撞到的既不是地面，也不是自己的队友(虽然队友通常不检测自碰撞)
                if contact[2] != 0 and contact[2] not in self.robot_ids: 
                    reward -= 50
                    done = True
                    break

        if done:
            if dist_to_goal < 0.5: 
                print("原因：到达终点")
            elif self.current_step >= self.max_steps: 
                print("原因：步数耗尽（超时）")
            # 修改这里：加上 Y 轴的判断
            elif abs(leader_pos[0]) > self.wall_length - 0.5 or abs(leader_pos[1]) > self.wall_width - 0.5: 
                print("原因：越界/撞墙 (X或Y轴)")
            else: 
                print("原因：检测到物理碰撞 (障碍物)") 

        return obs, reward, done, {}

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
        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        leader_vel, _ = p.getBaseVelocity(self.robot_ids[0])
        
        obs = [leader_pos[0]/self.wall_length, leader_pos[1]/self.wall_width, leader_vel[0], leader_vel[1]]
        rel_goal = self.target_pos - np.array(leader_pos)
        obs.extend([rel_goal[0]/self.wall_length, rel_goal[1]/self.wall_width])
        
        for i in range(1, 3):
            f_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[i])
            rel_f = np.array(f_pos) - np.array(leader_pos)
            obs.extend([rel_f[0]/self.wall_length, rel_f[1]/self.wall_width])
            
        dists = []
        for obs_id in self.obstacles:
            o_pos, _ = p.getBasePositionAndOrientation(obs_id)
            d = np.linalg.norm(np.array(o_pos[:2]) - np.array(leader_pos[:2]))
            rel = np.array(o_pos[:2]) - np.array(leader_pos[:2])
            dists.append((d, rel[0], rel[1]))
        
        dists.sort(key=lambda x: x[0])
        for i in range(3):
            if i < len(dists):
                obs.extend([dists[i][1]/self.wall_length, dists[i][2]/self.wall_width, 0.3])
            else:
                obs.extend([1.0, 1.0, 0])   # 没有障碍物的时候应该默认障碍物距离很远，即归一化之后为1.0
                
        return np.array(obs, dtype=np.float32)

    def _get_lidar(self, pos):
        """模拟激光雷达：向周围发射射线"""
        results = []
        ray_froms = []
        ray_tos = []
        
        # 起点: 机器人中心 (稍微抬高一点，防止扫到地面)
        start_pos = [pos[0], pos[1], 0.25]
        
        # 生成射线角度 (-90度 到 +90度)
        fov_rad = np.radians(self.lidar_fov)
        angles = np.linspace(-fov_rad/2, fov_rad/2, self.lidar_rays)
        
        # Leader当前的朝向角 (假设它是全向移动，我们用速度方向或者固定朝向)
        # 既然是全向小车，我们假设雷达始终朝向目标方向，或者固定朝向X轴
        # 更好的做法是：如果小车有朝向，沿朝向扫描。这里简单起见，沿机器人当前速度方向扫描，
        # 或者为了简化训练，固定沿X轴扫描(Global Frame)。
        # **为了让RL学会转向，我们让雷达固定在机器人身上，但输入里包含了相对目标的方位**
        # 这里假设机器人车头始终朝向 X 轴正方向 (简化)
        base_angle = 0 
        
        for angle in angles:
            # 计算终点
            end_x = start_pos[0] + self.lidar_range * np.cos(base_angle + angle)
            end_y = start_pos[1] + self.lidar_range * np.sin(base_angle + angle)
            
            ray_froms.append(start_pos)
            ray_tos.append([end_x, end_y, 0.25])
            
        # 批量射线检测 (效率高)
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        lidar_readings = []
        for i, res in enumerate(results):
            # res[2] 是 hitFraction (0~1)。如果没有撞到，是 1.0
            dist_ratio = res[2]
            lidar_readings.append(dist_ratio)
            
            # [可选] Debug: 画出射线
            if self.render_mode:
                color = [1-dist_ratio, dist_ratio, 0] # 越近越红，越远越绿
                hit_pos = [
                    ray_froms[i][0] + (ray_tos[i][0]-ray_froms[i][0])*dist_ratio,
                    ray_froms[i][1] + (ray_tos[i][1]-ray_froms[i][1])*dist_ratio,
                    0.25
                ]
                p.addUserDebugLine(ray_froms[i], hit_pos, color, lifeTime=0.1)
                
        return lidar_readings
    
    
    def _compute_reward(self, leader_pos):
        # 1. 目标奖励
        current_dist = np.linalg.norm(np.array(leader_pos[:2]) - self.target_pos[:2])

        # 需要在类初始化(reset)时记录 self.last_dist
        if not hasattr(self, 'last_dist'):
            self.last_dist = current_dist

        # 2. 进步奖励 (Progress Reward)
        # 如果当前距离比上一步近，result > 0 (奖励)
        # 如果当前距离比上一步远，result < 0 (惩罚)
        # 放大系数 100.0 是为了让梯度更明显
        reward_progress = (self.last_dist - current_dist) * 100.0
        
        # 更新上一步距离
        self.last_dist = current_dist
        
        # 3. 时间惩罚 (Time Penalty)
        # 强迫它快点跑，每走一步都要付出代价
        reward_step = -0.1 
        
        # 4. 避障惩罚 (Collision Risk)
        reward_obs = 0
        if self.enable_obstacles:
            for obs_id in self.obstacles:
                o_pos, _ = p.getBasePositionAndOrientation(obs_id)
                
                # 计算 Leader 到障碍物中心的距离
                d = np.linalg.norm(np.array(o_pos[:2]) - np.array(leader_pos[:2]))
                
                # 机器人半径(0.25) + 障碍物半径(0.3) = 0.55 是物理碰撞边缘
                # 我们设置 1.2 米为"警戒圈"
                safe_threshold = 0.9 
                
                if d < safe_threshold:
                    # 公式逻辑：距离越近，分母越小，惩罚越大
                    # 当 d=1.2 时，惩罚 ≈ 0
                    # 当 d=0.6 时，惩罚变得很大
                    # 系数 0.5 是权重，可以调整
                    penalty = (safe_threshold - d) * 2.0
                    reward_obs -= penalty

        # 如果离墙太近 (< 0.5m)，也给点负分，教它不要贴墙走
        limit_x = self.wall_length - 0.5 
        limit_y = self.wall_width - 0.5

        if abs(leader_pos[0]) > limit_x or abs(leader_pos[1]) > limit_y:
            reward_obs -= 0.5
        # 总奖励
        return reward_progress + reward_step + reward_obs