import pybullet as p
import pybullet_data
import os

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