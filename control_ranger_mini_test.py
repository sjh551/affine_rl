import pybullet as p
import pybullet_data
import numpy as np
import time

# 1. 初始化仿真环境
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# 2. 加载 Ranger Mini V3 模型
# 请确保你的 URDF 和 meshes 路径正确
robot_id = p.loadURDF("ranger_mini_v3/urdf/ranger_mini.urdf", basePosition=[0, 0, 0.15])

# 3. 核心索引配置 (根据你之前提供的 ID 信息)
# 对应顺序: [FL, FR, RL, RR]
steer_indices = [2, 0, 4, 6] 
wheel_indices = [3, 1, 5, 7]

def _drive_robot(v_x, v_y, w):
    """
    Swerve Drive 运动学逆解 + 多状态优选 + 终极物理防抖
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
    current_joint_states = p.getJointStates(robot_id, steer_indices)
    
    for i, (wx, wy) in enumerate(wheel_positions):
        wheel_vx = v_x - w * wy
        wheel_vy = v_y + w * wx
        
        speed_2d = np.hypot(wheel_vx, wheel_vy)
        current_angle = current_joint_states[i][0]
        
        # 速度极小时，保持当前角度，防止原地抽搐
        if speed_2d < 1e-3:
            target_steering_angles.append(current_angle) 
            target_wheel_speeds.append(0.0)
            continue
            
        ideal_angle = np.arctan2(wheel_vy, wheel_vx)
        ideal_angle = -ideal_angle # 修正 URDF Z 轴反向
        
        # 将基准角度归一化到 [-pi, pi] 之间
        ideal_angle = (ideal_angle + np.pi) % (2 * np.pi) - np.pi
        ideal_speed = speed_2d / R
        
        # ===================================================
        # 核心优化 1：生成两个等效的物理状态
        # ===================================================
        # 状态 1：车头朝前走
        state1_angle = ideal_angle
        state1_speed = ideal_speed
        
        # 状态 2：车头朝后退 (角度转 180 度，速度取反)
        if ideal_angle > 0:
            state2_angle = ideal_angle - np.pi
        else:
            state2_angle = ideal_angle + np.pi
        state2_speed = -ideal_speed
        
        # ===================================================
        # 核心优化 2：安全过滤与最短路径优选
        # ===================================================
        # 过滤出不会撞到 URDF 物理限位 [-2.1, 2.1] 的状态
        # 设定 1.9 为安全裕度
        valid_states = []
        if abs(state1_angle) <= 1.9:
            valid_states.append((state1_angle, state1_speed))
        if abs(state2_angle) <= 1.9:
            valid_states.append((state2_angle, state2_speed))
            
        # 从安全的选项中，挑出一个离当前轮子角度最近的
        best_angle = current_angle
        best_speed = 0.0
        min_diff = float('inf')
        
        for ang, spd in valid_states:
            diff = abs(ang - current_angle)
            if diff < min_diff:
                min_diff = diff
                best_angle = ang
                best_speed = spd
                
        # ===================================================
        # 核心优化 3：离合器防抖，角度不到位不给油门
        # ===================================================
        speed_penalty = max(0, np.cos(min_diff))
        best_speed *= speed_penalty
        
        target_steering_angles.append(best_angle)
        target_wheel_speeds.append(best_speed)
        
    # 3. 发送指令
    p.setJointMotorControlArray(
        robot_id,
        steer_indices,
        p.POSITION_CONTROL,
        targetPositions=target_steering_angles,
        forces=[80.0] * 4  # 保持大扭矩，坚决击碎侧向摩擦力死锁
    )
    
    p.setJointMotorControlArray(
        robot_id,
        wheel_indices,
        p.VELOCITY_CONTROL,
        targetVelocities=target_wheel_speeds
    )

# 4. 实时控制循环
print("控制说明: W/S (前进后退), A/D (左右横移), Q/E (自旋), Space (停止)")

vx, vy, w = 0, 0, 0
step_val = 0.2   # 速度步进值

try:
    while True:
        keys = p.getKeyboardEvents()
        
        # 键盘处理逻辑
        new_vx, new_vy, new_w = 0, 0, 0
        
        if p.B3G_UP_ARROW in keys or ord('w') in keys: new_vx = 1.8
        if p.B3G_DOWN_ARROW in keys or ord('s') in keys: new_vx = -1.8
        if p.B3G_LEFT_ARROW in keys or ord('a') in keys: new_vy = 1.8
        if p.B3G_RIGHT_ARROW in keys or ord('d') in keys: new_vy = -1.8
        if ord('q') in keys: new_w = 2.0
        if ord('e') in keys: new_w = -2.0
        if ord(' ') in keys: new_vx, new_vy, new_w = 0, 0, 0

        # 调用你的函数
        _drive_robot(new_vx, new_vy, new_w)
        
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()