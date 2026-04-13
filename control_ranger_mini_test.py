import pybullet as p
import pybullet_data
import numpy as np
import time
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RANGER_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "ranger_mini_v3", "urdf", "ranger_mini.urdf").replace("\\", "/")

# 1. 初始化仿真环境
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# 2. 加载 Ranger Mini V3 模型
# 请确保你的 URDF 和 meshes 路径正确
for i in range(4):
    robot_id = p.loadURDF(RANGER_URDF_PATH, basePosition=[0-i*2.0, 0, 0.15])

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