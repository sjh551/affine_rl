import pybullet as p
import pybullet_data
import time
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RANGER_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "ranger_mini_v3", "urdf", "ranger_mini.urdf").replace("\\", "/")

# (0, b'fr_steering_joint', 0, 7, 6, 1, 0.0, 0.0, -2.1, 2.1, 5.0, 6.28, b'fr_steering_wheel_link', (0.0, 0.0, -1.0), (0.2742698272228022, -0.1869691975386788, -0.1544267444673498), (0.0, 0.0, 0.0, 1.0), -1)
# (1, b'fr_wheel', 0, 8, 7, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'fr_wheel_link', (0.0, 1.0, 0.0), (-1.8493e-05, -1.2713e-06, -0.163679), (0.0, 0.0, 0.0, 1.0), 0)
# (2, b'fl_steering_joint', 0, 9, 8, 1, 0.0, 0.0, -2.1, 2.1, 5.0, 6.28, b'fl_steering_wheel_link', (0.0, 0.0, -1.0), (0.2742698272228022, 0.1930308024613212, -0.1544267444673498), (0.0, 0.0, 0.0, 1.0), -1)
# (3, b'fl_wheel', 0, 10, 9, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'fl_wheel_link', (0.0, 1.0, 0.0), (0.0017956, 9.6304e-08, -0.163642), (0.0, 0.0, 0.0, 1.0), 2)
# (4, b'rl_steering_joint', 0, 11, 10, 1, 0.0, 0.0, -2.1, 2.1, 5.0, 6.28, b'rl_steering_wheel_link', (0.0, 0.0, -1.0), (-0.2257301727771978, 0.1930308024613212, -0.1544267444673498), (0.0, 0.0, 0.0, 1.0), -1)
# (5, b'rl_wheel', 0, 12, 11, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rl_wheel_link', (0.0, 1.0, 0.0), (0.00017093, -6.1255e-08, -0.163678), (0.0, 0.0, 0.0, 1.0), 4)
# (6, b'rr_steering_joint', 0, 13, 12, 1, 0.0, 0.0, -2.1, 2.1, 5.0, 6.28, b'rr_steering_wheel_link', (0.0, 0.0, -1.0), (-0.2057301727771978, -0.20296919753867879, -0.1544267444673498), (0.0, 0.0, 0.0, 1.0), -1)
# (7, b'rr_wheel', 0, 14, 13, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rr_wheel_link', (0.0, 1.0, 0.0), (0.0, 0.0, -0.12), (0.0, 0.0, 0.0, 1.0), 6)



# 1. 初始化物理引擎并添加内置数据路径
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 2. 加载地板
plane_id = p.loadURDF("plane.urdf")

# 3. 加载差速小车模型 (设置初始高度避免穿模)
car_id = p.loadURDF(RANGER_URDF_PATH, basePosition=[0, 0, 0.05])

# num_joints = p.getNumJoints(car_id)
# print(f"小车模型加载成功，关节数量: {num_joints}")

# 4. 打印小车的关节信息，你会发现左右轮是独立配置的
for i in range(p.getNumJoints(car_id)):
    print(p.getJointInfo(car_id, i))

# 保持仿真环境运行
while p.isConnected():
    p.stepSimulation()
    time.sleep(1./240.)