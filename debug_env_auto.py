from pybullet_env import FormationPyBulletEnv
import time
import numpy as np

# 开启渲染模式 (GUI)
env = FormationPyBulletEnv(render_mode=True, enable_obstacles=True)
# `reset()` follows the Gym/Gymnasium convention, returning ``obs`` and ``info``.
# unpack both values so we don't accidentally treat a tuple as the observation.
obs, _ = env.reset()

print("开始 PyBullet 仿真...")

for i in range(1000):
    # 简单的测试动作：向目标方向移动
    # 这里的 obs[0], obs[1] 是目标相对位置 (参见 _get_obs)
    # action = np.array([obs['kinematics'][0], 0, 0, obs['kinematics'][0], 0, 0])
    
    # # 稍微加点随机扰动，看 Follower 跟得稳不稳
    # action[0] += np.random.normal(0, 0.1)
    action = env.action_space.sample()
    
    # 💡 极其重要的技巧：给动作加一个“向前”的偏置！
    # 如果完全随机，小车大概率在原地打转，扫不到丰富的障碍物特征。
    # 假设 action[0] 和 action[2] 是两辆车的纵向速度，我们强行让它们大概率往前开
    action[0] = abs(action[0]) * 0.8 + 0.2  # 保证有一定的基础前进速度
    action[2] = abs(action[2]) * 0.8 + 0.2 

    # (obs, reward, terminated, truncated, info)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated  # treat either case as episode end
    
    if done:
        print("回合结束，重置环境")
        obs, _ = env.reset()

env.close()