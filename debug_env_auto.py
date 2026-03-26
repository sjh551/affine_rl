from pybullet_env import FormationPyBulletEnv
import time
import numpy as np

# 开启渲染模式 (GUI)
env = FormationPyBulletEnv(render_mode=True, enable_obstacles=False)
# `reset()` follows the Gym/Gymnasium convention, returning ``obs`` and ``info``.
# unpack both values so we don't accidentally treat a tuple as the observation.
obs, _ = env.reset()

print("开始 PyBullet 仿真...")

for i in range(1000):
    # 简单的测试动作：向目标方向移动
    # 这里的 obs[0], obs[1] 是目标相对位置 (参见 _get_obs)
    action = np.array([obs['kinematics'][0], 0, 0, obs['kinematics'][0], 0, 0])
    
    # 稍微加点随机扰动，看 Follower 跟得稳不稳
    action[0] += np.random.normal(0, 0.1)
    
    # env.step 现在遵守最新 Gym/Gymnasium API and returns
    # (obs, reward, terminated, truncated, info)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated  # treat either case as episode end
    
    if done:
        print("回合结束，重置环境")
        obs, _ = env.reset()

env.close()