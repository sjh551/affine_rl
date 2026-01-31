import os
import time
from stable_baselines3 import PPO
from formation_env import FormationGymEnv
from pybullet_env import FormationPyBulletEnv
from leader_env import SingleRobotPyBulletEnv


# 阶段一：无障碍物训练 (让它学会怎么走到绿球),训练完一次就不管了
# print(">>> 阶段 1: 训练基础导航 (无障碍)...")
# env_simple = SingleRobotPyBulletEnv(render_mode=False, enable_obstacles=False) 
# model = PPO("MlpPolicy", env_simple, verbose=1, learning_rate=3e-4) # 学习率很重要
# model.learn(total_timesteps=100000)
# model.save("ppo_stage1_nav")
# env_simple.close()

# 阶段二：有障碍物训练 (加载刚才学会走路的脑子，继续学避障)
print(">>> 阶段 2: 训练避障能力 (加载阶段1模型)...")
env_hard = SingleRobotPyBulletEnv(render_mode=False, enable_obstacles=True)
# 加载模型
model = PPO.load("ppo_stage1_nav", env=env_hard, verbose=1)
# 继续训练 20万步
model.learn(total_timesteps=100000)
model.save("ppo_final_leader")
print("训练全部完成！")