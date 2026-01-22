import os
import time
from stable_baselines3 import PPO
from formation_env import FormationGymEnv

# 1. 创建环境
env = FormationGymEnv()

# 2. 定义模型 (PPO)
# policy="MlpPolicy" 表示使用全连接神经网络 (因为输入是坐标数值，不是图片)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_formation_tensorboard/")

print("开始训练 Leader... (按 Ctrl+C 可以随时停止)")

# 3. 开始训练
# total_timesteps=50000 
# 论文里用了 200,000 steps，先跑 50,000看效果
try:
    model.learn(total_timesteps=50000)
    print("训练完成！")
except KeyboardInterrupt:
    print("训练被手动中断，正在保存模型...")

# 4. 保存模型
model.save("ppo_leader_model")
print("模型已保存为 ppo_leader_model.zip")