import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from formation_env import FormationGymEnv
from pybullet_env import FormationPyBulletEnv
import torch.nn as nn

policy_kwargs = dict(
    activation_fn=nn.Tanh,  # 激活函数，Tanh 在连续控制中通常比 ReLU 稳
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)


# 阶段一：无障碍物训练 (让它学会怎么走到绿球),训练完一次就不管了
# print(">>> 阶段 1: 训练基础导航 (无障碍)...")
# env_simple = FormationPyBulletEnv(render_mode=False, enable_obstacles=False) 
# model = PPO("MlpPolicy", env_simple, verbose=1, learning_rate=3e-4, policy_kwargs=policy_kwargs,tensorboard_log="./ppo2_stage1_tensorboard/") # 学习率很重要
# model.learn(total_timesteps=100000)
# model.save("ppo2_stage1_nav")
# env_simple.close()

# 阶段二：有障碍物训练 (加载刚才学会走路的脑子，继续学避障)
print(">>> 阶段 2: 训练避障能力 (加载阶段1模型)...")
env_hard = FormationPyBulletEnv(render_mode=False, enable_obstacles=True)

# 1. 加载模型
# 注意：微调时，通常建议降低学习率，防止把第一阶段学好的直走能力瞬间“洗掉”
# 可以在 load 时传入 custom_objects 修改学习率，或者依赖 PPO 的自动调节
# # ppo2_stage1_nav  ppo2_final_leader
model = PPO.load("ppo2_final_leader", env=env_hard, verbose=1, 
                 learning_rate=0.0001,
                 batch_size=64,  # 小批次更新
                 n_steps=2048,    # 采样步长
                #  policy_kwargs=policy_kwargs,
                 tensorboard_log="./ppo2_stage2_tensorboard/")

# 2. 设置自动保存 (每 5万步存一个备份，防止训练崩了或者想回滚)
checkpoint_callback = CheckpointCallback(
    save_freq=100000, 
    save_path='./logs_checkpoints/',
    name_prefix='ppo2_stage2'
)

# 3. 增加训练步数
# 建议至少 50万步，PyBullet 跑得很快，这花不了太久
model.learn(
    total_timesteps=500000, 
    callback=checkpoint_callback,
    tb_log_name="stage2_obstacle" # 使用 TensorBoard 记录
)

model.save("ppo2_final_leader")
print("训练全部完成！")