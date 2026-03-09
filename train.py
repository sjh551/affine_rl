import os
import time
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 导入自定义环境
from pybullet_env import FormationPyBulletEnv

# ======================================================================
# 🛠️ 超参数与流程控制中心
# ======================================================================
# 1. 训练配置
NUM_ENVS = 1               # 并行环境数
START_STAGE = 1            # 🚩 选择开始阶段: 1 (无障碍) 或 2 (有障碍)

# 2. 模型加载配置
# 如果想从头开始训练某个阶段，设为 True
START_FROM_SCRATCH = True 

# 如果 START_FROM_SCRATCH 为 False，指定要加载的文件名 (不带 .zip)
# 例子: "ppo2_stage1_nav" 或 "ppo2_final_leader"
LOAD_MODEL_NAME = "ppo2_stage1_nav" 

# 3. 训练长度
TIMESTEPS_STAGE1 = 1000000
TIMESTEPS_STAGE2 = 100000
# ======================================================================

def make_env(enable_obstacles=False):
    def _init():
        env = FormationPyBulletEnv(render_mode=False, enable_obstacles=enable_obstacles)
        return Monitor(env)
    return _init

def create_vectorized_env(num_envs, enable_obstacles=False):
    if num_envs > 1:
        return SubprocVecEnv([make_env(enable_obstacles) for _ in range(num_envs)])
    else:
        return DummyVecEnv([make_env(enable_obstacles)])

if __name__ == '__main__':
    
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[512, 512], vf=[512, 512])
    )

    # ------------------------------------------------------------------
    # 🏁 阶段一：基础导航训练 (无障碍)
    # ------------------------------------------------------------------
    if START_STAGE == 1:
        print(f"\n[运行中] >>> 阶段 1: 基础导航训练 (并行环境: {NUM_ENVS})")
        
        # 安全检查：禁止阶段一加载阶段二的最终模型
        if LOAD_MODEL_NAME == "ppo2_final_leader":
            print("❌ 错误: 阶段 1 不能使用 ppo2_final_leader 模型进行训练！")
            exit()

        env = create_vectorized_env(NUM_ENVS, enable_obstacles=False)

        if START_FROM_SCRATCH:
            print("🆕 正在从零开始创建模型...")
            model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                        learning_rate=3e-4, batch_size=64, n_steps=2048,
                        tensorboard_log="./ppo2_stage1_tensorboard/",
                        device="auto")
        else:
            print(f"载入模型继续训练: {LOAD_MODEL_NAME}")
            model = PPO.load(LOAD_MODEL_NAME, env=env, tensorboard_log="./ppo2_stage1_tensorboard/", device="auto")

        model.learn(total_timesteps=TIMESTEPS_STAGE1, tb_log_name="stage1_nav")
        model.save("ppo2_stage1_nav")
        env.close()
        print("✅ 阶段 1 训练结束并保存模型。")

    # ------------------------------------------------------------------
    # 🚧 阶段二：避障能力训练 (有障碍)
    # ------------------------------------------------------------------
    elif START_STAGE == 2:
        print(f"\n[运行中] >>> 阶段 2: 复杂避障训练 (并行环境: {NUM_ENVS})")
        env = create_vectorized_env(NUM_ENVS, enable_obstacles=True)

        # 逻辑：即使 START_FROM_SCRATCH 是 True，阶段二通常也需要加载阶段一的成果，除非非要硬练
        if START_FROM_SCRATCH:
            print("⚠️ 警告: 正在完全从零开始进行阶段二训练（不推荐）。")
            model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                        tensorboard_log="./ppo2_stage2_tensorboard/", device="auto")
        else:
            print(f"📂 正在加载指定的模型作为起点: {LOAD_MODEL_NAME}")
            # 加载模型并重置一些参数 (例如增加熵系数 ent_coef 鼓励避障探索)
            model = PPO.load(LOAD_MODEL_NAME, env=env, 
                             ent_coef=0.01, 
                             tensorboard_log="./ppo2_stage2_tensorboard/", device="auto")

        # 设置自动保存回调
        save_freq = max(100000 // NUM_ENVS, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq, 
            save_path='./logs_checkpoints/',
            name_prefix=f'stage2_from_{LOAD_MODEL_NAME}'
        )

        model.learn(
            total_timesteps=TIMESTEPS_STAGE2,
            callback=checkpoint_callback,
            tb_log_name=f"stage2_obs_from_{LOAD_MODEL_NAME}"
        )

        model.save("ppo2_final_leader")
        env.close()
        print("✅ 阶段 2 训练结束，最终模型已保存。")

    else:
        print("❌ 错误: START_STAGE 只能设为 1 或 2")