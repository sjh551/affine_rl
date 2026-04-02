import os
import time
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 导入自定义环境
from pybullet_env import FormationPyBulletEnv


# ======================================================================
# 🛠️ 超参数与流程控制中心
# ======================================================================
# 1. 训练配置
NUM_ENVS = 2               # 并行环境数
START_STAGE = 2           # 🚩 选择开始阶段: 1 (无障碍) 或 2 (有障碍)

# 2. 模型加载配置
# 如果想从头开始训练某个阶段，设为 True
START_FROM_SCRATCH = False 

# 如果 START_FROM_SCRATCH 为 False，指定要加载的文件名 (不带 .zip)
# 例子: "ppo2_stage1_nav" 或 "ppo2_final_leader"
LOAD_MODEL_NAME = "ppo2_stage1_nav" 

# 3. 训练长度
TIMESTEPS_STAGE1 = 1000000
TIMESTEPS_STAGE2 = 1000000
# ======================================================================


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # 我们稍后会计算总维度，先随便填个 1
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "kinematics":
                # 运动学数据 (1D 向量) 直接通过，或者经过一个简单的线性层
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]
                
            elif "lidar" in key:
                # 雷达数据 (3, 180) 使用 1D 卷积提取时空特征
                n_input_channels = subspace.shape[0] # 这里是 3
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                
                # 推断 CNN 输出的维度大小
                with th.no_grad():
                    dummy_input = th.as_tensor(subspace.sample()[None]).float()
                    n_flatten = cnn(dummy_input).shape[1]
                
                # 将 CNN 提取的高维特征压缩成紧凑的向量
                linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())
                
                extractors[key] = nn.Sequential(cnn, linear)
                total_concat_size += 128 # 两个雷达加起来就是 256 维

        self.extractors = nn.ModuleDict(extractors)
        # 更新最终输出的特征总维度 (例如: 15运动学 + 128雷达A + 128雷达B = 271)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # 将所有处理后的特征拼接成一条长向量
        return th.cat(encoded_tensor_list, dim=1)

class TensorboardRewardCallback(BaseCallback):
    """
    自定义回调函数：用于捕获 info 字典里的分类奖励，并记录到 TensorBoard。
    """
    def __init__(self, verbose=0):
        super(TensorboardRewardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 遍历所有并行环境返回的 info 字典
        for info in self.locals.get("infos", []):
            # 只有当环境返回了我们自定义的 key（说明这局刚结束），才记录
            if "ep_rew_progress" in info:
                # 记录到 TensorBoard 的 "custom_rewards" 标签页下
                self.logger.record("custom_rewards/ep_rew_progress", info["ep_rew_progress"])
                self.logger.record("custom_rewards/ep_rew_shape", info["ep_rew_shape"])
                self.logger.record("custom_rewards/ep_rew_collision", info["ep_rew_collision"])
                self.logger.record("custom_rewards/ep_rew_smooth", info["ep_rew_smooth"])
                self.logger.record("custom_rewards/ep_rew_time", info["ep_rew_time"])
        return True

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
        features_extractor_class=CustomCombinedExtractor, 
        # 对于 CNN 来说 ReLU 通常比 Tanh 梯度更好
        activation_fn=nn.ReLU, 
        net_arch=dict(pi=[512, 512], vf=[512, 512])
    )
    tb_reward_callback = TensorboardRewardCallback()
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
            model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                        learning_rate=3e-4, batch_size=256, n_steps=2048,
                        tensorboard_log="./ppo2_stage1_tensorboard/",
                        device="cuda")
        else:
            print(f"载入模型继续训练: {LOAD_MODEL_NAME}")
            model = PPO.load(LOAD_MODEL_NAME, env=env, batch_size=256, tensorboard_log="./ppo2_stage1_tensorboard/", device="cuda")


        model.learn(total_timesteps=TIMESTEPS_STAGE1, tb_log_name="stage1_nav", callback=tb_reward_callback)
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
            model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs,batch_size=256,
                        tensorboard_log="./ppo2_stage2_tensorboard/", device="cuda")
        else:
            print(f"📂 正在加载指定的模型作为起点: {LOAD_MODEL_NAME}")
            # 加载模型并重置一些参数 (例如增加熵系数 ent_coef 鼓励避障探索)
            model = PPO.load(LOAD_MODEL_NAME, env=env, 
                             ent_coef=0.01, 
                             batch_size=256,
                             tensorboard_log="./ppo2_stage2_tensorboard/", device="cuda")

        # 设置自动保存回调
        save_freq = max(1000000 // NUM_ENVS, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq, 
            save_path='./logs_checkpoints/',
            name_prefix=f'stage2_from_{LOAD_MODEL_NAME}'
        )

        model.learn(
            total_timesteps=TIMESTEPS_STAGE2,
            callback=[tb_reward_callback, checkpoint_callback],
            tb_log_name=f"stage2_obs_from_{LOAD_MODEL_NAME}"
        )

        model.save("ppo2_final_leader")
        env.close()
        print("✅ 阶段 2 训练结束，最终模型已保存。")

    else:
        print("❌ 错误: START_STAGE 只能设为 1 或 2")