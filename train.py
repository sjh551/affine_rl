import os
import sys
import time
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from project_paths import (
    PRETRAINED_ENCODER_PATH,
    TB_STAGE1_DIR,
    TB_STAGE2_DIR,
    CHECKPOINTS_DIR,
    MODEL_FINAL_NAME,
    MODEL_STAGE1_NAME,
    ensure_output_dirs,
    model_load_path,
    model_save_base,
)

# 导入自定义环境
from POMDP_env import FormationPyBulletEnv


# ======================================================================
# 🛠️ 超参数与流程控制中心
# ======================================================================
# 1. 训练配置
NUM_ENVS = 8               # 并行环境数
START_STAGE = 1            # 🚩 选择开始阶段: 1 (无障碍) 或 2 (有障碍)

# 2. 模型加载配置
# 如果想从头开始训练某个阶段，设为 True
START_FROM_SCRATCH = True 

# 如果 START_FROM_SCRATCH 为 False，指定要加载的文件名 (不带 .zip)
# 例子: "ppo2_stage1_nav" 或 "ppo2_final_leader"
LOAD_MODEL_NAME = "ppo2_stage1_nav" 

# 3. 训练长度
TIMESTEPS_STAGE1 = 3000000
TIMESTEPS_STAGE2 = 1500000
# ======================================================================


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, pretrained_path=None, freeze_encoder=False):
        # 先随便填个 1
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        # 保存雷达模块的引用，用于后续加载权重和冻结
        self.lidar_modules = {}

        for key, subspace in observation_space.spaces.items():
            if key == "kinematics":
                # 运动学数据 (1D 向量) 直接通过，或者经过一个简单的线性层
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]

            elif "lidar" in key:
                # 雷达数据 (3, 60) 使用 1D 卷积提取时空特征
                n_input_channels = subspace.shape[0] # 这里是 5
                # print(subspace.shape)
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 16, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # 推断 CNN 输出的维度大小
                with th.no_grad():
                    dummy_input = th.as_tensor(subspace.sample()[None]).float()
                    n_flatten = cnn(dummy_input).shape[1]
                    # print(n_flatten)

                # 将 CNN 提取的高维特征压缩成紧凑的向量
                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())

                # 雷达处理模块
                radar_module = nn.Sequential(cnn, linear)
                extractors[key] = radar_module
                self.lidar_modules[key] = radar_module  # 保存引用
                total_concat_size += 64 # 两个雷达加起来就是 128 维

        self.extractors = nn.ModuleDict(extractors)
        # 更新最终输出的特征总维度 (例如: 15运动学 + 64雷达A + 64雷达B = 143)
        self._features_dim = total_concat_size

        # 加载预训练雷达编码器权重
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_encoder(pretrained_path)
            print(f"✅ 成功加载预训练雷达编码器权重: {pretrained_path}")

        # 冻结雷达编码器参数
        if freeze_encoder:
            self.freeze_encoder()
            print("✅ 雷达编码器参数已冻结")

    def load_pretrained_encoder(self, pretrained_path):
        """加载预训练的雷达编码器权重"""
        try:
            checkpoint = th.load(pretrained_path, map_location='cpu')
            saved_tensors = list(checkpoint.values())
            
            for lidar_name, radar_module in self.lidar_modules.items():
                current_params = list(radar_module.parameters())
                
                if len(saved_tensors) != len(current_params):
                    print(f"⚠️ {lidar_name} 参数数量({len(current_params)})与预训练权重({len(saved_tensors)})不匹配！")
                    continue
                    
                with th.no_grad():
                    for current_p, saved_p in zip(current_params, saved_tensors):
                        current_p.copy_(saved_p.view_as(current_p))
                        
                print(f"✅ 成功将预训练权重注入到 {lidar_name}")
                
        except Exception as e:
            print(f"⚠️ 加载预训练权重致命失败: {e}")


    def freeze_encoder(self):
        """冻结雷达编码器参数"""
        for lidar_name, radar_module in self.lidar_modules.items():
            for param in radar_module.parameters():
                param.requires_grad = False
            print(f"❄️ 已彻底冻结模块: {lidar_name}")

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
    ensure_output_dirs()

    TARGET_BUFFER_SIZE = 4096
    dynamic_batch_size = 256

    dynamic_n_steps = TARGET_BUFFER_SIZE // NUM_ENVS

    actual_buffer_size = NUM_ENVS * dynamic_n_steps

    assert actual_buffer_size % dynamic_batch_size == 0, (
        f"rollout buffer={actual_buffer_size} 不能被 "
        f"batch_size={dynamic_batch_size} 整除"
    )

    print(f"并行环境数: {NUM_ENVS}")
    print(f"每个环境采样步数: {dynamic_n_steps}")
    print(f"Rollout Buffer: {actual_buffer_size}")
    print(f"Batch Size: {dynamic_batch_size}")
    print(f"每轮 Minibatch 数: {actual_buffer_size // dynamic_batch_size}")

    



    # 预训练模型路径
    pretrained_encoder_path = str(PRETRAINED_ENCODER_PATH)

    # 根据训练阶段配置雷达参数
    if START_STAGE == 1:
        # 阶段一：基础导航，不冻结雷达参数
        use_pretrained = False  # 使用预训练权重
        freeze_encoder = False 
    else:
        # 阶段二：避障训练，冻结雷达参数
        use_pretrained = False  # 使用预训练自编码器权重
        freeze_encoder = False  # 冻结雷达编码器参数

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs={
            "pretrained_path": pretrained_encoder_path if use_pretrained else None,
            "freeze_encoder": freeze_encoder
        },
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    tb_reward_callback = TensorboardRewardCallback()
    # ------------------------------------------------------------------
    # 🏁 阶段一：基础导航训练 (无障碍)
    # ------------------------------------------------------------------
    if START_STAGE == 1:
        print(f"\n[运行中] >>> 阶段 1: 基础导航训练 (并行环境: {NUM_ENVS})")
        
        # 安全检查：禁止阶段一加载阶段二的最终模型
        if LOAD_MODEL_NAME == MODEL_FINAL_NAME:
            print("❌ 错误: 阶段 1 不能使用 ppo2_final_leader 模型进行训练！")
            exit()

        env = create_vectorized_env(NUM_ENVS, enable_obstacles=False)

        if START_FROM_SCRATCH:
            print("正在从零开始创建模型...")
            model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                        learning_rate=3e-4, batch_size=dynamic_batch_size, n_steps=dynamic_n_steps,
                        tensorboard_log=str(TB_STAGE1_DIR),
                        device="cuda")
            try:
                total_params = sum(p.numel() for p in model.policy.parameters())
                trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
                print(f'Policy class: {type(model.policy).__name__}')
                print(f'Total params: {total_params:,}, Trainable params: {trainable_params:,}')
            except Exception as e:
                print('无法统计模型参数:', e)
        else:
            print(f"载入模型继续训练: {LOAD_MODEL_NAME}")
            model = PPO.load(model_load_path(LOAD_MODEL_NAME), env=env, batch_size=dynamic_batch_size, n_steps=dynamic_n_steps, tensorboard_log=str(TB_STAGE1_DIR), device="cuda")


        model.learn(total_timesteps=TIMESTEPS_STAGE1, tb_log_name="stage1_nav", callback=tb_reward_callback)
        model.save(model_save_base(MODEL_STAGE1_NAME))
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
            print("警告: 正在完全从零开始进行阶段二训练（不推荐）。")
            model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs,batch_size=dynamic_batch_size, n_steps=dynamic_n_steps, learning_rate=3e-4,
                        tensorboard_log=str(TB_STAGE2_DIR), device="cuda")
        else:
            print(f"正在加载指定的模型作为起点: {LOAD_MODEL_NAME}")
            # 加载模型并重置一些参数 (例如增加熵系数 ent_coef 鼓励避障探索)
            model = PPO.load(model_load_path(LOAD_MODEL_NAME), env=env, 
                             ent_coef=0.001, 
                             batch_size=dynamic_batch_size, n_steps=dynamic_n_steps,
                             learning_rate=3e-5,
                             n_epochs=5,
                             clip_range=0.1,
                             tensorboard_log=str(TB_STAGE2_DIR), device="cuda")

        # 设置自动保存回调
        save_freq = max(1000000 // NUM_ENVS, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq, 
            save_path=str(CHECKPOINTS_DIR),
            name_prefix=f'stage2_from_{LOAD_MODEL_NAME}'
        )

        model.learn(
            total_timesteps=TIMESTEPS_STAGE2,
            callback=[tb_reward_callback, checkpoint_callback],
            tb_log_name=f"stage2_obs_from_{LOAD_MODEL_NAME}"
        )

        model.save(model_save_base(MODEL_FINAL_NAME))
        env.close()
        print("✅ 阶段 2 训练结束，最终模型已保存。")

    else:
        print("❌ 错误: START_STAGE 只能设为 1 或 2")