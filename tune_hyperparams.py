import optuna
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# 导入你自己的环境类 (请根据你的实际文件名修改)
from POMDP_env import FormationPyBulletEnv 

def objective(trial):
    """
    Optuna 的核心优化目标函数。
    每一次 trial（试验），Optuna 都会尝试一组新的参数组合。
    """
    # ==========================================
    # 1. 采样环境的“奖励权重”超参数 (Environment Params)
    # ==========================================
    # 让 Optuna 帮你寻找最完美的惩罚力度，防止小车原地转圈或绕大弯！
    env_kwargs = {
        # 采样角速度变化率惩罚 (范围 0.01 到 0.5)
        'k_accel_angular': trial.suggest_float('k_accel_angular', 0.01, 0.5, log=True),
        # 采样速度对齐惩罚 (范围 0.01 到 0.2)
        'k_vel_align': trial.suggest_float('k_vel_align', 0.01, 0.2, log=True),
        # 如果你加了 CTE 直线惩罚，也可以放进来调
        # 'k_cte': trial.suggest_float('k_cte', 0.01, 0.1),
        
        # 保持你的默认设置
        'n_followers': 7, 
        'enable_obstacles': False
    }

    # ==========================================
    # 2. 采样 PPO 算法的超参数 (Algorithm Params)
    # ==========================================
    ppo_kwargs = {
        # 学习率通常在 1e-5 到 1e-3 之间
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        # 每次更新的步数，SB3 建议是 batch_size 的倍数
        'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
        # 批次大小
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        # 探索熵系数 (决定小车敢不敢尝试新动作，太大会乱抖，太小会陷入局部最优)
        'ent_coef': trial.suggest_float('ent_coef', 0.00001, 0.01, log=True),
        # 折扣因子
        'gamma': trial.suggest_categorical('gamma', [0.95, 0.98, 0.99, 0.995]),
    }

    # ==========================================
    # 3. 实例化环境与模型
    # ==========================================
    # 使用 env_kwargs 动态创建环境
    # 注意：如果使用多进程，可以把 make_vec_env 的 n_envs 设大一点以加速训练
    env = make_vec_env(FormationPyBulletEnv, n_envs=4, env_kwargs=env_kwargs)
    
    model = PPO("MlpPolicy", env, verbose=0, **ppo_kwargs)

    # ==========================================
    # 4. 开始一次短途试训 (Trial Training)
    # ==========================================
    # 不需要训练到完全收敛，给个 5 万到 10 万步，
    # 足够 Optuna 看出这组参数有没有“爆雷”或者“跑偏”的趋势了
    TRAIN_TIMESTEPS = 80000 
    
    try:
        model.learn(total_timesteps=TRAIN_TIMESTEPS)
    except Exception as e:
        # 如果训练过程中因为物理引擎穿模或者数值爆炸报错，直接终止这次 trial 并给极低分
        print(f"Trial failed due to error: {e}")
        return -10000.0 

    # ==========================================
    # 5. 独立评估这组参数的真正实力
    # ==========================================
    # 创建一个纯净的测试环境进行评估 (不要复用训练环境)
    eval_env = FormationPyBulletEnv(**env_kwargs)
    
    # 跑 5 个回合看看平均能拿多少分
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    
    eval_env.close()
    env.close()

    # Optuna 的目标就是最大化这个返回值 (Mean Reward)
    return mean_reward


if __name__ == "__main__":
    print("🚀 开始全自动超参数搜索 (Optuna)...")
    
    # 创建一个优化研究，方向是 "maximize" (分数越高越好)
    study = optuna.create_study(direction="maximize", study_name="Formation_RL_Tuning")
    
    # 开始搜索！n_trials 是尝试的组合次数。可以设为 30-50 次，放在电脑上跑一晚上。
    study.optimize(objective, n_trials=50)

    # ==========================================
    # 6. 打印最终的“武林秘籍”
    # ==========================================
    print("\n🎉 搜索完成！")
    print("🏆 最强参数组合 (Best Trial):")
    best_trial = study.best_trial
    print(f"  --> 最高平均得分: {best_trial.value:.2f}")
    
    print("\n🔧 请把以下参数填回你的主代码中：")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")