import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from pybullet_env import FormationPyBulletEnv 
from sb3_contrib import RecurrentPPO
import time
from project_paths import MODEL_FINAL_NAME, ensure_output_dirs, model_load_path

def evaluate_model(model_path, num_episodes=100):
    print(f"正在加载模型: {model_path}")
    print(f"准备进行 {num_episodes} 轮无渲染批量测试...")
    
    env = FormationPyBulletEnv(render_mode=False, enable_obstacles=True)
    model = PPO.load(model_path)

    stats = {"Success": 0, "Crash_Obstacle": 0, "Crash_Team": 0, "Wall_Out": 0, "Timeout": 0, "Unknown": 0}
    
    # ==========================================
    # ⏱️ 新增：用于记录每一步的真实决策耗时
    # ==========================================
    step_times = []  
    success_steps = []

    start_time = time.time()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        # lstm_states = None
        while not done:
            # ⏳ 记录单步开始时间 (使用更高精度的 perf_counter)
            t0 = time.perf_counter()
            
            # 1. 神经网络推理耗时
            action, _ = model.predict(obs, deterministic=True)
            # 2. 物理环境交互与跟随者计算耗时
            obs, reward, terminated, truncated, info = env.step(action)
            
            # ⏳ 记录单步结束时间并计算差值
            t1 = time.perf_counter()
            step_times.append(t1 - t0)
            
            steps += 1
            done = terminated or truncated
            
            if done:
                reason = info.get('reason', 'Unknown')
                if "Success" in reason:
                    stats["Success"] += 1
                    success_steps.append(steps)
                elif "Crash_Obstacle" in reason:
                    stats["Crash_Obstacle"] += 1
                elif "Crash_Team" in reason:
                    stats["Crash_Team"] += 1
                elif "Wall_Out" in reason:
                    stats["Wall_Out"] += 1
                elif "Timeout" in reason or info.get("TimeLimit.truncated", False):
                    stats["Timeout"] += 1
                else:
                    stats["Unknown"] += 1
                    
        if (episode + 1) % 10 == 0:
            print(f"已完成: {episode + 1}/{num_episodes} 轮...")

    env.close()
    
    # === 计算测试结果 ===
    total_time = time.time() - start_time
    success_rate = (stats["Success"] / num_episodes) * 100
    
    # === 🚀 计算核心指标：单步耗时与控制频率 ===
    avg_step_time = np.mean(step_times)  # 平均每步耗时 (秒)
    max_step_time = np.max(step_times)   # 最慢的一步耗时 (秒)
    control_freq = 1.0 / avg_step_time if avg_step_time > 0 else 0.0 # 频率 (Hz)

    print("\n" + "="*45)
    print(" 📊 毕设评估报告 (Evaluation Report) ")
    print("="*45)
    print(f"总测试轮数: {num_episodes} | 总耗时: {total_time:.2f} 秒")
    print("-" * 45)
    print(f"✅ 成功到达 (Success): {stats['Success']} 次")
    print(f"💥 撞击障碍 (Obstacle): {stats['Crash_Obstacle']} 次")
    print(f"💥 团队互撞 (Team): {stats['Crash_Team']} 次")
    print(f"🚧 撞击边界 (Wall_Out): {stats['Wall_Out']} 次")
    print(f"⏱️ 超时未完 (Timeout): {stats['Timeout']} 次")
    print("-" * 45)
    print(f"🏆 综合成功率: ** {success_rate:.1f}% **")
    if success_steps:
        print(f"🎯 成功回合平均步数: {np.mean(success_steps):.1f} 步")
    
    print("-" * 45)
    print(" ⚡ 系统性能指标 (Sim-to-Real Readiness) ")
    print("-" * 45)
    print(f"⏱️ 单步平均耗时: {avg_step_time * 1000:.2f} ms")
    print(f"⏱️ 极限最慢单步: {max_step_time * 1000:.2f} ms")
    print(f"⚙️ 实际控制频率: ** {control_freq:.1f} Hz **")
    
    if control_freq >= 10.0:
        print("🟩 [测试通过] 频率 >= 10 Hz，满足毕设任务书要求，可支持实物迁移！")
    else:
        print("🟥 [测试未达标] 频率 < 10 Hz，需要进一步优化代码性能。")
    print("="*45)

if __name__ == "__main__":
    ensure_output_dirs()
    MODEL_PATH = model_load_path(MODEL_FINAL_NAME)
    evaluate_model(MODEL_PATH, 100)