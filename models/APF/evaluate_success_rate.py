import gymnasium as gym
import numpy as np
import os
import sys
import time

# 添加项目根目录到 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from APF_env import FormationPyBulletEnv 
from sb3_contrib import RecurrentPPO
from project_paths import MODEL_FINAL_NAME, ensure_output_dirs, model_load_path

def evaluate_model(num_episodes=100):

    env = FormationPyBulletEnv(render_mode=False, enable_obstacles=True)

    stats = {"Success": 0, "Crash_Obstacle": 0, "Crash_Team": 0, "Wall_Out": 0, "Timeout": 0, "Unknown": 0}
    
    # ==========================================
    # ⏱️ 新增：用于记录每一步的真实决策耗时
    # ==========================================
    step_times = []  
    success_steps = []
    formation_error_avgs = []
    formation_error_maxs = []

    start_time = time.time()
    
    try:
        for episode in range(num_episodes):
            try:
                obs, _ = env.reset()
            except Exception as e:
                print(f"❌ 第 {episode+1} 轮重置失败: {e}")
                continue
            
            done = False
            steps = 0
            # lstm_states = None
            while not done:
                try:
                    # ⏳ 记录单步开始时间 (使用更高精度的 perf_counter)
                    t0 = time.perf_counter()

                    # 2. 物理环境交互与跟随者计算耗时
                    obs, reward, terminated, truncated, info = env.step(None)
                    
                    # ⏳ 记录单步结束时间并计算差值
                    t1 = time.perf_counter()
                    step_times.append(t1 - t0)
                    
                    steps += 1
                    done = terminated or truncated
                    
                    if done:
                        reason = info.get('reason', 'Unknown')
                        # 收集 env 返回的回合级编队误差统计（如果有）
                        if isinstance(info, dict) and 'ep_formation_error_avg' in info:
                            formation_error_avgs.append(info['ep_formation_error_avg'])
                        if isinstance(info, dict) and 'ep_formation_error_max' in info:
                            formation_error_maxs.append(info['ep_formation_error_max'])
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
                except Exception as e:
                    print(f"⚠️ 第 {episode+1} 轮、第 {steps+1} 步出错: {e}")
                    done = True
                        
            if (episode + 1) % 10 == 0:
                print(f"已完成: {episode + 1}/{num_episodes} 轮...")
    finally:
        env.close()
    
    # === 计算测试结果 ===
    total_time = time.time() - start_time
    success_rate = (stats["Success"] / num_episodes) * 100
    
    # === 🚀 计算核心指标：单步耗时与控制频率 ===
    if step_times:
        avg_step_time = np.mean(step_times)  # 平均每步耗时 (秒)
        max_step_time = np.max(step_times)   # 最慢的一步耗时 (秒)
        control_freq = 1.0 / avg_step_time if avg_step_time > 0 else 0.0 # 频率 (Hz)
    else:
        avg_step_time = 0.0
        max_step_time = 0.0
        control_freq = 0.0

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
    if formation_error_avgs:
        print(f"📐 回合平均编队误差: {np.mean(formation_error_avgs):.4f}")
        print(f"📐 回合最大编队误差: {np.mean(formation_error_maxs):.4f}")
    
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
    evaluate_model(100)