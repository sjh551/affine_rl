import numpy as np
import time
from ORCA_env import FormationPyBulletEnv 

def evaluate_orca_baseline(num_episodes=100):
    print("=" * 45)
    print(f"🚀 准备进行 {num_episodes} 轮 ORCA 算法无渲染批量测试...")
    print("=" * 45)
    
    # 强制关闭渲染以最高速度运行评估
    env = FormationPyBulletEnv(render_mode=False, enable_obstacles=True)

    stats = {"Success": 0, "Crash_Obstacle": 0, "Crash_Team": 0, "Wall_Out": 0, "Timeout": 0, "Unknown": 0}
    
    # ==========================================
    # ⏱️ 用于记录每一步的真实决策耗时 (纯 C++ 求解耗时)
    # ==========================================
    step_times = []  
    success_steps = []

    start_time = time.time()
    
    for episode in range(num_episodes):
        env.reset()
        done = False
        steps = 0
        
        while not done:
            # ⏳ 记录单步开始时间 (使用更高精度的 perf_counter)
            t0 = time.perf_counter()
            
            # 1. 物理环境交互与 ORCA C++ 底层求解
            # 不需要传入动作，环境内部的 pyrvo 会自动接管
            res = env.step(action=None)
            
            # ⏳ 记录单步结束时间并计算差值
            t1 = time.perf_counter()
            step_times.append(t1 - t0)
            
            # 处理不同 gym 版本的 step() 返回值
            if isinstance(res, tuple) and len(res) == 5:
                obs, reward, terminated, truncated, info = res
                try:
                    done = bool(terminated or truncated)
                except Exception:
                    done = np.logical_or(terminated, truncated)
            else:
                obs, reward, done, info = res
                
            steps += 1
            
            # 记录结束原因
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
    print(" 📊 ORCA 基准算法评估报告 (Baseline Report) ")
    print("="*45)
    print(f"总测试轮数: {num_episodes} | 现实总耗时: {total_time:.2f} 秒")
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
    print(" ⚡ 系统性能指标 (Computation Performance) ")
    print("-" * 45)
    print(f"⏱️ 单步平均耗时: {avg_step_time * 1000:.2f} ms")
    print(f"⏱️ 极限最慢单步: {max_step_time * 1000:.2f} ms")
    print(f"⚙️ 实际解算频率: ** {control_freq:.1f} Hz **")
    
    if control_freq >= 10.0:
        print("🟩 [性能极佳] 纯 C++ 底层求解极快，满足对比要求！")
    else:
        print("🟥 [性能异常] 解算频率过低，请检查是否有进程阻塞。")
    print("="*45)

if __name__ == "__main__":
    evaluate_orca_baseline(100)