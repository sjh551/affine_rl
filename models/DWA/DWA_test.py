import numpy as np
import pybullet as p
from DWA_env import FormationPyBulletEnv
import time
from DWA import DWA, Config

def get_dwa_action(env, dwa, robot_id, config_id):
    # --- A. 獲取 DWA 所需的世界座標系狀態 ---
    # Leader A
    pos_A, quat_A = p.getBasePositionAndOrientation(env.robot_ids[robot_id])
    yaw_A = p.getEulerFromQuaternion(quat_A)[2]
    vel_world_A, _ = p.getBaseVelocity(env.robot_ids[robot_id])
    state_A = np.array([pos_A[0], pos_A[1], yaw_A, vel_world_A[0], vel_world_A[1]])

    # 獲取障礙物
    obs_list = [[p.getBasePositionAndOrientation(oid)[0][0], p.getBasePositionAndOrientation(oid)[0][1]] for oid in env.obstacles]
    obs_np = np.array(obs_list) if obs_list else np.empty((0, 2))

    # --- B. 調用 DWA 規劃 (目標點 = 隊形中心目標 + 機器人標稱偏移) ---
    best_u, _ = dwa.plan(state_A, env.target_pos[:2] + env.g_nom[robot_id][:2], obs_np, config_id)
    
    return best_u

# 修正後的評估函數
def evaluate_dwa(num_episodes=100):
    print(f"準備進行 {num_episodes} 輪 DWA 批量測試...")
    
    # 在循環外初始化一次環境和控制器
    env = FormationPyBulletEnv(render_mode=False, enable_obstacles=True)
    cfg = Config()
    dwa = DWA(cfg)

    stats = {"Success": 0, "Crash_Obstacle": 0, "Crash_Team": 0, "Wall_Out": 0, "Timeout": 0, "Unknown": 0}
    step_times = []  
    success_steps = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        current_config = env.config_id # 獲取環境當前的配置 ID
        done = False
        steps = 0
        
        while not done:
            full_action_list = []
              
            # --- 為所有機器人計算動作 ---
            for i in range(env.num_robots):
                # 1. 獲取世界系速度
                u_world = get_dwa_action(env, dwa, i, current_config)
                
                # 2. 座標系轉換 (World -> Target Frame)
                vec_to_goal = env.virtual_target_pos[:2] - env._get_center_pos()[:2]
                target_yaw = np.arctan2(vec_to_goal[1], vec_to_goal[0])
                c, s = np.cos(-target_yaw), np.sin(-target_yaw)
                R = np.array([[c, -s], [s, c]])
                
                u_target = R @ np.array(u_world)
                
                # 3. 歸一化並加入列表
                full_action_list.extend([u_target[0] / env.max_vel, u_target[1] / env.max_vel])

            # 轉化為 numpy 數組
            full_action = np.array(full_action_list, dtype=np.float32)
            
            # 執行環境步進
            t0 = time.perf_counter()
            
            obs, reward, terminated, truncated, info = env.step(full_action)
            t1 = time.perf_counter()
            step_times.append(t1 - t0)

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
                elif "Timeout" in reason:
                    stats["Timeout"] += 1
                else:
                    stats["Unknown"] += 1
                    
        if (episode + 1) % 10 == 0:
            print(f"已完成: {episode + 1}/{num_episodes} 輪...")

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

def watch_dwa_performance():
    """
    可視化觀察 DWA 控制所有機器人的運行效果
    """
    print("🎬 啟動 DWA 可視化測試...")
    
    # 1. 必須將 render_mode 設為 True 才能看到畫面
    env = FormationPyBulletEnv(render_mode=True, enable_obstacles=True)
    cfg = Config()
    dwa = DWA(cfg)

    obs, _ = env.reset()
    done = False
    
    # 獲取當前環境的配置 ID（兼容你剛加的動態障礙物判斷）
    current_config = getattr(env, 'config_id', 1) 
    
    steps = 0
    while not done:
        actions_to_send = []
        
        # 提前計算座標轉換矩陣 (World -> Target Frame)
        vec_to_goal = env.virtual_target_pos[:2] - env._get_center_pos()[:2]
        target_yaw = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        c, s = np.cos(-target_yaw), np.sin(-target_yaw)
        R = np.array([[c, -s], [s, c]])
        
        # 獲取點狀障礙物坐標
        obs_list = []
        for oid in env.obstacles:
            o_pos, _ = p.getBasePositionAndOrientation(oid)
            obs_list.append([o_pos[0], o_pos[1]])
        obs_np = np.array(obs_list) if obs_list else np.empty((0, 2))

        # 2. 為每一台機器人計算 DWA 動作
        for i in range(env.num_robots):
            # 獲取狀態 [x, y, yaw, vx, vy]
            pos, quat = p.getBasePositionAndOrientation(env.robot_ids[i])
            yaw = p.getEulerFromQuaternion(quat)[2]
            vel, _ = p.getBaseVelocity(env.robot_ids[i])
            state = np.array([pos[0], pos[1], yaw, vel[0], vel[1]])

            # 目標點 = 總目標點 + 該機器人的標稱編隊偏移
            personal_goal = env.target_pos[:2] + env.g_nom[i][:2]

            # 調用 DWA
            # 注意：這裡傳入了 current_config 以適配你修改後的 calc_obstacle_cost
            u_world, _ = dwa.plan(state, personal_goal, obs_np, current_config)

            # 座標轉換與歸一化
            u_target = R @ np.array(u_world)
            actions_to_send.extend([u_target[0] / env.max_vel, u_target[1] / env.max_vel])

        # 3. 執行環境步進
        full_action = np.array(actions_to_send, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(full_action)
        
        done = terminated or truncated
        steps += 1
        
        # 4. ⚠️ 關鍵：加入延遲，否則畫面會一閃而過
        # env.step 裡面有 24 次 p.stepSimulation()，通常跑得極快
        time.sleep(0.05) 

        # 實時打印進度
        if steps % 20 == 0:
            center = env._get_center_pos()
            dist = np.linalg.norm(env.target_pos[:2] - center[:2])
            print(f"📍 運行中 | 步數: {steps} | 距離目標: {dist:.2f}m")

    # 測試結束
    reason = info.get('reason', '未知')
    print(f"\n✅ 測試結束！")
    print(f"🛑 終止原因: {reason}")
    print(f"⏱️ 總共花費步數: {steps}")
    
    # 結束後停留幾秒鐘讓你看清最終狀態，再關閉窗口
    time.sleep(3)
    env.close()


if __name__ == "__main__":
    evaluate_dwa(100)
    # watch_dwa_performance()