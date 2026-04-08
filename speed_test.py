import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from pybullet_env import FormationPyBulletEnv # 替换为你实际的环境类名

def measure_tracking_error(test_speed, test_steps=300):
    """
    让领导者以固定速度前进，分别记录两个跟随者的偏离误差
    """
    print(f"🚀 正在测试领导者速度: {test_speed} m/s ...")
    
    env = FormationPyBulletEnv(render_mode=False, enable_obstacles=False)
    obs, _ = env.reset()
    
    error_f1_history = []
    error_f2_history = []
    
    for step in range(test_steps):
        # 强行让编队以 test_speed 的速度在 X 轴前进
        action = np.array([test_speed / env.max_vel, 0.0, test_speed, np.random.choice([0.2, -0.2])]) 
        
        # 1. 先执行环境步进，让物理引擎把所有车推向新位置
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 2. 提取当前物理引擎中【所有】机器人的最新位置
        # (必须在 step 之后获取，否则会出现 1 帧的滞后误差放大！)
        all_pos = [np.array(p.getBasePositionAndOrientation(rid)[0][:2]) for rid in env.robot_ids]
        
        # 3. 提取当前实际的 Leader 位置与朝向
        pos_A, quat_A = p.getBasePositionAndOrientation(env.robot_ids[0])
        _, _, yaw_A = p.getEulerFromQuaternion(quat_A)
        pos_B, quat_B = p.getBasePositionAndOrientation(env.robot_ids[1])
        _, _, yaw_B = p.getEulerFromQuaternion(quat_B)
        
        pA = np.array([pos_A[0], pos_A[1], yaw_A])
        pB = np.array([pos_B[0], pos_B[1], yaw_B])
        p_leader_flat = np.concatenate([pA, pB])
        
        # 4. 拉普拉斯数学推导：计算所有 Follower 的【理想无延迟位置】
        p_follower_ideal = -env.M_ff_inv @ env.M_fl @ p_leader_flat
        
        # 5. 分别计算两个跟随者的真实误差
        # 跟随者 1 (Follower A，在 all_pos 中索引为 2)
        ideal_pos_f1 = np.array([p_follower_ideal[0*3], p_follower_ideal[0*3 + 1]])
        actual_pos_f1 = all_pos[2][:2]
        err_f1 = np.linalg.norm(actual_pos_f1 - ideal_pos_f1)
        
        # 跟随者 2 (Follower B，在 all_pos 中索引为 3)
        ideal_pos_f2 = np.array([p_follower_ideal[1*3], p_follower_ideal[1*3 + 1]])
        actual_pos_f2 = all_pos[3][:2]
        err_f2 = np.linalg.norm(actual_pos_f2 - ideal_pos_f2)
        
        # 记录到各自的历史列表中
        error_f1_history.append(err_f1)
        error_f2_history.append(err_f2)
        
        if terminated or truncated:
            break
            
    env.close()
    return error_f1_history, error_f2_history

def plot_speed_comparison():
    # 测试三个典型的速度梯度
    test_speeds = [0.4, 0.7, 1.0]
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(test_speeds)))
    
    plt.figure(figsize=(10, 6))
    
    for speed, color in zip(test_speeds, colors):
        err_f1, err_f2 = measure_tracking_error(speed)
        
        # 将 step 转换为大致的时间 (假设控制频率为 10Hz)
        time_axis = np.arange(len(err_f1)) * 0.1 
        
        # 画出跟随者 1 (实线)
        plt.plot(time_axis, err_f1, label=f'Follower 1 ({speed} m/s)', 
                 color=color, linestyle='-', linewidth=2)
        
        # 画出跟随者 2 (虚线，稍微加一点透明度防止重叠看不清)
        plt.plot(time_axis, err_f2, label=f'Follower 2 ({speed} m/s)', 
                 color=color, linestyle='--', linewidth=2, alpha=0.8)

    # 设定一条“安全容忍基准线” (假设偏离 0.5 米是极限)
    plt.axhline(y=0.2, color='black', linestyle=':', label='Safety Threshold (0.5m)')

    # 图表美化设置
    plt.title('Follower 1 & 2 Tracking Error at Different Leader Speeds', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Tracking Error (Meters)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 将图例放在图外侧或最佳位置，防止挡住曲线
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    
    # 保存并显示图片
    plt.savefig('tracking_error_analysis_f1_f2.png', dpi=300, bbox_inches='tight')
    print("\n✅ 测试完成！图表已保存为 tracking_error_analysis_f1_f2.png")
    plt.show()

if __name__ == "__main__":
    plot_speed_comparison()