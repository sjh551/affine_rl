import numpy as np
import os
import sys
# Add parent directory to Python path to import pybullet_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 请把这里的导入替换为你自己实际的环境类
from POMDP_env import FormationPyBulletEnv 
from project_paths import DATASET_100K_PATH

def collect_lidar_dataset(total_frames=100000, save_path="lidar_dataset.npy"):
    """
    运行随机策略（带一定向前偏置），采集雷达数据用于自编码器预训练
    """
    print(f" 开始采集雷达数据，目标数量: {total_frames} 组 (包含两辆Leader的数据)...")
    
    env = FormationPyBulletEnv(render_mode=False, enable_obstacles=True)
    obs, _ = env.reset()
    
    collected_data = []
    
    # 因为每次 step 我们可以同时拿到 A 和 B 两辆车的雷达数据
    # 所以实际只需要跑 total_frames / 2 步
    target_steps = total_frames // 2 
    
    for step in range(target_steps):
        # 1. 生成随机动作
        action = env.action_space.sample()
        
        # 💡 极其重要的技巧：给动作加一个“向前”的偏置！
        # 如果完全随机，小车大概率在原地打转，扫不到丰富的障碍物特征。
        # 假设 action[0] 和 action[2] 是两辆车的纵向速度，我们强行让它们大概率往前开
        action[0] = abs(action[0]) * 0.8 + 0.2  # 保证有一定的基础前进速度
        action[2] = abs(action[2]) * 0.8 + 0.2 
        
        # 2. 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. 提取 3 帧雷达数据并保存 
        # (确保 obs["lidar_A"] 的 shape 是 (3, 180))
        if "lidar_A" in obs:
            collected_data.append(obs["lidar_A"])
        if "lidar_B" in obs:
            collected_data.append(obs["lidar_B"])
            
        # 4. 如果撞墙了或者超时了，直接重置环境继续跑
        if terminated or truncated:
            obs, _ = env.reset()
            
        # 打印进度条
        if (step + 1) % 5000 == 0:
            current_collected = (step + 1) * 2
            print(f"✅ 进度: 已采集 {current_collected} / {total_frames} 组数据...")
            
    env.close()
    
    # 5. 转化为 Numpy 矩阵并保存
    # 最终的 dataset 形状应该是 (50000, 3, 180)
    dataset = np.array(collected_data, dtype=np.float32)
    print(f"采集完成！数据集最终形状: {dataset.shape}")
    
    np.save(save_path, dataset)
    print(f"数据已成功保存至当前目录下的: {save_path}")

if __name__ == "__main__":
    # 先采集 100,000 组数据
    collect_lidar_dataset(total_frames=100000, save_path=str(DATASET_100K_PATH))