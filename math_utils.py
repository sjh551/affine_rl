import numpy as np

# 直接从你的 main_reproduce.py 复刻过来的核心逻辑
def compute_W_matrix(gi, gj, gk, theta=0):
    """
    计算局部权重矩阵 (复用你代码中的核心逻辑)
    输入: gi, gj, gk 是标称位置 (3维: x, y, phi)
    输出: W_jk, W_ki
    """
    # 1. 构造旋转矩阵 (Scaling Direction)
    c, s = np.cos(theta), np.sin(theta)
    R_theta = np.array([[c, -s], [s, c]])
    
    # 2. 构造投影矩阵 Theta (3x3)
    # 对应论文中的 Θ = diag(R, 1)
    # 注意：你的代码里可能是分开处理位置和角度的，这里为了通用，我们组装成 3x3
    Theta = np.eye(3)
    Theta[:2, :2] = R_theta 

    # 3. 计算相对标称状态
    g_jk = gj - gk
    g_ki = gk - gi # 注意方向，你的代码逻辑可能是 i->k 或 k->i，需保持一致
    # 修正：根据 formation_graph.py 的逻辑：
    # W_jk 对应 edge (j,k), W_ki 对应 edge (i,k)
    # 约束方程: W_jk(gi - gk) + W_ki(gj - gk) = 0
    # 所以我们需要的是 g_ik 和 g_jk 的系数
    
    # p_jk_proj = R^T * (pj - pk)
    # p_ki_proj = R^T * (pk - pi)
    p_j = gj[:2]
    p_k = gk[:2]
    p_i = gi[:2]
    
    p_jk_proj = R_theta.T @ (p_j - p_k)
    p_ki_proj = R_theta.T @ (p_k - p_i) # 注意这里是 k - i
    
    # 角度差
    phi_jk = gj[2] - gk[2]
    phi_ki = gk[2] - gi[2]
    
    # 4. 构造对角加权矩阵 w (小写)
    # w_jk = diag(p_jk_x, p_jk_y, phi_jk)
    w_jk = np.diag([p_jk_proj[0], p_jk_proj[1], phi_jk])
    w_ki = np.diag([p_ki_proj[0], p_ki_proj[1], phi_ki])
    
    # 5. 计算最终矩阵 W (大写) = w * Theta^T
    W_jk = w_jk @ Theta.T
    W_ki = w_ki @ Theta.T
    
    return W_jk, W_ki

def get_agent_nominal_config():
    """
    为了简单，我们定义一个标准的等腰三角形
    """
    # 格式: [x, y, phi]
    g_nom = {
        # 让 Leader A 稍微靠前一点 (0.2), Leader B 稍微靠后一点 (-0.2)
        # 这样 X 轴差值是 0.4 (非零)，矩阵就可逆了！
        0: np.array([ 0.2,  1.0, 0.0]),   # Leader A (前左)
        1: np.array([-0.2, -1.0, 0.0]),   # Leader B (后右)
        2: np.array([-1.5,  0.0, 0.0])    # Follower (后面)
    }
    return g_nom

def process_lidar_to_vfh(lidar_frames, num_bins=36, max_range=6.0):
    """
    将原始雷达射线转换为 VFH 障碍物密度直方图
    lidar_frames: shape (1, 180) 的 numpy 数组
    num_bins: 划分的扇区数量 (180度36份，每份5度)
    max_range: 雷达的最大探测距离
    """
    channels, num_rays = lidar_frames.shape
    rays_per_bin = num_rays // num_bins  # 180 // 36 = 5 根射线对应一个扇区
    
    # 初始化 VFH 密度图 (1, 36)
    vfh_density = np.zeros((channels, num_bins), dtype=np.float32)
    
    for c in range(channels):
        for b in range(num_bins):
            # 取出当前扇区内的 10 根射线
            start_idx = b * rays_per_bin
            end_idx = start_idx + rays_per_bin
            sector_rays = lidar_frames[c, start_idx:end_idx]
            
            # 🌟 核心 1：将“距离”转化为“密度（危险度）”
            # 假设 sector_rays 存的是物理距离。距离越近，密度越大；超过 max_range 密度为 0
            # 这里的公式：danger = max(0, (max_range - dist) / max_range)
            # 例如：距离 0m -> 密度 1.0 (极度危险)；距离 5m -> 密度 0.0 (绝对安全)
            density_vals = np.clip((max_range - sector_rays) / max_range, 0.0, 1.0)
            
            # 🌟 核心 2：池化策略 (Pooling)
            # 绝对不要用平均值 (np.mean)！如果 9 根线是空的，1 根线扫到一根细柱子，平均值会被拉低，导致撞车。
            # 必须用最大值 (np.max)，保证只要扇区里有障碍物，这个扇区就标为高危！
            vfh_density[c, b] = np.max(density_vals)
            
    return vfh_density