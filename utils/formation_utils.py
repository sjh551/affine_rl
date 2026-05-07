"""
Paper: Distributed Non-Uniform Scaling Control of Multi-Agent Formation via Matrix-Valued Constraints
"""
import numpy as np


# ==========================================
# 1. 标称构型 (Nominal Configuration)
# 来源: Section V 
# 格式: [x, y, phi]
# ==========================================
def get_nominal_configuration():
    g_nom = np.array([
        [-4, -2, np.pi/8],   # g1 (Leader)
        [ 2,  4, -np.pi/4],  # g2 (Leader)
        [ 0,  1, -np.pi/16], # g3
        [ 1, -1, np.pi/16],  # g4
        [-1,  0, 0.0],       # g5
        [-2, -3, -np.pi/16], # g6
        [-3, -4, np.pi/4],   # g7
        [ 3,  2, -np.pi/8],  # g8
        [ 4,  3, -np.pi/16]  # g9 
    ])
    return g_nom


# ==========================================
# 2. 图拓扑结构 (Sensing Graph)
# 来源: Fig. 2 
# ==========================================

def get_topology():
    """
    返回边列表 (Edge List)。
    注意：这是有向图还是无向图？
    论文定义是双向图 (Bidirectional) [cite: 69]，
    意味着如果 i 能测 k，k 也能测 i。
    这里我们要列出所有的双向边 (i, j)。
    """
    # 这里的数字对应索引 0-8 (即 g1-g9)
    # Path 1: 1-5-4-3-2 (红色路径)
    # Path 2: 1-7-6-5   (蓝色路径)
    # Path 3: 3-8-9-2   (黄色路径)
    # 注意：索引要比论文编号少 1 (0-based indexing)
    edges = [
        # Path 1
        (0, 4), (4, 3), (3, 2), (2, 1),
        # Path 2 (连接到 Path 1 的节点 0 和 4)
        (0, 6), (6, 5), (5, 4),
        # Path 3 (连接到 Path 1 的节点 2 和 1)
        (2, 7), (7, 8), (8, 1)
    ]
    # 因为是双向图，我们需要确保 (i,j) 和 (j,i) 都在
    # 这里我们只存唯一的连接对，后续构建 Laplacian 时再处理双向
    return edges


def get_subgraph_config():
    original_g_nom = get_nominal_configuration()
    original_edges = get_topology()

    # 你的映射字典 {新索引: 旧索引}
    leader_follower_dict = {
        0: 0, # 新Leader 1 -> 原 g1
        1: 4, # 新Leader 2 -> 原 g5
        2: 5, # 新Follower 1 -> 原 g6
        3: 6  # 新Follower 2 -> 原 g7
    }

    # 1. 提取并生成新的 g_nom
    new_g_nom = np.zeros((4, 3))
    for new_idx, old_idx in leader_follower_dict.items():
        new_g_nom[new_idx] = original_g_nom[old_idx]

    # 2. 构建反向映射 {旧索引: 新索引}，用于更新边
    reverse_dict = {v: k for k, v in leader_follower_dict.items()}

    # 3. 提取并映射新的 edges
    new_edges = []
    for u, v in original_edges:
        # 只有当边的两个端点都在我们的子集里，才保留这条边
        if u in reverse_dict and v in reverse_dict:
            new_edges.append((reverse_dict[u], reverse_dict[v]))

    return new_g_nom, new_edges

# ==========================================
# 3. 核心数学工具: W 矩阵计算
# 来源: Eq. (13) 和 (14) 
# ==========================================

def compute_W_matrix(g_nom_i, g_nom_j, g_nom_k, theta=0):
    """
    计算局部约束矩阵 W_jk 和 W_ki
    
    参数:
        g_nom_i, g_nom_j, g_nom_k: 标称状态 [x, y, phi]
        theta: 缩放方向 (默认为 0)
    返回:
        W_jk, W_ki: 3x3 矩阵
    """
    # 1. 计算相对标称状态 (Relative Nominal States)
    # 论文符号: p_jk = p_j - p_k
    p_jk_nom = g_nom_j[:2] - g_nom_k[:2]
    phi_jk_nom = g_nom_j[2] - g_nom_k[2]
    
    p_ki_nom = g_nom_k[:2] - g_nom_i[:2]
    phi_ki_nom = g_nom_k[2] - g_nom_i[2]

    # 2. 构造旋转矩阵 R(theta) 
    # 用于处理非均匀缩放的方向
    c, s = np.cos(theta), np.sin(theta)
    R_theta = np.array([[c, -s], [s, c]])
    
    # Theta 矩阵 (3x3) 
    # Theta = diag(R(theta)^T, 1) -> 文中定义的是 R(theta)^T 在左上角
    # 注意：Eq. 13 中 定义的是 Theta^T 在最后乘，这里我们先构造 Theta
    Theta = np.eye(3)

    Theta[:2, :2] = R_theta.T  # 注意这里是转置

    # 3. 构造 w_jk (小写 w) - 包含投影和缩放因子
    # Eq. 13 中 w_jk = diag(p_jk_theta, phi_jk)
    # 需要先将 p 投影到 theta 坐标系: p_theta = Theta^T * p
    # 实际上由于 Theta 结构，相当于 R(theta)^T * p
    
    p_jk_proj = R_theta.T @ p_jk_nom
    p_ki_proj = R_theta.T @ p_ki_nom
    
    # 构造对角矩阵 w
    w_jk = np.diag([p_jk_proj[0], p_jk_proj[1], phi_jk_nom])
    w_ki = np.diag([p_ki_proj[0], p_ki_proj[1], phi_ki_nom])
    
    # 4. 计算最终的 W 矩阵 (大写 W)
    # 根据 Eq. 13: W_jk = w_jk * Theta^T
    # 因为 Theta = diag(R^T, 1), 所以 Theta^T = diag(R, 1)
    # 但根据 python 矩阵乘法，我们要仔细看公式
    # W_jk = w_jk @ Theta  (这里需要仔细确认是否再次转置，
    # 论文 Eq.279 定义: W_jk = w_jk * Theta^T. 
    # 而 Theta 定义里左上角已经是 R^T. 所以 Theta^T 左上角是 R)
    
    W_jk = w_jk @ Theta.T
    W_ki = w_ki @ Theta.T
    
    return W_jk, W_ki

# ==========================================
# 新增：2 Leader + 2 Follower 正方形测试拓扑
# ==========================================

def get_square_configuration():
    """
    返回旋转45度的正方形配置。
    旋转是为了保证任意两个有连接的智能体，
    其 X 方向和 Y 方向的坐标差都不为 0 (避开拉普拉斯矩阵奇异陷阱)
    """
    y = 1.0
    x = 3 * y
    g_nom = np.array([
        [ x,  y,  0.0],       # 0: Leader 1 (顶点)
        [ y,  -x,  np.pi/2],   # 1: Leader 2 (右点)
        [-y,  x, -np.pi/2],   # 2: Follower 1 (左点)
        [ -x, -y,  np.pi]      # 3: Follower 2 (底点)
    ])
    # c, s = np.cos(np.pi/3), np.sin(np.pi/3)
    # R = np.zeros((3,3))
    # R[:2,:2] = np.array([
    #     [c, -s], 
    #     [s, c]
    # ]) 
    # R[2,2] = 1
    # g_nom = (R @ g_nom.T).T
    return g_nom

def get_square_topology():
    """
    构成一个长度为 2 的 DEP (Dual-Entry Path)。
    两个入口是 Leader (0和1)，内部节点是 Follower (2和3)。
    双向通信路径: Leader1(0) <-> Follower1(2) <-> Follower2(3) <-> Leader2(1)
    """
    edges = [
        (0, 2),
        (2, 3),
        (3, 1)
    ]
    return edges

def get_five_agent_configuration():
    """
    返回旋转45度的正方形配置。
    旋转是为了保证任意两个有连接的智能体，
    其 X 方向和 Y 方向的坐标差都不为 0 (避开拉普拉斯矩阵奇异陷阱)
    """
    y = 1.0
    x = 3 * y
    g_nom = np.array([
        [ x,  y,  0.0],       # 0: Leader 1 (顶点)
        [ y,  -x,  np.pi/2],   # 1: Leader 2 (右点)
        [-y,  x, -np.pi/2],   # 2: Follower 1 (左点)
        [ -x, -y,  np.pi],      # 3: Follower 2 (底点)
        [-0,0,0]                 # 4: 第五个智能体 (可选)
    ])
    # c, s = np.cos(np.pi/3), np.sin(np.pi/3)
    # R = np.zeros((3,3))
    # R[:2,:2] = np.array([
    #     [c, -s], 
    #     [s, c]
    # ]) 
    # R[2,2] = 1
    # g_nom = (R @ g_nom.T).T
    return g_nom

def get_five_agent_topology():
    """
    构成一个长度为 2 的 DEP (Dual-Entry Path)。
    两个入口是 Leader (0和1)，内部节点是 Follower (2和3)。
    双向通信路径: Leader1(0) <-> Follower1(2) <-> Follower2(3) <-> Leader2(1)
    """
    edges = [
        (0, 2),
        (2, 4),
        (4, 3),
        (3, 1),
    ]
    return edges

# === 测试代码 ===
if __name__ == "__main__":
    g_nom, edges = get_subgraph_config()
    
    print("标称构型形状:", g_nom.shape)
    print("Agent 1 (Leader) 状态:", g_nom[0])
    print("edges",edges)
    
    # 测试一下 Fig. 6 的例子数据 (用于验证函数正确性)
    # g_i=[-2,1,pi/2], g_j=[1.5,0.5,pi/4], g_k=[0,0,0]
    # 应该输出 Eq. (14) 的结果
    gi = np.array([-2, 1, np.pi/2])
    gj = np.array([1.5, 0.5, np.pi/4])
    gk = np.array([0, 0, 0])
    
    W_jk_test, W_ki_test = compute_W_matrix(gi, gj, gk, theta=0)
    print("\n--- 验证 W 矩阵 (对照论文 Eq. 14) ---")
    print("W_jk (Calculated):\n", np.round(W_jk_test, 4))
    print("W_ki (Calculated):\n", np.round(W_ki_test, 4))