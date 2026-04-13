import numpy as np
from numpy.linalg import matrix_rank

from utils.formation_utils import compute_W_matrix, get_nominal_configuration

class FormationGraph:
    def __init__(self, g_nom):
        self.n_agents = len(g_nom)
        self.g_nom = g_nom
        self.dim = 3  # 每个智能体状态维度 (x, y, phi)
        
        # 初始化 27x27 的大矩阵 (9 agents * 3 dim)
        self.M = np.zeros((self.n_agents * self.dim, self.n_agents * self.dim))
        
        # 获取三元组约束
        self.triplets = self._get_triplets()
        
    def _get_triplets(self):
        """
        根据节点数量动态返回 DEP 三元组 (i, j, k)
        其中 k 是中间节点 (Follower)，i 和 j 是它的两个邻居。
        """
        triplets = []
        
        if self.n_agents == 4:
            # ==========================================
            # 2 Leader + 2 Follower 的基础正方形拓扑
            # 路径串联: Leader1(0) <-> Follower1(2) <-> Follower2(3) <-> Leader2(1)
            # ==========================================
            
            # 1. 对于 Follower1 (索引 k=2): 
            # 它的两个邻居是 Leader1(0) 和 Follower2(3)。
            # 按照 (i, j, k) 且 i < j 的习惯：
            triplets.append((0, 3, 2)) 
            
            # 2. 对于 Follower2 (索引 k=3): 
            # 它的两个邻居是 Follower1(2) 和 Leader2(1)。
            # 按照 (i, j, k) 且 i < j 的习惯：
            triplets.append((1, 2, 3)) 
            
        elif self.n_agents == 9:
            # ==========================================
            # 原有的 9 节点复杂拓扑
            # ==========================================
            # === Path 1 (Red) ===
            triplets.append((0, 3, 4)) 
            triplets.append((2, 4, 3)) 
            triplets.append((1, 3, 2)) 
            
            # === Path 2 (Blue) ===
            triplets.append((0, 5, 6)) 
            triplets.append((4, 6, 5)) 
            
            # === Path 3 (Yellow) ===
            triplets.append((2, 8, 7)) 
            triplets.append((1, 7, 8)) 
            
        else:
            raise ValueError(f"未定义的拓扑结构！当前智能体数量为 {self.n_agents}")
            
        return triplets


    def build_laplacian(self, theta=0):
        """
        根据 Eq. (18) 构建矩阵值 Laplacian M
        """
        # 清空矩阵
        self.M.fill(0)
        
        for (i, j, k) in self.triplets:
            # 1. 获取标称状态
            gi = self.g_nom[i]
            gj = self.g_nom[j]
            gk = self.g_nom[k]
            
            # 2. 计算局部 W 矩阵
            # Eq. 13: W_jk 对应 edge (j,k), W_ki 对应 edge (i,k) [注意下标顺序]
            # 函数 get_W_matrix(i, j, k) 返回的是 W_jk, W_ki
            W_jk, W_ki = compute_W_matrix(gi, gj, gk, theta)
            
            # 3. 填充大矩阵 M (Eq. 18)
            # M 的第 k 行块 (Row Block k)
            
            # 3.1 对角块 M_kk += (W_jk + W_ki)
            self._add_block(k, k, W_jk + W_ki)
            
            # 3.2 非对角块 M_ki = -W_jk (注意：列是 i，系数是 W_jk)
            self._add_block(k, i, -W_jk)
            
            # 3.3 非对角块 M_kj = -W_ki (注意：列是 j，系数是 W_ki)
            self._add_block(k, j, -W_ki)
            
        return self.M

    def _add_block(self, row_idx, col_idx, matrix_block):
        """辅助函数：将 3x3 块加到大矩阵的指定位置"""
        r_start = row_idx * self.dim
        c_start = col_idx * self.dim
        self.M[r_start:r_start+3, c_start:c_start+3] += matrix_block

    def check_validity(self):
        """
        验证构建是否成功
        依据: Lemma III.5, Rank(M) 应该等于 3n - 6
        对于 n=9, Rank 应该是 27 - 6 = 21
        """
        rank = matrix_rank(self.M)
        expected_rank = 3 * self.n_agents - 6
        print(f"矩阵秩检查: 实际 Rank = {rank}, 预期 Rank = {expected_rank}")
        
        if rank == expected_rank:
            print("✅ Laplacian 构建成功！系统满足最大可操纵性条件。")
            return True
        else:
            print("❌ Laplacian 构建可能出错，或者标称构型存在奇异性。")
            return False

# ==========================================
# 运行测试
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    g_nom = get_nominal_configuration()
    
    # 2. 初始化图结构
    formation = FormationGraph(g_nom)
    
    # 3. 构建矩阵
    M = formation.build_laplacian(theta=0)
    
    # 4. 打印调试信息
    print(M)
    print(f"Laplacian 矩阵形状: {M.shape}")  # 应该是 (27, 27)
    
    # 5. 验证秩 (这是最关键的一步)
    formation.check_validity()
    
    # 6. 可选：打印一下跟随者 g5 (index 4) 的对角块，看看是否有数值
    # g5 是 Path 1 的中间节点，应该有非零值
    idx = 4
    print(f"\nAgent 5 (Index 4) 的对角块 M_55:\n{np.round(M[idx*3:idx*3+3, idx*3:idx*3+3], 2)}")