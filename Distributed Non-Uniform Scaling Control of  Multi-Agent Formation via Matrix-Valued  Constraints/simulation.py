import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint

from main_reproduce import get_nominal_configuration
from formation_graph import FormationGraph 



class ManeuverTrajectory:
    """
    负责生成 Table I 中的时变参数 s(t), tau(t)
    使用三次样条插值 (Cubic Spline) 保证平滑
    """
    def __init__(self):
        # [cite_start]Table I 数据 [cite: 474]
        self.times = np.array([0, 5, 10, 15, 20, 25, 30, 35, 45])
        
        # 参数: [sx, tx, sy, ty, s_phi, t_phi]
        # 按表格数值录入
        self.params = np.array([
            [5.0, -47, 2.5, 4.0,  1.0, 0.0], # t=0
            [0.0, -30, 2.5, 0.0,  0.0, 0.0], # t=5 (Line formation sx=0)
            [0.0, -15, 2.5, 0.0,  0.0, 0.0], # t=10
            [0.0, -10, 2.5, 0.0, -1.0, 0.0], # t=15
            [0.0,   0, 4.5, 0.0, -1.0, 0.0], # t=20
            [0.0,   5, 4.5, 0.0,  0.0, 0.785], # t=25
            [0.0,  15, 4.5, 10.,  0.0, 0.785], # t=30
            [4.5,  35, 4.5, 10.,  1.0, -0.785], # t=35 (tx,ty guess from fig)
            [2.0,  50, 2.0, -5.,  1.0, -0.785]  # t=45
        ])
        
        # 创建插值器 (对每一列参数创建一个样条函数)
        self.splines = []
        for i in range(6):
            # bc_type='clamped' 强制首尾导数为0，保证平稳起停
            cs = CubicSpline(self.times, self.params[:, i], bc_type='clamped')
            self.splines.append(cs)

        

    def get_state(self, t):
        """返回 t 时刻的参数 [sx, tx, sy, ty, s_phi, t_phi]"""
        return np.array([s(t) for s in self.splines])

    def get_derivative(self, t):
        """返回 t 时刻的参数导数 (用于计算前馈速度)"""
        return np.array([s(t, 1) for s in self.splines])

def formation_dynamics(y, t, M_ff_inv, M_fl, M, g_nom, trajectory, k_l=1.0, k_f=5.0):
    """
    ODE 系统动力学方程
    y: 当前所有智能体状态 (扁平化 1D 数组)
    """
    # 1. 恢复状态形状 (9, 3)
    g_current = y.reshape(-1, 3)
    
    # 2. 获取当前时刻的目标参数
    params = trajectory.get_state(t)
    d_params = trajectory.get_derivative(t)
    
    s_p = params[[0, 2]] # sx, sy
    tau_p = params[[1, 3]] # tx, ty
    s_phi = params[4]
    tau_phi = params[5]
    
    ds_p = d_params[[0, 2]]
    dtau_p = d_params[[1, 3]]
    ds_phi = d_params[4]
    dtau_phi = d_params[5]
    
    # 3. 计算领导者的目标状态和速度 (Target & Velocity)
    # 公式 (7): g* = (I \kron S) * g_nom + 1 \kron tau
    # 这里我们只计算 Leader (index 0, 1)
    
    # 构建 S 矩阵 (3x3)
    # S(t) = diag(sx, sy, s_phi) (因为 theta=0, R=I)
    S = np.diag([s_p[0], s_p[1], s_phi])
    dS = np.diag([ds_p[0], ds_p[1], ds_phi])
    
    tau = np.array([tau_p[0], tau_p[1], tau_phi])
    dtau = np.array([dtau_p[0], dtau_p[1], dtau_phi])
    
    # 领导者索引
    leader_indices = [0, 1]
    g_dot = np.zeros_like(g_current)
    
    # --- 计算领导者控制律 (Eq. 28 upper part / Eq 29 error dynamics) ---
    # 论文中 Leaders 也是有控制律的: g_dot_l = -k_l * (g_l - g_l*) + g_dot_l*
    for i in leader_indices:
        g_nom_i = g_nom[i]
        
        # 目标状态 g*
        g_star = S @ g_nom_i + tau
        # 目标速度 g_dot*
        g_dot_star = dS @ g_nom_i + dtau
        
        # 误差反馈
        g_dot[i] = -k_l * (g_current[i] - g_star) + g_dot_star

    # --- 计算跟随者控制律 (Eq. 28 lower part) ---
    # 使用矩阵形式加速: M_ff * g_dot_f = -k_f * (M * g)_f - M_fl * g_dot_l
    # 这种形式等价于 Eq. 28，但避免了代数环求解
    
    # 提取领导者速度向量 (flatten)
    v_l = g_dot[0:2].flatten() # shape (6,)
    
    # 计算全系统偏差项: -k_f * (M @ g)
    # M (27x27) @ g (27,) -> (27,)
    g_flat = g_current.flatten()
    term_feedback = -k_f * (M @ g_flat)
    
    # 截取对应跟随者的部分 (rows 6 to 27)
    term_feedback_f = term_feedback[6:] 
    
    # 计算耦合项: - M_fl @ v_l
    term_coupling = - M_fl @ v_l
    
    # 求解跟随者速度: v_f = M_ff_inv @ (term_feedback + term_coupling)
    v_f = M_ff_inv @ (term_feedback_f + term_coupling)
    
    # 填回 g_dot
    g_dot[2:] = v_f.reshape(-1, 3)
    
    return g_dot.flatten()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    g_nom = get_nominal_configuration()
    traj = ManeuverTrajectory()
    
    # 2. 构建矩阵
    fg = FormationGraph(g_nom)
    M = fg.build_laplacian(theta=0)
    
    # 3. 矩阵分块 (Partitioning)
    # Leaders: indices 0, 1 (Rows 0-5)
    # Followers: indices 2-8 (Rows 6-26)
    n_l = 2
    dim = 3
    split_idx = n_l * dim
    
    M_ff = M[split_idx:, split_idx:] # (21, 21)
    M_fl = M[split_idx:, :split_idx] # (21, 6)
    
    # 预计算 M_ff 的逆 (因为 M 是常数)
    # 论文证明 M_ff 非奇异，所以可以求逆
    M_ff_inv = np.linalg.inv(M_ff)
    
    print(f"M_ff shape: {M_ff.shape}, Rank: {np.linalg.matrix_rank(M_ff)}")
    
    # 4. 设置初始状态 (Random initialization as per paper)
    # [cite_start]论文: "initial positions ... are randomly assigned" [cite: 465]
    np.random.seed(42)
    g0 = np.random.rand(9, 3) * 10 - 5 # Random range [-5, 5]
    # 或者为了好看，我们可以让 g0 接近 t=0 的目标状态
    
    # 5. 运行仿真
    t_span = np.linspace(0, 45, 500) # 0到45秒，500个点
    
    print("开始仿真...")
    # odeint 需要函数签名 f(y, t, ...)
    sol = odeint(formation_dynamics, g0.flatten(), t_span, 
                 args=(M_ff_inv, M_fl, M, g_nom, traj, 1.0, 5.0))
    
    print("仿真结束，开始绘图...")
    
    # 6. 绘图 (仿照 Fig. 7)
    g_final = sol.reshape(len(t_span), 9, 3)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制轨迹
    colors = ['r', 'r'] + ['b'] * 7 # Leaders red, Followers blue
    for i in range(9):
        # 绘制路径
        plt.plot(g_final[:, i, 0], g_final[:, i, 1], 
                 color=colors[i], linestyle=':', alpha=0.6)
        
        # 绘制几个关键帧的形状 (t=0, 15, 30, 45)
        # 用三角形表示智能体
        indices = [0, 100, 250, 400, 499] # 对应的大致时间点
        for idx in indices:
            x, y, phi = g_final[idx, i]
            # 简单的箭头绘制
            plt.arrow(x, y, np.cos(phi), np.sin(phi), 
                      head_width=0.8, head_length=1.0, fc=colors[i], ec=colors[i])
            
            # 如果是关键帧，画出连线 (拓扑)
            if i == 0: # 只在第一个智能体循环里画一次连线，避免重复
                current_pos = g_final[idx]
                # 绘制边
                edges = [(0,4),(4,3),(3,2),(2,1), # Path 1
                         (0,6),(6,5),(5,4),       # Path 2
                         (2,7),(7,8),(8,1)]       # Path 3
                for u, v in edges:
                    p1 = current_pos[u]
                    p2 = current_pos[v]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=0.5, alpha=0.3)

    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Formation maneuver trajectories in 2-D space.')
    plt.grid(True)
    plt.show()