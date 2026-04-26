import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

# --- 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 参数设置 ---
d_min = 0.5
limit = 2 * d_min
obj_pos = [0, 0]

x = np.linspace(-1.5, 1.5, 500)
y = np.linspace(-1.5, 1.5, 500)
X, Y = np.meshgrid(x, y)
D = np.sqrt((X - obj_pos[0])**2 + (Y - obj_pos[1])**2)

Z = np.zeros_like(D)
Z[D <= d_min] = -50
mask = (D > d_min) & (D < limit)
Z[mask] = -( (1.0 / D[mask]) - (1.0 / (2 * d_min)) )
Z[D >= limit] = 0

# --- 自定义色图：深红→橙→黄→浅绿 ---
colors = ['#8B0000', '#CD5C5C', '#FF8C00', '#FFD700', '#90EE90', '#F0FFF0']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('danger_safe', colors, N=n_bins)

# --- 绘图 ---
fig, ax = plt.subplots(figsize=(8, 6))

vmin, vmax = -1.2, 0.1
cf = ax.contourf(X, Y, Z, levels=np.linspace(vmin, vmax, 60), cmap=cmap, extend='both')

cbar = fig.colorbar(cf, ticks=np.linspace(vmin, vmax, 7))
# cbar.set_label('奖励值 $r_{obs}$', rotation=270, labelpad=20, fontsize=12)

# 标注关键物理边界
ax.plot(obj_pos[0], obj_pos[1], 'ko', markersize=10, 
        markerfacecolor='white', markeredgewidth=2, label='障碍物中心')

c_min = Circle(obj_pos, d_min, color='darkred', fill=False, 
               linestyle='--', linewidth=2.5, label=r'$d_{\min}$ 碰撞边界')
ax.add_patch(c_min)

c_max = Circle(obj_pos, limit, color='black', fill=False, 
               linestyle=':', linewidth=2, label=r'$2d_{\min}$ 势场范围')
ax.add_patch(c_max)

# ax.set_xlabel('X 坐标 [m]', fontsize=11)
# ax.set_ylabel('Y 坐标 [m]', fontsize=11)
# ax.set_title('避障势场奖励函数分布', fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, linestyle='-', alpha=0.2)

plt.tight_layout()
plt.savefig('./outputs/figures/reward_field.png', bbox_inches='tight', dpi=300)
plt.show()