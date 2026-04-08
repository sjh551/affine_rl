import numpy as np
import matplotlib.pyplot as plt  # 画图神器

# 你给的点位函数
def get_nominal_configuration():
    g_nom = np.array([
        [-4, -2, np.pi/8],   # g1
        [ 2,  4, -np.pi/4],  # g2
        [ 0,  1, -np.pi/16], # g3
        [ 1, -1, np.pi/16],  # g4
        [-1,  0, 0.0],       # g5
        [-2, -3, -np.pi/16], # g6
        [-3, -4, np.pi/4],   # g7
        [ 3,  2, -np.pi/8],  # g8
        [ 4,  3, -np.pi/16]  # g9
    ])
    return g_nom

# 1. 获取点位
g_nom = get_nominal_configuration()

# 2. 只取前两维：x = 第一列，y = 第二列
x = g_nom[:, 0]  # 所有行，第0列
y = g_nom[:, 1]  # 所有行，第1列

# 3. 画图！
plt.figure(figsize=(6,6))  # 正方形画布，好看
plt.scatter(x, y, s=100, c='red', edgecolors='black')  # 画点

# 4. 给每个点标上 g1~g9
for i in range(len(x)):
    plt.text(x[i]+0.1, y[i]+0.1, f'g{i+1}', fontsize=12)

# 5. 辅助显示
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('9个智能体位置分布图')
plt.axis('equal')  # 让x/y轴比例一样，不畸变
plt.show()