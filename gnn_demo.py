import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data

# 1. 构造图数据 (以你的 3 机器人编队为例)
# 节点特征：假设每个机器人有 2 维状态 [x, y]
x = torch.tensor([
    [0.0, 0.0],  # Robot 0 (Leader 1)
    [1.0, 1.0],  # Robot 1 (Leader 2)
    [0.5, 0.0]   # Robot 2 (Follower)
], dtype=torch.float)

# 边索引 (Edge Index): 描述谁能观测到谁 (Source -> Target)
# 假设 Follower (2) 观测 Leaders (0 和 1)
edge_index = torch.tensor([
    [0, 1],      # 起点
    [2, 2]       # 终点 (2号机器人聚合0和1的信息)
], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

# 2. 定义 GCN 网络结构
class RobotGNN(torch.nn.Module):
    def __init__(self):
        super(RobotGNN, self).__init__()
        # 输入维度 2 (x,y)，隐藏层维度 4
        self.conv1 = GCNConv(2, 4)
        # 输出层，假设输出 2 维动作指令
        self.conv2 = GCNConv(4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层卷积 + 激活函数
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        
        return x

# 3. 运行模型
model = RobotGNN()
output = model(data)

print("原始坐标特征:\n", data.x)
print("GNN 提取后的特征/动作预测:\n", output)