import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
from project_paths import DATASET_100K_PATH, FIGURES_DIR, PRETRAINED_ENCODER_PATH, ensure_output_dirs

# ==========================================
# 1. 定义自编码器模型 (Autoencoder)
# ==========================================
class LidarAutoencoder(nn.Module):
    def __init__(self, in_channels=3, ray_length=180, latent_dim=128):
        super().__init__()
        
        #  必须与你在 SB3 中 CustomCombinedExtractor 的结构一模一样！
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 自动推断展平后的维度 (与你的 SB3 写法一致)
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, ray_length)
            self.n_flatten = self.cnn(dummy_input).shape[1] 
            # 经过计算，180的输入在这里会被压缩成 1216 (64 * 19)

        # 编码器的全连接层
        self.encoder_linear = nn.Sequential(
            nn.Linear(self.n_flatten, latent_dim),
            nn.ReLU()
        )
        
        # -------------------------------------------
        # 解码器 (Decoder)：精确的逆向操作
        # -------------------------------------------
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, self.n_flatten),
            nn.ReLU()
        )
        
        # 使用转置卷积还原维度：19 -> 21 -> 44 -> 180
        self.decoder_deconv = nn.Sequential(
            # 输入: (Batch, 64, 19) -> 输出: (Batch, 64, 21)
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # 输入: (Batch, 64, 21) -> 输出: (Batch, 32, 44)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # 输入: (Batch, 32, 44) -> 输出: (Batch, 3, 180)
            nn.ConvTranspose1d(32, in_channels, kernel_size=8, stride=4),
            
            # 因为雷达危险指数是 0.0 ~ 1.0，所以最后一层用 Sigmoid 激活！
            nn.Sigmoid() 
        )

    def encode(self, x):
        features = self.cnn(x)
        latent = self.encoder_linear(features)
        return latent

    def decode(self, latent):
        x = self.decoder_linear(latent)
        x = x.view(-1, 64, 19) # 恢复成 1D 卷积所需的形状 (Batch, Channels, Length)
        reconstruction = self.decoder_deconv(x)
        return reconstruction

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

# ==========================================
# 2. 训练主流程
# ==========================================
def train_autoencoder(dataset_path=str(DATASET_100K_PATH), epochs=30, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")
    ensure_output_dirs()

    # 1. 加载数据
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集 {dataset_path}，请先运行采集脚本！")
        
    data = np.load(dataset_path) # shape: (50000, 3, 180)
    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_data) # 自监督：输入和标签都是自己
    
    # 划分训练集和验证集 (90% 训练, 10% 验证)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型、损失函数和优化器
    model = LidarAutoencoder().to(device)
    criterion = nn.MSELoss() # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. 开始训练
    print("🚀 开始训练自编码器...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 4. 提取并只保存 Encoder 权重！(这是我们真正在 RL 里需要的东西)
    encoder_state_dict = {
        '0.weight': model.cnn[0].weight, '0.bias': model.cnn[0].bias,
        '2.weight': model.cnn[2].weight, '2.bias': model.cnn[2].bias,
        '4.weight': model.cnn[4].weight, '4.bias': model.cnn[4].bias,
        '7.0.weight': model.encoder_linear[0].weight, '7.0.bias': model.encoder_linear[0].bias
    }
    torch.save(encoder_state_dict, str(PRETRAINED_ENCODER_PATH))
    print(f"💾 编码器权重已成功保存至: {PRETRAINED_ENCODER_PATH}")
    
    # 5. 可视化一下结果 (看看它学得怎么样)
    visualize_reconstruction(model, val_dataset, device)

# ==========================================
# 3. 可视化验证
# ==========================================
def visualize_reconstruction(model, dataset, device, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            # 随机抽样
            idx = np.random.randint(len(dataset))
            real_data, _ = dataset[idx]
            
            # 模型推理
            input_tensor = real_data.unsqueeze(0).to(device)
            reconstructed = model(input_tensor).cpu().squeeze(0)
            
            # 我们只画出最近的一帧 (通道 2)
            real_frame = real_data[2].numpy()
            recon_frame = reconstructed[2].numpy()
            
            axes[i, 0].plot(real_frame, color='blue')
            axes[i, 0].set_title("Ground Truth Lidar")
            axes[i, 0].set_ylim(0, 1)
            
            axes[i, 1].plot(recon_frame, color='red')
            axes[i, 1].set_title("Reconstructed Lidar")
            axes[i, 1].set_ylim(0, 1)
            
    plt.tight_layout()
    fig_path = FIGURES_DIR / "ae_reconstruction_test.png"
    plt.savefig(str(fig_path))
    print(f"重建效果对比图已保存为: {fig_path}")
    plt.show()

if __name__ == "__main__":
    train_autoencoder(epochs=300)

