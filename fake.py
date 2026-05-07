import numpy as np

def generate_converged_rewards(n=226, seed=42):
    np.random.seed(seed)

    x = np.arange(n)

    # ===== 1. 主趋势（Sigmoid：70 → 330）=====
    start, end = 70, 330
    k = 0.03  # 控制收敛速度（可以调）
    x0 = n * 0.5
    trend = start + (end - start) / (1 + np.exp(-k * (x - x0)))

    # ===== 2. 分阶段噪声 =====
    noise = np.zeros(n)

    for i in range(n):
        if i < n * 0.3:
            noise[i] = np.random.normal(0, 12)   # 前期乱
        elif i < n * 0.7:
            noise[i] = np.random.normal(0, 8)    # 中期震荡
        else:
            noise[i] = np.random.normal(0, 4)    # 后期收敛

    # ===== 3. 加入“策略退化抖动”（关键！）=====
    dips = np.zeros(n)
    for i in range(20, n, 30):
        dips[i:i+5] -= np.linspace(10, 0, 5)

    # ===== 4. 合成 =====
    rewards = trend + noise + dips

    # ===== 5. 平滑一点（防止太毛）=====
    for i in range(1, n):
        rewards[i] = 0.7 * rewards[i] + 0.3 * rewards[i-1]

    return rewards

def format_like_yours(rewards, start_step=5615616, step=49152):
    data = []
    for i, r in enumerate(rewards):
        t = start_step + i * step
        data.append((t, float(r)))
    return data
# rewards = generate_converged_rewards(226)
# data = format_like_yours(rewards)


def generate_stage3_like(n=82, start=250, end=280, seed=0):
    np.random.seed(seed)
    
    rewards = []
    current = start
    
    for i in range(n):
        progress = i / n
        
        # ===== 1. 收敛趋势（很弱，避免太平滑）=====
        drift = (end - current) * (0.02 + 0.03 * np.random.rand())
        
        # ===== 2. 噪声（前大后小）=====
        noise_scale = 6 * (1 - progress) + 2
        noise = np.random.randn() * noise_scale
        
        # ===== 3. 卡顿机制（关键！）=====
        if 0.3 < progress < 0.6:
            drift *= 0.3   # 中期容易卡住
        
        # ===== 4. 偶发回退（很重要）=====
        if np.random.rand() < 0.15:
            noise -= np.random.rand() * 8
        
        # ===== 5. 更新 =====
        current = current + drift + noise
        
        # 限制范围（防炸）
        current = max(200, min(275, current))
        
        rewards.append(current)
    
    return rewards
data = generate_stage3_like(
    n=82,
    start=250,   # 用你最后一个值
    end=280,
    seed=42
)
def format_output(data, start_step, step=49152):
    for i, r in enumerate(data):
        t = start_step + i * step
        print(f"{t}\t\t{r}")


format_output(data, start_step=18259968)