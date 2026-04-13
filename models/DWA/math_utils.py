import numpy as np

def process_lidar(stacked_lidar, num_sectors=18, obs_inflate=1):
    """
    VFH 雷达特征提取 → 输出 [0,1] 归一化直方图
    0 = 最安全
    1 = 最危险
    """
    channels, num_rays = stacked_lidar.shape
    rays_per_sector = num_rays // num_sectors
    
    hist = np.zeros((channels, num_sectors), dtype=np.float32)
    
    for c in range(channels):
        for i in range(num_rays):
            val = stacked_lidar[c, i]
            if val < 1e-4:
                continue

            mag = val ** 2
            base_sector = i // rays_per_sector

            # 扩散到相邻扇区
            for ds in range(-obs_inflate, obs_inflate + 1):
                s = (base_sector + ds) % num_sectors
                w = max(0.0, 1.0 - abs(ds) / (obs_inflate + 1))
                hist[c, s] += mag * w

    hist = np.clip(hist, 0.0, 1.5)

    return hist