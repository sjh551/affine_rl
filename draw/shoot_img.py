import pybullet as p
import matplotlib.pyplot as plt
import numpy as np
import os

def capture_thesis_image(filename="scenario.png"):
    # 1. 定义超高清分辨率 (比如 1920x1080)
    width = 1920
    height = 1080
    
    # 2. 计算视图矩阵 (View Matrix)
    # 摄像机位置: x=0, y=0, z=12 (在高空12米处)
    # 目标位置: x=0, y=0, z=0 (看向地面中心)
    # 相机朝上方向: [0, 1, 0] (决定了图片的“上方”是环境的Y轴)
    cam_pos = [0, 0, 21.7]
    target_pos = [0, 0, 0]
    up_vector = [0, 1, 0]
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vector
    )
    
    # 3. 计算投影矩阵 (Projection Matrix)
    # fov: 视场角，aspect: 宽高比，near/far: 渲染范围
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.1,
        farVal=100.0
    )
    
    # 4. 获取图像数据
    img_arr = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL # 使用硬件加速渲染
    )
    
    # 5. 提取 RGB 数据并保存为图片
    # img_arr[2] 包含了 RGBA 数据 (一维数组)，将其 reshape 为 (height, width, 4)
    rgba = np.reshape(img_arr[2], (height, width, 4))
    rgb = rgba[:, :, :3] # 去掉 Alpha 通道
    
    # 转换数据类型为 uint8 (matplotlib 需要)
    rgb = rgb.astype(np.uint8)
    
    # 创建输出文件夹
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建完整的输出路径
    output_path = os.path.join(output_dir, filename)
    
    # 使用 matplotlib 保存 (不会失真)
    plt.imsave(output_path, rgb)
    print(f"✅ 论文高清配图已保存至: {output_path}")

# 在你需要截图的时刻调用：
# capture_thesis_image("narrow_corridor_top_view.png")