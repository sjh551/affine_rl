import time
import pybullet as p
import numpy as np
from uniform_scale_env import FormationPyBulletEnv

# 初始化环境
env = FormationPyBulletEnv(render_mode=True, enable_obstacles=True)
env.reset()

print("================================================")
print(" 方向键[↑↓←→] 控制 Leader A 移动, 1,2 控制转动")
print(" [WSAD] 控制 Leader B 移动，Q,E 控制转动")
print(" 按下 [R] 重置环境")
print("================================================")

while True:
    # 1. 获取 PyBullet 的键盘事件
    keys = p.getKeyboardEvents()
    
    # 默认动作 [Ax, Ay, Bx, By] = [0, 0, 0, 0]（控制两个Leader机器人）
    action = np.array([0.0, 0.0, 0.0]) 
    
    # 2. 键盘映射
    # A机器人：上下左右方向键
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        action[1] = 0.5  
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        action[1] = -0.5 
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        action[0] = -0.5 
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        action[0] = 0.5  
    if ord('1') in keys and keys[ord('1')] & p.KEY_IS_DOWN:
        action[2] = 1.0  
    if ord('2') in keys and keys[ord('2')] & p.KEY_IS_DOWN:
        action[2] = -1.0 
    # 重置键 'r'
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        env.reset()
        print("环境已重置！")
        continue

    # 3. 喂给环境
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 4. 实时打印奖励，帮你调试 Reward 函数写得对不对
    # \r 可以让打印在同一行刷新，不会刷屏
    print(f"Reward: {reward:.3f} | Action: {action} ", end='\r')
    
    time.sleep(0.03) # 约 30FPS