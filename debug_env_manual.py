import time
import pybullet as p
import numpy as np
from pybullet_env import FormationPyBulletEnv

# 初始化环境
env = FormationPyBulletEnv(render_mode=True, enable_obstacles=True)
env.reset()

print("================================================")
print(" 按下键盘 [上/下/左/右] 控制 Leader 移动")
print(" 按下 [R] 重置环境")
print("================================================")

while True:
    # 1. 获取 PyBullet 的键盘事件
    keys = p.getKeyboardEvents()
    
    # 默认动作 [vx, vy] = [0, 0]
    action = np.array([0.0, 0.0])
    
    # 2. 简单的键盘映射
    # 65297: UP, 65298: DOWN, 65295: LEFT, 65296: RIGHT (PyBullet按键码)
    # 或者用 p.B3G_UP_ARROW 等常量
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        action[1] = 0.5  # 向前
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        action[1] = -0.5 # 向后
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        action[0] = -0.5 # 向左
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        action[0] = 0.5  # 向右
        
    # 重置键 'r'
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        env.reset()
        print("环境已重置！")
        continue

    # 3. 喂给环境
    obs, reward, done, info = env.step(action)
    
    # 4. 实时打印奖励，帮你调试 Reward 函数写得对不对
    # \r 可以让打印在同一行刷新，不会刷屏
    print(f"Reward: {reward:.3f} | Action: {action} ", end='\r')
    
    time.sleep(0.03) # 约 30FPS