# 三领导者仿射编队强化学习

在 PyBullet 中训练一套 PPO 策略，控制 3 台 Ranger 领导车，引导 7 台车组成的编队到达目标。标称点型、领导者编号 1–3 和 12 条通信边来自 `Zhao2018Affine` 的默认编队。环境按 1.5 倍标称尺寸生成初始编队，并保留原训练任务的四列圆柱障碍和场地围墙。

## 训练

在项目根目录运行。已用 Python 3.10 验证。

```powershell
python -m pip install -r requirements.txt
python train.py --smoke
python train.py
```

`train.py` 默认训练带障碍的完整编队，使用 4 个并行环境。可用 `--num-envs`、`--timesteps`、`--device` 和 `--seed` 修改训练设置。`--stage 1` 使用无障碍场地，仍保留全部 7 台车。通过 `--resume-from <模型路径>` 可以继续训练三领导者模型，或将无障碍阶段的权重用于带障碍训练。旧二领导者权重与新动作及观测空间不兼容。

正式训练为每次运行创建独立的 `outputs/stage*_*/` 目录，存放 TensorBoard 日志、周期性检查点和最终 `ppo_affine.zip`。`--smoke` 在 CPU 上完成 128 步采样和 PPO 更新，不生成训练文件。

## 可视化测试

从项目根目录运行。脚本直接打开训练使用的 PyBullet 环境，可以观察围墙、障碍物、目标点和 7 台车。绿色车辆是领导车，蓝色是跟随车。

```powershell
python -m examples.view_env
python -m examples.view_env --no-obstacles --demo
```

默认进入手动模式。先点击 PyBullet 窗口让它接收键盘输入，再按下列按键操作。移动方向以本回合从起点指向目标的方向为前方。

| 按键 | 操作 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左移 / 右移 |
| `0` / `1` / `2` / `3` | 同时控制三台领导车 / 单独控制对应编号的领导车 |
| 空格 / `R` / `Q` | 暂停或继续 / 重置当前场景 / 退出 |

初始编队中，领导车 1 在最前方，领导车 2 在左侧，领导车 3 在右侧。`--demo` 会执行约 4 秒的剪切和非均匀缩放，随后暂停。按空格可继续手动驾驶，按 `R` 可重播。终端显示到目标的距离、跟随误差和回合结束原因。可用 `--seed` 指定要复现的场景。

## 动作与观测

平面坐标系以该回合起点至目标的方向为前轴，并在回合内固定。动作和观测中的领导者均按编号 1、2、3 排列。

| 空间 | 形状 | 内容 |
| --- | --- | --- |
| 动作 | `(6,)` | 三台领导车的前向与侧向速度分量，每个分量在 `[-1, 1]` 内，单车合速度上限为 1 m/s |
| `kinematics` | `(29,)` | 目标相对仿射编队质心的二维向量，三台领导车相对该质心的位置、线速度、航向误差的正余弦及角速度，以及上一步的六维动作 |
| `lidar` | `(3, 5, 60)` | 每台领导车的 5 帧历史扫描。每圈 180 条射线以 3 条为一组取最近距离，量程 7 m，归一化至 `[0, 1]` |

奖励使用全队实际质心的目标进度和跟踪误差。到达目标时还检查领导三角形和随从跟踪精度。物理碰撞、车间距小于 0.7 m 或越过围墙边界会终止回合，超时返回截断信号。

## 跟随控制

训练和可视化共用 `--follower-control` 参数。默认的 `centralized` 模式每 0.1 秒用预计算映射 `F = -solve(Ω_ff, Ω_fl)` 计算跟随速度 `v_f = F @ v_l - (p_f - F @ p_l)`。

`distributed` 模式沿用原始 12 条通信边。每个跟随者每 1/60 秒从同一拍的物理快照读取自身位置和邻居的位置、实测速度，计算一次

$$
u_i[k] = \frac{1}{\Omega_{ii}} \sum_{j\in\mathcal{N}_i} w_{ij}
\bigl(v_j[k] + p_j[k] - p_i[k]\bigr).
$$

这里的 $v_j[k]$ 是下发本拍指令前的底盘实测速度，领航者也按此方式采样。每次局部更新后推进 4 个 PyBullet 物理子步。分布式控制只使用邻居边权，没有拍内迭代或额外通信时延。环境仍用 $F$ 计算奖励和观测中的仿射参考。两种模式的跟随指令均限速至 1.3 m/s，PPO 动作和观测均以 10 Hz 更新。

```powershell
python -m examples.view_env --follower-control distributed --no-obstacles --demo
python train.py --follower-control distributed
```

## 文件

| 路径 | 内容 |
| --- | --- |
| `train.py` | PPO 训练、并行环境、日志和模型保存 |
| `training_common.py` | 三台领导车共享的 LiDAR 特征提取器 |
| `POMDP_env.py` | Gymnasium 环境、动作与观测、地图和奖励 |
| `formation.py` | 标称构型、应力矩阵和集中式跟随控制律 |
| `controllers/distributed.py` | 按通信边计算的分布式跟随控制律 |
| `ranger.py`、`assets/` | Ranger 仿真和 URDF 资源 |
| `examples/view_env.py` | PyBullet 可视化、键盘驾驶和仿射机动演示 |
