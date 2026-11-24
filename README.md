# 三领导者仿射编队强化学习

用 PPO 控制三台 Ranger 领导车，带领四台跟随车避障并到达目标。标称构型、领导者 1–3 和 12 条通信边来自 `Zhao2018Affine`。仿真使用 PyBullet，训练使用 Gymnasium 和 Stable-Baselines3。

## 安装

在 `affine_rl` 仓库根目录运行。项目使用 Python 3.10 或更新版本。

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

开发工具可用 `python -m pip install -e ".[dev]"` 安装。

## 训练与续训

```powershell
python scripts/train.py --smoke
python scripts/train.py --config configs/experiments/formation_ppo.yaml
python scripts/train.py --map forest --follower-control distributed --neighbor-velocity command
tensorboard --logdir outputs
```

默认配置包含 4 个并行环境，使用 CUDA 训练。CPU 环境可传 `--device cpu`。`--smoke` 用单个 CPU 环境做 128 步采样与更新。实验的场景、随机种子、PPO 参数和评估种子放在 YAML 中，命令行可以覆盖地图、控制模式、训练步数等常用选项。

正式训练在 `outputs/stage*_*/` 写入 `experiment.yaml`、训练配置记录 `training_sessions.jsonl`、TensorBoard 日志、检查点和最终模型 `ppo_affine.zip`。日志记录回合奖励、成功率、PPO 指标、目标距离和编队误差。

```powershell
python scripts/train.py --resume-from "outputs/<运行目录>/ppo_affine.zip" --timesteps 100000
```

续训、评估和回放会优先读取模型目录的 `experiment.yaml`，也可用 `--config` 显式指定。较早的模型没有配置快照时，请补上对应的场景参数。使用原环境配置续训会接上原 TensorBoard 曲线和累计步数。`--timesteps` 是计划追加的步数，PPO 按完整采样批次更新。也可以从 `checkpoints/` 中恢复。使用 `--new-run` 可另建日志，例如从早于最终模型的检查点开始一条新曲线。

## 评估与回放

```powershell
python scripts/evaluate.py "outputs/<运行目录>/ppo_affine.zip"
python scripts/evaluate.py "outputs/<运行目录>/ppo_affine.zip" --seeds 1000 1001 --output outputs/evaluation.json
python scripts/play.py --model "outputs/<运行目录>/ppo_affine.zip"
python scripts/play.py --map forest
python scripts/play.py --no-obstacles --demo
```

评估默认用 YAML 中的固定测试种子和确定性动作，输出每个种子的回合结果以及成功率、碰撞率、超时率、平均回报、平均步数、跟随误差和终点距离。复现比较时，两组模型使用相同的环境配置和测试种子。

手动模式中，绿色为领导车，蓝色为跟随车。点击 PyBullet 窗口后，用 `W/A/S/D` 移动，按 `1/2/3` 选中领导车，按 `0` 同时控制三台。空格暂停或继续，`R` 重置同一场景，`Q` 退出。碰撞或回合结束后按 `R` 重新开始。`--demo` 展示约 4 秒的剪切和非均匀缩放。

## 地图、动作与观测

| 场景 | 参数和障碍物 |
| --- | --- |
| 固定地图 | 默认 `--map car-like`，23 个箱体和圆柱，几何等比例缩放为 0.5 倍，场地 54 × 34 m |
| 随机森林 | `--map forest`，60 个圆柱，场地 48 × 20 m |
| 无障碍 | `--stage 1`，可视化也支持 `--no-obstacles` |

外圈围墙由仿真场景根据地图尺寸生成，可用 `--no-walls` 关闭。初始目标方向定义本回合的前向坐标轴。动作空间为 `(6,)`，每位领导者有前向和侧向速度分量。观测包含 `kinematics (29,)` 和 `lidar (3, 5, 60)`。后者包含三台领导车各 5 帧雷达扫描。

## 控制与奖励

`centralized` 根据应力矩阵的跟随映射每 0.1 秒统一计算一次跟随指令。`distributed` 沿原始通信边每 1/60 秒更新一次。用 `--neighbor-velocity measured` 读取邻居实测速度，或用 `command` 读取上一拍广播的限速后速度指令。后者在重置时置零。PPO 的动作和观测频率为 10 Hz。

奖励包含目标进度、每步成本、动作变化、跟随误差、近目标构型、到达奖励和碰撞惩罚。权重和成功阈值在任务的 `config.py`，各项贡献由 `rewards.py` 的 `reward_terms()` 计算。

## 工程目录

| 目录 | 内容 |
| --- | --- |
| `scripts/` | 训练、评估、手动驾驶与策略回放入口 |
| `configs/experiments/` | 可复用的任务、训练和评估配置 |
| `src/robot_rl/tasks/formation_navigation/` | Gymnasium 接口、任务配置、奖励和观测 |
| `src/robot_rl/sim/`、`maps/`、`robots/`、`controllers/` | PyBullet 世界、场景布局、Ranger 车体和跟随控制 |
| `src/robot_rl/training/` | 训练流程、策略网络、评估与日志 |
| `src/robot_rl/assets/ranger/` | URDF 和网格 |
| `tests/` | 奖励和环境回归测试 |
| `outputs/` | 运行时生成的配置快照、日志、模型和评估报告 |
