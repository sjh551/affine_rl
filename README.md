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

### 常用训练命令

```powershell
# 1. 冒烟测试：单个 CPU 环境，128 步，不保存产物
python scripts/train.py --smoke

# 2. 默认训练：读取 configs/experiments/formation_ppo.yaml（stage 2，car-like 地图，500 万步）
python scripts/train.py
python scripts/train.py --config configs/experiments/formation_ppo.yaml

# 3. 课程训练：先在无障碍场地训练（默认 100 万步），再加载该模型进入障碍地图
python scripts/train.py --stage 1
python scripts/train.py --resume-from "outputs/<stage1运行目录>/ppo_affine.zip" --stage 2 --map car-like

# 4. 切换地图与跟随控制方式
python scripts/train.py --map forest
python scripts/train.py --follower-control distributed --neighbor-velocity measured
python scripts/train.py --map forest --follower-control distributed --neighbor-velocity command

# 5. 指定设备、并行环境数、步数、种子和输出目录
python scripts/train.py --device cpu --num-envs 2 --timesteps 200000
python scripts/train.py --seed 1 --output-dir outputs/seed1

# 6. 查看训练曲线
tensorboard --logdir outputs
```

默认配置包含 4 个并行环境，使用 CUDA 训练。CPU 环境可传 `--device cpu`。`--smoke` 用单个 CPU 环境做 128 步采样与更新。实验的场景、随机种子、PPO 参数和评估种子放在 YAML 中，命令行可以覆盖地图、控制模式、训练步数等常用选项。未传 `--timesteps` 时，stage 1 训练 `open_timesteps` 步，stage 2 训练 `obstacle_timesteps` 步。

### 命令行参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--config` | YAML 路径 | 实验配置，默认 `configs/experiments/formation_ppo.yaml`；续训时默认读模型目录的 `experiment.yaml` |
| `--stage` | `1` / `2` | 1 为无障碍场地，2 为障碍地图 |
| `--map` | `car-like` / `forest` | stage 2 使用的障碍地图 |
| `--no-walls` | 开关 | 关闭外圈围墙 |
| `--follower-control` | `centralized` / `distributed` | 跟随车控制方式 |
| `--neighbor-velocity` | `measured` / `command` | distributed 模式下邻居速度来源 |
| `--num-envs` | 正整数 | 并行环境数（`SubprocVecEnv`） |
| `--timesteps` | 正整数 | 本次训练（或续训追加）的步数，覆盖 YAML 的阶段默认步数 |
| `--device` | `cuda` / `cpu` 等 | PyTorch 设备 |
| `--seed` | 非负整数 | 随机种子 |
| `--output-dir` | 路径 | 运行产物根目录，默认 `outputs` |
| `--resume-from` | 模型或检查点路径 | 从已有权重继续训练 |
| `--new-run` | 开关 | 加载权重但新建运行目录和日志 |
| `--smoke` | 开关 | 冒烟测试，不保存产物 |

### YAML 配置参数

`configs/experiments/formation_ppo.yaml` 中的参数及默认值：

| 段 | 参数 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `environment` | `stage` | `2` | 训练阶段 |
| | `map` | `car-like` | 障碍地图 |
| | `walls` | `true` | 是否生成围墙 |
| | `follower_control` | `centralized` | 跟随控制方式 |
| | `neighbor_velocity` | `measured` | 邻居速度来源 |
| | `max_steps` | `1500` | 每回合最大步数（10 Hz，约 150 s） |
| `training` | `seed` | `0` | 随机种子 |
| | `device` | `cuda` | 训练设备 |
| | `num_envs` | `4` | 并行环境数 |
| | `timesteps` | `null` | 显式步数；为空时按阶段取下面两项 |
| | `open_timesteps` | `1000000` | stage 1 默认步数 |
| | `obstacle_timesteps` | `5000000` | stage 2 默认步数 |
| | `checkpoint_steps` | `1000000` | 每隔多少步保存检查点 |
| | `output_dir` | `outputs` | 输出目录 |
| `ppo` | `rollout_steps` | `4096` | 每次更新的总采样步数（按 `num_envs × batch_size` 向上取整） |
| | `batch_size` | `256` | 小批量大小 |
| | `n_epochs` | `5` | 每批数据的优化轮数 |
| | `clip_range` | `0.1` | PPO 裁剪范围 |
| | `target_kl` | `0.03` | 提前停止更新的 KL 阈值 |
| | `learning_rate` | `3e-5` | stage 2 学习率 |
| | `open_learning_rate` | `1e-4` | stage 1 学习率 |
| | `ent_coef` | `0.001` | stage 2 熵系数 |
| | `open_ent_coef` | `0.0005` | stage 1 熵系数 |
| `evaluation` | `seeds` | `[1000, ..., 1004]` | 评估使用的固定测试种子 |

PPO 参数只能在 YAML 中修改。需要不同超参数时，复制一份 YAML 后用 `--config` 指定。

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

奖励沿用 `Leader_Follower_RL/train.py` 所用 POMDP 环境的设计，包含目标进度、每步成本、领导车速度朝向、近目标构型、到达奖励和碰撞惩罚。构型项用三条有向边相对标称三角形的偏差均方根。

带障碍阶段还计算三项连续惩罚。跟随车相对仿射参考的误差超过 0.5 m 时按超出量扣分。用三台领导车的实测平面速度向前外推 0.5 s，经仿射映射得到四台跟随车的预测位置，预测净距低于 0.7 m 时扣分。实际车体与障碍物表面间距进入 0.5 m 预警区时扣除反距离代价。各项权重集中在任务的 `config.py`，TensorBoard 的 `reward/` 下可查看分项均值。奖励权重变化后加载旧权重会创建新的日志曲线。

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
