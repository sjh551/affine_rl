# 项目文件用途速查

本文按当前仓库文件整理（2026-09-18）。路径均相对于项目根目录。**代码、配置、文档逐个列出**；大量由同一程序自动生成的检查点、图像和 TensorBoard 日志按文件名模式合并说明。这里记录的是当前实际调用关系，不保证文件名或旧 README 中的描述仍然准确。

## 最容易混淆的环境与入口

| 文件 | 当前用途 |
| --- | --- |
| `POMDP_env.py` | 主 PyBullet 部分可观测编队环境。训练、主测试、成功率评估和手动调试等脚本导入它；跟随者可选集中式、分布式或同步速度内迭代控制，速度指令经舵轮关节电机执行。还包含 APF、死区、谱诊断和可选构型优化。 |
| `POMDP_env test.py` | 控制律隔离实验用的另一份完整环境，不是 pytest 测试。其 `_drive_robot()` 用 `resetBaseVelocity()` 直接给底盘设速度，绕过舵轮执行环节；默认构型和增益也与主环境不同。文件名含空格，不能用通常的 `from POMDP_env test import ...` 导入。 |
| `debug_env_auto.py` | 自动让两名领导者向前运动的诊断脚本；用 `importlib` 加载 `POMDP_env test.py`，关闭跟随者 APF，并观察速度/越界等现象。 |
| `debug_env_manual.py` | 键盘手动控制和观察主环境；导入的是 `POMDP_env.py`。 |
| `debug_timestep_stability.py` | 用当前优化构型和 `k_f=1` 对照满足/违反 Schur–Cohn 步长条件的离散闭环，并补充轮式 PyBullet 环境对照。 |
| `MDP_env.py` | 与主 POMDP 环境对照的全状态观测版本，用于检验观测信息增加后的训练效果。 |
| `train.py` / `train_mdp.py` | 分别训练 POMDP 主环境和 MDP 对照环境的 PPO 策略。 |
| `test.py` / `test_mdp.py` | 分别加载 POMDP / MDP 环境做策略回放；与 `POMDP_env test.py` 不是一回事。 |

`POMDP_env test.py` 与主环境不只是执行器不同：目前主环境默认使用 `get_nominal_configuration()/get_topology()`、可配置编队增益；实验版默认使用方形构型/拓扑，跟随者增益写为 `0.5`。直接比较两者轨迹时，须先统一这些设置。

## 根目录：配置、训练和分析

| 文件 | 用途 |
| --- | --- |
| `config.py` | 机器人尺寸、安全距离、控制步长等共享常量。 |
| `project_paths.py` | 根据项目位置解析模型、检查点、TensorBoard、图像等输出路径。 |
| `training_common.py` | PPO 特征提取器、奖励回调、分阶段配置和向量化环境等共用训练代码。 |
| `evaluation_common.py` | PPO 模型评估、单回合轨迹记录、碰撞/路径统计及绘图所需的共用代码。 |
| `evaluate_success_rate.py` | 批量评估主 POMDP 策略的成功率和其他指标。 |
| `tune_hyperparams.py` | 用 Optuna 搜索主环境中 PPO 的超参数。 |
| `speed_test.py` | 测试不同指令速度下的跟踪误差并绘制对比图。 |
| `train_autoencoder.py` | 训练激光雷达数据自动编码器，并可绘制重建结果。 |
| `control_ranger_mini_test.py` | 单独测试 Ranger Mini 车辆模型的舵轮关节和底盘运动。 |
| `README.md` | 项目概览和运行方式；其中 `pybullet_env.py` 是旧文件名，当前主环境为 `POMDP_env.py`。 |
| `PLAN.md` | POMDP 与全状态 MDP 对照实验的计划。 |
| `CHECKLIST.md` | 上述对照实验的实施/验证清单。 |
| `requirements.txt` | 主项目 Python 依赖列表。 |
| `affine_paper.txt` | 仿射编队论文的本地文本摘录，供查阅；不是程序入口。 |

## `utils/`：编队与数值计算

| 文件 | 用途 |
| --- | --- |
| `utils/formation_utils.py` | 标称构型、编队拓扑与局部矩阵权重 `W` 的计算；含论文构型、方形构型及五智能体构型函数。 |
| `utils/formation_graph.py` | 根据构型和三元约束建立编队图/块拉普拉斯矩阵，并检查其性质。 |
| `formation_optimization/optimizer.py` | 独立搜索满足跟随者速度迭代谱半径要求的标称构型。 |
| `formation_optimization/run_optimization.py` | 离线优化入口，保存 `optimized_configuration.npy` 和 `optimization_summary.json`。 |
| `formation_optimization/evaluate_formation.py` | 加载已经保存的构型，独立启动主 PyBullet 环境做验证，不重新优化。 |
| `formation_optimization/PLAN.md` / `README.md` | 构型优化的研究计划、指标和运行说明。 |
| `utils/follower_velocity_iteration.py` | 在冻结的控制周期内，对跟随者速度做同步 Jacobi 内迭代并记录每轮变化范数。 |
| `utils/math_utils.py` | 旧/通用数学辅助函数，包括 `W` 计算、标称参数和雷达数据分扇区处理；注意与 `formation_utils.py` 中同名函数区分。 |

## `scripts/`、`data/` 与绘图

| 文件 | 用途 |
| --- | --- |
| `scripts/run_train.py` | 从任意工作目录转到项目根目录后启动 `train.py`。 |
| `scripts/run_test.py` | 同样启动根目录 `test.py`。 |
| `scripts/run_eval.py` | 同样启动根目录 `evaluate_success_rate.py`。 |
| `scripts/run_collect_data.py` | 同样启动 `data/collect_data.py`。 |
| `scripts/run_train_autoencoder.py` | 同样启动 `train_autoencoder.py`。 |
| `data/collect_data.py` | 从主环境采集雷达观测，生成自动编码器训练数据。 |
| `draw/trajectory_formation_plot.py` | 绘制机器人轨迹、编队连线、障碍物和围墙。 |
| `draw/distance_statistics_plot.py` | 统计并绘制机器人之间及机器人—障碍物之间的距离。 |
| `draw/reward.py` | 绘制奖励相关的分析图。 |
| `draw/shoot_img.py` | 截取仿真环境场景图，供论文/汇报使用。 |

## `models/`：其他方法及基线

| 文件 | 用途 |
| --- | --- |
| `models/README.md` | 不同算法目录的组织约定。 |
| `models/APF/APF_env.py` | 人工势场（APF）方法的独立仿真环境。 |
| `models/APF/apf_controller.py` | APF 速度指令/斥力控制器。 |
| `models/APF/test.py` | APF 环境的单次可视化测试。 |
| `models/APF/evaluate_success_rate.py` | APF 方法的批量成功率评估。 |
| `models/DWA/DWA_env.py` | 动态窗口法（DWA）所用的独立环境。 |
| `models/DWA/DWA.py` | DWA 轨迹预测、动态窗口和代价评价算法。 |
| `models/DWA/DWA_test.py` | DWA 方法的测试与批量评估入口。 |
| `models/DWA/math_utils.py` | DWA 环境使用的雷达扇区处理工具。 |
| `models/ORCA/ORCA_env.py` | ORCA 避碰方法的独立环境。 |
| `models/ORCA/test.py` | ORCA 方法的可视化测试。 |
| `models/ORCA/evaluate_success_rate.py` | ORCA 方法的批量成功率评估。 |
| `models/ORCA-F/ORCA_F_env.py` | ORCA-F 变体的独立环境。 |
| `models/ORCA-F/test.py` | ORCA-F 方法的可视化测试。 |
| `models/ORCA-F/evaluate_success_rate.py` | ORCA-F 方法的批量成功率评估。 |
| `models/IPPO-F/IPPO_env.py` | 独立的多智能体 IPPO-F 环境。 |
| `models/IPPO-F/ippo_shared.py` | 共享参数的 IPPO actor–critic 网络与观测/动作整理工具。 |
| `models/IPPO-F/train.py` | IPPO-F 的训练入口。 |
| `models/IPPO-F/test.py` | IPPO-F 模型加载与回放。 |
| `models/IPPO-F/evaluate_success_rate.py` | IPPO-F 模型的批量成功率评估。 |
| `models/IPPO-F/README.md` | 当前内容误写为 MAPPO 说明，与此目录的 IPPO-F 代码不符；阅读时以代码为准。 |
| `models/MAPPO/mappo_env.py` | 在主环境外封装多智能体领导者接口与集中式全局状态。 |
| `models/MAPPO/mappo_core.py` | MAPPO 的 actor、critic、轨迹缓冲区与训练更新。 |
| `models/MAPPO/train_mappo.py` | MAPPO 训练入口。 |
| `models/MAPPO/README.md` | MAPPO 最小实现与运行命令的说明。 |
| `models/uniform_scale_transform/uniform_scale_env.py` | 匀尺度编队变换方法的独立环境。 |
| `models/uniform_scale_transform/formation_utils.py` | 该方法目录中的标称构型、拓扑和权重计算副本。 |
| `models/uniform_scale_transform/formation_graph.py` | 该方法目录中的编队图/拉普拉斯矩阵构造副本。 |
| `models/uniform_scale_transform/debug_env_manual.py` | 键盘测试 `uniform_scale_env.py`。 |
| `models/uniform_scale_transform/train.py` | 目录内的旧 PPO 训练脚本；当前实际导入根目录 `POMDP_env.py`，不是 `uniform_scale_env.py`。 |
| `models/uniform_scale_transform/test.py` | 目录内的旧策略回放脚本；当前同样导入根目录 `POMDP_env.py`。 |

## 独立最小实验与历史文件

| 文件 | 用途 |
| --- | --- |
| `minimal_distributed_cube_demo/distributed_cube_demo.py` | 用简单方块智能体演示局部分布式编队、领导者手动输入、速度谱/轨迹和可选 APF。 |
| `minimal_distributed_cube_demo/manual_control.py` | 为最小实验提供便捷的手动启动入口。 |
| `minimal_distributed_cube_demo/README.md` | 最小实验的参数和运行说明。 |
| `minimal_distributed_cube_demo/requirements.txt` | 最小实验的独立依赖。 |
| `old_files/POMDP_maneuver_env.py` | 早期 POMDP 编队机动环境备份。 |
| `old_files/DWA_env.py` | 早期 DWA 环境备份。 |
| `old_files/main_demo.py` | 早期仿真演示脚本。 |
| `old_files/IDEA_REPORT.md` | 历史研究选题/方法构思报告。 |
| `docs/project_structure.md` | 此前两阶段目录整理的记录。 |
| `docs/FILE_PURPOSES.md` | 本文件：当前文件用途与入口关系索引。 |

## 资源文件和自动生成产物

下列文件不是需要逐个运行的源代码；同组文件用途相同，新增编号文件沿用本组说明。

| 文件或模式 | 用途 |
| --- | --- |
| `assets/ranger_mini_v3/urdf/ranger_mini.urdf` | Ranger Mini 机器人模型，定义底盘、关节、碰撞体等。 |
| `assets/ranger_mini_v3/meshes/ranger_base.obj` / `.mtl` | 车体网格几何及材质。 |
| `assets/ranger_mini_v3/meshes/steering_wheel.obj` / `.mtl` | 转向轮网格几何及材质。 |
| `assets/ranger_mini_v3/meshes/wheel_v3.obj` / `.mtl` | 车轮网格几何及材质。 |
| `models/MAPPO/checkpoints/mappo_update_20.pt` 至 `mappo_update_340.pt`（每 20 次更新一个） | MAPPO 训练过程检查点，数字是更新次数。 |
| `outputs/autoencoder/pretrained_encoder.pth` | 雷达自动编码器预训练权重。 |
| `outputs/figures/ae_reconstruction_test.png` | 自动编码器重建可视化。 |
| `outputs/figures/distance_statistics*.png` | 距离统计图的不同实验/版本。 |
| `outputs/figures/environment*.png` | 环境/编队场景截图的不同版本。 |
| `outputs/figures/reward_field.png` | 奖励场示意图。 |
| `outputs/figures/tracking_error_analysis_f1_f2.png` | 两名跟随者的速度/轨迹跟踪误差分析图。 |
| `outputs/follower_inner_iteration/inner_iteration_trace.csv` | 每控制周期、每轮内迭代的速度变化记录。 |
| `outputs/follower_inner_iteration/summary.csv` | 不同内迭代设置的汇总指标。 |
| `outputs/tensorboard/stage1/**/events.out.tfevents.*` | 第一阶段导航训练曲线；路径中的 `pomdp`、`mdp_full_state` 和运行名区分实验。 |
| `outputs/tensorboard/stage2/**/events.out.tfevents.*` | 第二阶段（领导者与障碍物）训练曲线。 |
| `outputs/tensorboard/stage3/**/events.out.tfevents.*` | 第三阶段完整编队训练曲线。 |

修改文件名或移动入口前，建议先检索其导入/调用者；特别是 `debug_env_auto.py` 明确依赖带空格的实验环境文件名。
