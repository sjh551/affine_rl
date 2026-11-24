# 工程目录

在本仓库新增代码时，按所属职责放到现有目录。

- `scripts/` 放用户入口和命令行参数。训练、评估的实现放在 `src/robot_rl/training/`。
- `configs/experiments/` 放实验 YAML。任务默认值、奖励公式、观测编码和 Gymnasium 回合逻辑放在 `src/robot_rl/tasks/<task>/`。
- PyBullet 场景与物理操作放在 `src/robot_rl/sim/`，地图布局放在 `maps/`，车体及执行器接口放在 `robots/`，控制律放在 `controllers/`。这些包均在 `src/robot_rl/` 下。
- 模型静态资源放在 `src/robot_rl/assets/`，回归测试放在 `tests/`，训练日志、模型、评估报告等运行产物放在 `outputs/`。
- 仓库根目录放项目配置、文档和目录。新增 Python 模块时使用对应的子目录。

使用 `python -m unittest discover -s tests -v` 验证任务行为。改动训练入口时再运行 `python scripts/train.py --smoke`。
