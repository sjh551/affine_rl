# Leader Follower RL

一个基于 PyBullet 的多车编队强化学习项目。

## 工程目录（已整理）

```text
Leader_Follower_RL/
  assets/                     # 资源文件（URDF / meshes）
    ranger_mini_v3/
      urdf/
      meshes/
  models/                     # 不同方法/算法实现（后续扩展方法都放这里）
    DWA/
    uniform_scale_transform/
  data/                       # 数据集与采集脚本
  outputs/                    # 训练产物统一输出
    models/                   # *.zip 模型
    checkpoints/              # 阶段训练checkpoint
    tensorboard/
      stage1/
      stage2/
    autoencoder/              # 预训练编码器权重
    figures/                  # 分析图像
  docs/                       # 工程说明文档
  scripts/                    # 统一运行入口脚本
  old_files/                  # 历史备份代码

  pybullet_env.py             # 主环境
  train.py                    # 训练入口
  test.py                     # 测试入口
  evaluate_success_rate.py    # 成功率评估
```

## 关键约定

- `models/` 只放算法方法实现，不放资源文件。
- 小车 URDF 统一放在 `assets/ranger_mini_v3/`。
- 所有环境脚本都通过“基于文件位置的绝对路径”加载 URDF，不依赖当前工作目录。

## 常用运行方式

在项目根目录执行：

```powershell
python scripts/run_train.py
python scripts/run_test.py
python scripts/run_eval.py
python scripts/run_collect_data.py
python scripts/run_train_autoencoder.py
```

也支持直接运行原脚本（`train.py`、`test.py` 等），路径会自动解析到 `outputs/`。

## 为什么这样改

之前 `p.loadURDF("./ranger_mini_v3/..." )` 依赖启动目录，切换目录或拆分脚本后容易报找不到模型。现在改为按脚本文件位置解析路径，目录重构后仍可稳定运行。

第二阶段进一步把模型、tensorboard、checkpoint、图像等全部收敛到 `outputs/`，根目录保持干净，便于版本管理和后续扩展新方法。
