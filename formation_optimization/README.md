# 编队构型优化

本目录只负责离线构型优化和优化结果检验。`POMDP_env.py` 不会在初始化时自动运行优化器。

谱半径统一按 follower 的平面状态 `[x1, y1, x2, y2, ...]` 计算，
即使用 `T_xy = D_ff,xy^(-1) A_ff,xy`；姿态 `phi` 不进入优化约束或环境中的谱诊断。

## 优化

在项目根目录执行：

```powershell
python formation_optimization/run_optimization.py --rho-limit 0.95
```

谱半径版本同时约束：固定标称符号的局部/DEP 横纵坐标差、所有智能体的
最小平面距离、跟随者搜索边界，以及 `sigma_min(D_ff)` 的数值裕度。默认
`coordinate_margin=1e-3`、`min_distance=0.8`；显式工作空间边界可例如写为：

```powershell
python formation_optimization/run_optimization.py `
  --rho-limit 0.95 `
  --coordinate-margin 0.1 `
  --min-distance 0.8 `
  --x-min -8 --x-max 8 `
  --y-min -8 --y-max 8
```

领导者和全部姿态分量不进入决策变量，始终保持基准构型中的值。若没有显式
提供工作空间边界，跟随者边界仍由 `--search-radius` 相对基准位置生成。

一次运行连续 20 个随机种子（从 `--seed` 指定的起始值开始）：

```powershell
python formation_optimization/run_optimization.py --seed 0 --num-seeds 20
```

也可以显式指定不连续的种子：

```powershell
python formation_optimization/run_optimization.py --seeds 0 3 8 15
```

每个种子的构型和摘要分别保存在 `results/seeds/`。程序会从所有可行结果中
选择目标函数 `J` 最小者，并将其复制为兼容现有检验脚本的
`results/optimized_configuration.npy`；总体结果写入
`results/optimization_summary.json`。单个种子失败不会中断后续运行。

以上是谱半径约束版本。谱范数约束版本单独运行，不会覆盖谱半径结果：

```powershell
python formation_optimization/run_spectral_norm_optimization.py --norm-limit 1.05
```

谱范数版本现在使用与谱半径版本相同的固定领导者、固定姿态、带符号坐标
裕度、全体最小距离、工作空间边界和 `sigma_min(D_ff)` 约束。批量运行 20 个
随机种子可使用：

```powershell
python formation_optimization/run_spectral_norm_optimization.py `
  --norm-limit 1.05 `
  --coordinate-margin 0.8 `
  --min-distance 1.0 `
  --num-seeds 20
```

各个种子的谱范数结果保存在 `results/spectral_norm/seeds/`，最佳可行结果仍
保存为 `results/spectral_norm/optimized_configuration.npy`。

谱范数结果默认保存到 `formation_optimization/results/spectral_norm/`。
由于 `rho(T) <= ||T||_2`，谱范数约束更严格。此前不含固定符号坐标裕度的
九智能体搜索曾在 follower 各坐标 `±3` 的搜索域内得到约 `1.0` 的最小谱
范数；加入本轮完整几何约束后该数值不能直接作为新的可达下界，应先用扫描
程序重新检查。不要直接照搬 `--rho-limit 0.5` 作为谱范数上限。

若目的是探索谱范数本身在不同搜索范围内能够达到多小，而不是寻找离原构型
最近的可行点，运行：

```powershell
python formation_optimization/sweep_spectral_norm_search.py `
  --search-radii 3 6 12 `
  --seeds 0 1 `
  --coordinate-margin 0.8 `
  --min-pairwise-distance 0.8
```

该扫描以 `||T||_2` 本身作为目标函数，结果单独保存在
`formation_optimization/results/spectral_norm_sweep/`，不会覆盖前两种构型。
扫描默认约束任意两台机器人中心距离不小于 `0.8 m`，避免优化器通过产生
物理重叠构型来降低矩阵范数。

两种结果生成后，执行统一的矩阵指标对比：

```powershell
python formation_optimization/compare_optimization_methods.py
```

对比报告默认保存为
`formation_optimization/results/optimization_comparison.json`。该脚本不启动
PyBullet，以保持优化和物理检验解耦。

使用完全相同的 PyBullet 参数，对两个已保存构型进行无键盘、领导者零输入的
独立物理对比：

```powershell
python formation_optimization/compare_pybullet_formations.py --steps 200
```

结果保存为 `formation_optimization/results/pybullet_comparison.json`。该程序只读取
优化结果，不会重新执行优化。

如果要优化自己的四智能体构型，可先保存为 `.npy`（形状为 `(4, 3)`），再执行：

```powershell
python formation_optimization/run_optimization.py --base-config path\to\my_configuration.npy --n-leaders 2
```

结果默认写入 `formation_optimization/results/`：

- `optimized_configuration.npy`：供环境加载的构型；
- `optimization_summary.json`：可复现实验摘要。

## PyBullet 检验

```powershell
python formation_optimization/evaluate_formation.py --config formation_optimization/results/optimized_configuration.npy --render
```

检验谱范数优化结果：

```powershell
python formation_optimization/evaluate_formation.py --config formation_optimization/results/spectral_norm/optimized_configuration.npy --render
```

不指定 `--config` 时，脚本默认加载
`formation_optimization/results/optimized_configuration.npy`。构型行数会自动识别：4 行使用四智能体拓扑，5 行使用五智能体拓扑，9 行使用论文拓扑；不支持的行数需要在脚本中提供对应拓扑。

带 `--render` 时，交互方式与 `debug_env_manual.py` 一致：方向键控制 Leader A，W/S/A/D 控制 Leader B，R 重置，Q 或 Esc 退出。默认不设运行步数，会一直运行。

检验脚本会以 `max_steps=None` 创建主环境，因此不会因为 1500 步超时而重置；只有成功、撞墙或碰撞等实际回合终止事件才会自动重置并继续运行。主环境本身默认仍是 `max_steps=1500`，不影响训练脚本的原有行为。

检验脚本也可以无界面运行：

```powershell
python formation_optimization/evaluate_formation.py --steps 300
```

`POMDP_env.FormationPyBulletEnv` 接收 `nominal_configuration`，因此检验只加载结果，不重新优化。
