# 最小分布式方块编队 Demo

这个目录用于单独检验论文《Distributed Non-Uniform Scaling Control of
Multi-Agent Formation via Matrix-Valued Constraints》的跟随者控制律。

它刻意不使用以下内容：

- URDF；
- 轮子、转向关节和底盘速度解算；
- Gym/Gymnasium 环境；
- 强化学习模型、观测和奖励；
- `POMDP_env.py` 中的集中式 `M_ff` 联立求解。

四个智能体全部由 PyBullet 的 `GEOM_BOX` 原语直接创建。重力和智能体碰撞
被关闭，控制输入直接设为世界坐标系速度 `[vx, vy, yaw_rate]`，所以这是一个
理想全向一阶运动学实验。

## 分布式执行顺序

每个控制周期严格按下面的因果顺序执行：

1. 同时采样所有智能体的位置、姿态和实际速度，形成只读快照；
2. Follower 2 只读取邻居 0、3 的快照，独立计算自己的控制量；
3. Follower 3 只读取邻居 1、2 的快照，独立计算自己的控制量；
4. 两个跟随者都计算结束后，再同时向所有方块施加速度；
5. 执行若干 PyBullet 物理子步，然后进入下一个控制周期。

同一周期中，一个跟随者看不到另一个跟随者刚计算出的期望速度。程序会构造
`T=D_ff^{-1}A_ff` 供诊断和打印谱半径，但 `T` 不参与控制计算。

## 运行

### 手动控制领导者

在项目根目录、`general` Conda 环境中运行：

```powershell
python minimal_distributed_cube_demo/manual_control.py
```

手动入口默认不注入初始位置扰动，避免还没按键就触发不稳定模态；如需主动测试
微小扰动，可增加 `--perturbation 1e-5`。

按键与 `debug_env_manual.py` 保持一致：

- Leader A：方向键移动，`1`、`2` 旋转；
- Leader B：`W`、`A`、`S`、`D` 移动，`Q`、`E` 旋转；
- `R`：恢复初始构型；
- `P`：截图到 `minimal_distributed_cube_demo/screenshots/`。

默认平移速度为 `0.5 m/s`，可以修改：

```powershell
python minimal_distributed_cube_demo/manual_control.py `
  --manual-speed 0.2 --manual-angular-speed 0.3
```

也可以直接运行主程序：

```powershell
python minimal_distributed_cube_demo/distributed_cube_demo.py --manual
```

### 自动运动测试

让两个领导者按照命令行给定的恒定速度运动：

```powershell
python minimal_distributed_cube_demo/distributed_cube_demo.py --gui --realtime
```

无界面运行并保存误差与速度曲线：

```powershell
python minimal_distributed_cube_demo/distributed_cube_demo.py `
  --plot minimal_distributed_cube_demo/result.png
```

静止领导者实验：

```powershell
python minimal_distributed_cube_demo/distributed_cube_demo.py `
  --leader-vx 0 --plot minimal_distributed_cube_demo/static_leader.png
```

改变控制周期。实际控制周期为
`physics_timestep * substeps`：

```powershell
python minimal_distributed_cube_demo/distributed_cube_demo.py `
  --physics-timestep 0.0041666667 --substeps 1

python minimal_distributed_cube_demo/distributed_cube_demo.py `
  --physics-timestep 0.0010416667 --substeps 1
```

加入 APF（默认关闭，障碍物位于 `[5, 0]`）：

```powershell
python minimal_distributed_cube_demo/distributed_cube_demo.py `
  --gui --realtime --enable-apf
```

## 如何解释结果

- 去掉 URDF 后仍然发散：不能把根因归结为 URDF 或轮式底盘；应继续检查
  当前拓扑下的速度递推结构和 `rho(T)`。
- 去掉 URDF 后稳定，而原环境发散：重点检查舵轮解算、执行器响应、坐标系
  变换、速度测量和物理接触。
- 只有大控制周期发散：说明离散化和控制周期是主要因素。
- 只有开启 APF 后异常：说明 APF 与编队控制的直接叠加或速度饱和需要处理。

默认在 Follower 2 的 x 坐标加入 `1e-5 m` 扰动，否则精确标称构型可能使
不稳定模态因为初值恰好为零而暂时不被激发。可通过 `--perturbation` 修改。
