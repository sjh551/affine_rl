# 编队构型优化计划

## 目标

在不改变领导者标称状态、姿态和编队拓扑的前提下，只优化跟随者的平面位置，使分布式跟随者速度迭代矩阵

\[
T(p)=D_{ff}(p)^{-1}A_{ff}(p)
\]

满足给定谱半径上限，同时尽量保持原始构型：

\[
\min_p \sum_i\|p_i-\tilde p_i\|^2,
\qquad \rho(T(p))\leq \rho_{\max}.
\]

## 工作流

1. `run_optimization.py` 读取 `utils/formation_utils.py` 中的基准构型和 `FormationGraph` 中的拓扑。
2. 优化器只修改 follower 的 `x/y`，保持 leader 的 `x/y/phi` 以及全部姿态不变。
   谱半径也只对 follower 的平面 `x/y` 子矩阵计算，姿态 `phi` 不参与。
3. 优化完成后，将构型保存为 `optimized_configuration.npy`，将谱半径、偏差、奇异值和求解器状态保存为 `optimization_summary.json`。
4. `evaluate_formation.py` 独立加载 `.npy`，通过 `POMDP_env.FormationPyBulletEnv(nominal_configuration=...)` 启动与主环境相同的 PyBullet 环境。
5. 检验阶段不执行优化；它只验证矩阵谱、构型残差以及零输入或手动领导者运动下的实际仿真表现。

## 当前必须报告的指标

- 初始和优化后的 `rho(D_ff^-1 A_ff)`；
- `sigma_min(D_ff)`、`rank(M)` 和 `||M g_nom||`；
- 所有几何约束、机器人间最小距离和边界约束；
- 分布式速度内迭代的每轮变化范数；
- PyBullet 中的编队误差、速度、碰撞和是否发散。

## 注意事项

- `rho(T)<1` 只说明冻结周期的 Jacobi 速度迭代收敛，不自动证明带位置积分、舵轮执行器、APF 和采样延迟的完整闭环稳定。
- 每次优化后都必须使用 `evaluate_formation.py` 实测；优化器和仿真检验是两个独立过程。
- 若需要更换拓扑，应同时修改 `FormationGraph` 和优化入口，并重新检查 `M g_nom=0`、秩和局部矩阵可逆性。
