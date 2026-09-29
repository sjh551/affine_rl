# Spectral-norm candidate board

| Candidate | Level | Strategy | Status | Expected gain | Promotion rule |
| --- | --- | --- | --- | --- | --- |
| radius-3 | implementation | exploit | completed | Best feasible norm `1.00000002` | Retain as matrix incumbent, not deployment-ready |
| radius-6 | implementation | explore | archived | Best observed norm `1.10355` | No improvement over radius 3 |
| radius-12 | implementation | explore | archived | Best observed norm `1.08544` | No improvement; larger box made search harder |
| radius-3-unconstrained | implementation | exploit | archived | Reached `0.8660` | Rejected: minimum spacing `0.263 m`, immediate PyBullet termination |

All candidates keep the same topology, leaders, yaw values, planar matrix construction,
solver family, `sigma_min(D_ff)` lower bound and `d_min=0.8 m` collision constraint.
