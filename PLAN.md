# Full-State Observation Experiment Plan

## Objective

- Run id: mdp-full-state-vs-pomdp
- Question: does follower full-state observation improve PPO training under the same reward and budget?
- Baseline: POMDP_env.py.
- Variant: MDP_env.py with follower position, velocity, heading, and yaw rate.

## Comparability

- Keep reward, robot count, map, seeds, PPO settings, and budget identical.
- Compare success rate, return, episode length, formation error, collisions, and cross-track error.
- Current risk: POMDP uses zero followers while MDP uses two; align this before the main comparison.

## Implementation

- Synchronize MDP reward with the current POMDP reward.
- Add train_mdp.py with isolated model, checkpoint, and TensorBoard names.
- Run a short CPU smoke test; do not launch the full training automatically.

## Commands

```powershell
python train_mdp.py --stage 1 --smoke --device cpu
python train_mdp.py --stage 1
python train_mdp.py --stage 1 --resume-from mdp_ppo2_stage1_nav
python train_mdp.py --stage 2
python test_mdp.py --stage 2 --render
python test_mdp.py --stage 1 --render
```
