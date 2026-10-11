# Full-State Observation Experiment Plan

## Objective

- Run id: mdp-full-state-vs-pomdp
- Question: does follower full-state observation improve PPO training under the same reward and budget?
- Baseline: POMDP_env.py.
- Variant: MDP_env.py with follower position, velocity, heading, and yaw rate.

## Comparability

- Keep reward, robot count, map, seeds, PPO settings, and budget identical.
- Compare success rate, return, episode length, formation error, collisions, and cross-track error.
- Both variants now use the same three-stage curriculum and PPO settings.
- POMDP kinematics stay at 22 dimensions; MDP kinematics stay at 40 dimensions.
- MDP uses 26 leader/goal/nominal-command values plus 14 follower-state values.
- Stages 1/2 run without followers and zero-pad the 14 follower-state values.

## Implementation

- Synchronize MDP reward with the current POMDP reward.
- Add train_mdp.py with isolated model, checkpoint, and TensorBoard names.
- Run a short CPU smoke test; do not launch the full training automatically.

## Commands

Run all commands from the project root.

### Help

```powershell
python train.py --help
python train_mdp.py --help
python test.py --help
python test_mdp.py --help
```

### POMDP training

```powershell
# Stage 1: no obstacles, no followers, train from scratch.
python train.py --stage 1 --seed 0

# Stage 2: obstacles, no followers, automatically load Stage 1.
python train.py --stage 2 --seed 0

# Stage 3: obstacles and followers, automatically load Stage 2.
python train.py --stage 3 --seed 0

# Continue the same stage and append to its existing TensorBoard curve.
# Always use a new environment seed range for every continuation.
python train.py --stage 1 --continue-training --seed 10000 --timesteps 3000000
python train.py --stage 2 --continue-training --seed 20000 --timesteps 3000000
python train.py --stage 3 --continue-training --seed 30000 --timesteps 5000000

# Continue the same stage from a checkpoint while preserving timesteps.
python train.py --stage 3 --continue-training --resume-from outputs/checkpoints/pomdp/CHECKPOINT.zip --seed 40000 --timesteps 5000000

# Smoke-test any stage on CPU without saving a model.
python train.py --stage 1 --smoke
python train.py --stage 2 --smoke --from-scratch
python train.py --stage 3 --smoke --from-scratch

# Override parallel environments, budget, device, and random seed.
python train.py --stage 1 --num-envs 8 --timesteps 3000000 --device cuda --seed 0

# Initialize from an explicit model or transfer-learning checkpoint.
# Use --continue-training as well when this is a same-stage continuation.
python train.py --stage 1 --continue-training --resume-from ppo2_stage1_nav --seed 10000
python train.py --stage 2 --resume-from outputs/models/ppo2_stage1_nav.zip --seed 0
python train.py --stage 3 --resume-from outputs/models/ppo2_stage2_obstacles_no_followers.zip --seed 0

# Disable the automatic previous-stage model and initialize a new policy.
python train.py --stage 2 --from-scratch
python train.py --stage 3 --from-scratch
```

### Full-state MDP training

```powershell
# Stage 1: no obstacles, no followers, with zero-padded follower state slots.
python train_mdp.py --stage 1 --seed 0

# Stage 2: obstacles, no followers, automatically load MDP Stage 1.
python train_mdp.py --stage 2 --seed 0

# Stage 3: obstacles and followers, automatically load MDP Stage 2.
python train_mdp.py --stage 3 --seed 0

# Recommended continuation sequence. Each command appends to the same
# TensorBoard curve and advances the environment to a new random stream.
python train_mdp.py --stage 1 --continue-training --seed 10000 --timesteps 3000000
python train_mdp.py --stage 2 --continue-training --seed 20000 --timesteps 3000000
python train_mdp.py --stage 2 --continue-training --seed 30000 --timesteps 3000000
python train_mdp.py --stage 3 --continue-training --seed 40000 --timesteps 5000000
python train_mdp.py --stage 3 --continue-training --seed 50000 --timesteps 5000000

# Continue Stage 3 from a same-stage checkpoint and preserve TensorBoard steps.
python train_mdp.py --stage 3 --continue-training --resume-from outputs/checkpoints/mdp_full_state/CHECKPOINT.zip --seed 60000 --timesteps 5000000

# Smoke-test any stage on CPU without saving a model.
python train_mdp.py --stage 1 --smoke
python train_mdp.py --stage 2 --smoke --from-scratch
python train_mdp.py --stage 3 --smoke --from-scratch

# Override parallel environments, budget, device, and random seed.
python train_mdp.py --stage 1 --num-envs 8 --timesteps 3000000 --device cuda --seed 0

# Initialize from an explicit model or transfer-learning checkpoint.
# Use --continue-training as well when this is a same-stage continuation.
python train_mdp.py --stage 1 --continue-training --resume-from mdp_ppo2_stage1_nav --seed 10000
python train_mdp.py --stage 2 --resume-from outputs/models/mdp_ppo2_stage1_nav.zip --seed 0
python train_mdp.py --stage 3 --resume-from outputs/models/mdp_ppo2_stage2_obstacles_no_followers.zip --seed 0

# Disable the automatic previous-stage model and initialize a new policy.
python train_mdp.py --stage 2 --from-scratch
python train_mdp.py --stage 3 --from-scratch
```

Do not combine `--from-scratch` with `--resume-from` or `--continue-training`.
Same-stage continuation requires an explicit new `--seed` and uses
`reset_num_timesteps=False`. Entering a new stage resets that stage's curve to zero.
Existing historical TensorBoard runs are not retroactively merged.

### POMDP evaluation

```powershell
python test.py --stage 1 --render
python test.py --stage 2 --render
python test.py --stage 3 --render
python test.py --stage 3 --model outputs/models/ppo2_final_leader-best.zip --render
python test.py --stage 3 --stochastic --render
python test.py --stage 2 --random-policy --no-plot --max-steps 100 --seed 0
python test.py --stage 3 --output outputs/figures/pomdp_stage3.png --seed 0
# Evaluate an explicit seed list and print aggregate statistics.
python test.py --stage 3 --seeds 0 1 2 3 4 --no-plot
# Evaluate 20 consecutive seeds (0 through 19) and save JSON metrics.
python test.py --stage 3 --seed 0 --num-seeds 20 --no-plot --metrics-output outputs/pomdp_stage3_20_seeds.json
```

### Full-state MDP evaluation

```powershell
python test_mdp.py --stage 1 --render
python test_mdp.py --stage 2 --render --seed 1  # 0 2 3 5 12
python test_mdp.py --stage 3 --render
python test_mdp.py --stage 3 --model outputs/models/mdp_ppo2_final_leader-best.zip --render
python test_mdp.py --stage 3 --stochastic --render
python test_mdp.py --stage 2 --random-policy --no-plot --max-steps 100 --seed 0
python test_mdp.py --stage 3 --output outputs/figures/mdp_stage3.png --seed 0
# Evaluate an explicit seed list and print aggregate statistics.
python test_mdp.py --stage 3 --seeds 0 1 2 3 4 --no-plot
# Evaluate 100 held-out consecutive seeds and save JSON metrics.
python test_mdp.py --stage 3 --seed 100000 --num-seeds 100 --no-plot --metrics-output outputs/mdp_stage3_test_100000_100099.json
```

### TensorBoard

```powershell
tensorboard --logdir outputs/tensorboard
tensorboard --logdir outputs/tensorboard --port 6006
```
