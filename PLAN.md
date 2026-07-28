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
- POMDP kinematics stay at 22 dimensions; MDP kinematics stay at 36 dimensions.
- Stages 1/2 run without followers. MDP pads the 14 follower-state values with zeros.

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
python train.py --stage 1

# Stage 2: obstacles, no followers, automatically load Stage 1.
python train.py --stage 2

# Stage 3: obstacles and followers, automatically load Stage 2.
python train.py --stage 3

# Smoke-test any stage on CPU without saving a model.
python train.py --stage 1 --smoke
python train.py --stage 2 --smoke --from-scratch
python train.py --stage 3 --smoke --from-scratch

# Override parallel environments, budget, device, and random seed.
python train.py --stage 1 --num-envs 8 --timesteps 3000000 --device cuda --seed 0

# Continue from an explicit model name or path.
python train.py --stage 1 --resume-from ppo2_stage1_nav
python train.py --stage 2 --resume-from outputs/models/ppo2_stage2_obstacles_no_followers.zip
python train.py --stage 3 --resume-from outputs/models/ppo2_final_leader-best.zip

# Disable the automatic previous-stage model and initialize a new policy.
python train.py --stage 2 --from-scratch
python train.py --stage 3 --from-scratch
```

### Full-state MDP training

```powershell
# Stage 1: no obstacles, no followers, with zero-padded follower state slots.
python train_mdp.py --stage 1

# Stage 2: obstacles, no followers, automatically load MDP Stage 1.
python train_mdp.py --stage 2

# Stage 3: obstacles and followers, automatically load MDP Stage 2.
python train_mdp.py --stage 3

# Smoke-test any stage on CPU without saving a model.
python train_mdp.py --stage 1 --smoke
python train_mdp.py --stage 2 --smoke --from-scratch
python train_mdp.py --stage 3 --smoke --from-scratch

# Override parallel environments, budget, device, and random seed.
python train_mdp.py --stage 1 --num-envs 8 --timesteps 3000000 --device cuda --seed 0

# Continue from an explicit model name or path.
python train_mdp.py --stage 1 --resume-from mdp_ppo2_stage1_nav
python train_mdp.py --stage 2 --resume-from outputs/models/mdp_ppo2_stage2_obstacles_no_followers.zip
python train_mdp.py --stage 3 --resume-from outputs/models/mdp_ppo2_final_leader-best.zip

# Disable the automatic previous-stage model and initialize a new policy.
python train_mdp.py --stage 2 --from-scratch
python train_mdp.py --stage 3 --from-scratch
```

Do not combine `--from-scratch` and `--resume-from` in one command.

### POMDP evaluation

```powershell
python test.py --stage 1 --render
python test.py --stage 2 --render
python test.py --stage 3 --render
python test.py --stage 3 --model outputs/models/ppo2_final_leader-best.zip --render
python test.py --stage 3 --stochastic --render
python test.py --stage 2 --random-policy --no-plot --max-steps 100 --seed 0
python test.py --stage 3 --output outputs/figures/pomdp_stage3.png --seed 0
```

### Full-state MDP evaluation

```powershell
python test_mdp.py --stage 1 --render
python test_mdp.py --stage 2 --render --seed 1  # 0 2 3 4 12
python test_mdp.py --stage 3 --render
python test_mdp.py --stage 3 --model outputs/models/mdp_ppo2_final_leader-best.zip --render
python test_mdp.py --stage 3 --stochastic --render
python test_mdp.py --stage 2 --random-policy --no-plot --max-steps 100 --seed 0
python test_mdp.py --stage 3 --output outputs/figures/mdp_stage3.png --seed 0
```

### TensorBoard

```powershell
tensorboard --logdir outputs/tensorboard
tensorboard --logdir outputs/tensorboard --port 6006
```
