# MAPPO (Minimal Version)

This directory contains a minimal MAPPO integration built on top of the existing project environment, without modifying the original environment implementation.

## Files

- `mappo_env.py`: Wraps `FormationPyBulletEnv` into a 2-agent leader interface with centralized global state.
- `mappo_core.py`: Minimal MAPPO core (actor, critic, rollout buffer, PPO-style update).
- `train_mappo.py`: Training entrypoint.

## Install dependencies

At minimum, install PyTorch in your active environment.

Example (CPU):

```bash
pip install torch
```

If you use CUDA, install a matching PyTorch build from the official selector:
https://pytorch.org/get-started/locally/

## Run training

From project root:

```bash
python models/MAPPO/train_mappo.py --total-timesteps 200000 --rollout-horizon 1024
```

Enable obstacles:

```bash
python models/MAPPO/train_mappo.py --enable-obstacles
```

Saved checkpoints are written to:

- `models/MAPPO/checkpoints/`

## Notes

- This is a minimal baseline for quick integration and experimentation.
- The algorithm uses parameter sharing for the two leader actors and a centralized critic.
