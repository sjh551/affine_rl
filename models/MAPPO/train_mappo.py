import argparse
import os
import random
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.MAPPO.mappo_core import MAPPOConfig, MAPPOTrainer
from models.MAPPO.mappo_env import LeaderFollowerMAPPOEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO for leader-follower control")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--rollout-horizon", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-obstacles", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-dir", type=str, default=os.path.join(CURRENT_DIR, "checkpoints"))
    parser.add_argument("--save-interval", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    env = LeaderFollowerMAPPOEnv(
        render_mode=args.render,
        enable_obstacles=args.enable_obstacles,
    )

    config = MAPPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        update_epochs=args.update_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        rollout_horizon=args.rollout_horizon,
    )

    trainer = MAPPOTrainer(env=env, config=config, device=args.device)

    num_updates = max(1, args.total_timesteps // args.rollout_horizon)
    print(f"[MAPPO] device={args.device}, updates={num_updates}, horizon={args.rollout_horizon}")

    for update in range(1, num_updates + 1):
        buffer, rollout_stats = trainer.collect_rollout()
        train_stats = trainer.update(buffer)

        print(
            f"[Update {update}/{num_updates}] "
            f"rollout_reward={rollout_stats['mean_rollout_reward']:.2f} "
            f"episodes={rollout_stats['episodes_finished']} "
            f"actor_loss={train_stats['actor_loss']:.4f} "
            f"critic_loss={train_stats['critic_loss']:.4f}"
        )

        if update % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"mappo_update_{update}.pt")
            trainer.save(ckpt_path)
            print(f"[MAPPO] checkpoint saved to: {ckpt_path}")

    final_path = os.path.join(args.save_dir, "mappo_final.pt")
    trainer.save(final_path)
    print(f"[MAPPO] final model saved to: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
