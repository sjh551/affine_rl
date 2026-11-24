"""Drive the formation manually or replay a PPO policy in PyBullet."""

import argparse
from pathlib import Path
import time

import numpy as np
import pybullet as p

from robot_rl.tasks.formation_navigation.env import FormationPyBulletEnv
from robot_rl.training.config import add_environment_arguments, experiment_from_args


_DEMO_STEPS = 40


def pressed(events, key):
    return bool(events.get(ord(key), 0) & p.KEY_WAS_TRIGGERED)


def manual_action(events, leader_count, selection):
    def held(key):
        return bool(events.get(ord(key), 0) & p.KEY_IS_DOWN)

    direction = np.array([
        int(held("w")) - int(held("s")),
        int(held("a")) - int(held("d")),
    ], dtype=np.float32)
    direction *= 0.6 / max(1.0, np.linalg.norm(direction))
    commands = np.zeros((leader_count, 2), dtype=np.float32)
    if selection == 0:
        commands[:] = direction
    else:
        commands[selection - 1] = direction
    return commands.ravel()


def shear_action(nominal_leaders):
    velocity_gradient = np.array([[0.02, 0.08], [0.05, -0.05]])
    return (np.array([0.45, 0.0]) + nominal_leaders @ velocity_gradient).astype(np.float32).ravel()


def run_viewer(env, seed, demo, policy=None):
    period = 1 / env.metadata["render_fps"]
    client = env.physics_client_id
    n_leaders = env.n_leaders
    demo_command = shear_action(env.formation.nominal[:n_leaders])
    print(f"Map: {env.obstacle_map or 'open'}")
    print(f"Follower control: {env.follower_control_mode}")
    if env.follower_control_mode == "distributed":
        print(f"Neighbor velocity: {env.neighbor_velocity_source}")
    if policy is None:
        print("W/A/S/D: move | 0: all leaders | 1/2/3: select leader")
    else:
        print("Policy playback is active.")
    print("Space: pause/resume | R: reset with the same seed | Q: quit")
    print("Leaders are green, followers are blue, and the green sphere is the goal.")

    obs, _ = env.reset(seed=seed)
    selection = 0
    paused = finished = False
    steps = 0
    demo_remaining = _DEMO_STEPS if demo else 0
    episode_return = 0.0
    while p.isConnected(physicsClientId=client):
        start = time.monotonic()
        events = p.getKeyboardEvents(physicsClientId=client)
        if pressed(events, "q"):
            break
        if pressed(events, "r"):
            obs, _ = env.reset(seed=seed)
            paused = finished = False
            steps = 0
            demo_remaining = _DEMO_STEPS if demo else 0
            episode_return = 0.0
            print(f"Reset (seed={seed}).")
        if policy is None:
            for index in range(n_leaders + 1):
                if pressed(events, str(index)):
                    selection = index
                    print(f"Selected leader {index}." if index else "Selected all leaders.")
                    break
        if pressed(events, " ") and not finished:
            paused = not paused
            print("Paused." if paused else "Resumed.")

        if not (paused or finished):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            elif demo_remaining:
                action = demo_command
                demo_remaining -= 1
            else:
                action = manual_action(events, n_leaders, selection)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            episode_return += reward
            finished = terminated or truncated
            if steps % env.metadata["render_fps"] == 0 or finished:
                print(f"Step {steps:4d} | goal {info['goal_distance']:.2f} m | "
                      f"tracking RMS {info['formation_error']:.3f} m | "
                      f"shape {info['shape_error']:.3f} m | return {episode_return:.2f}")
            if finished:
                print(f"{info['reason']}. Press R to reset.")
            elif demo and steps == _DEMO_STEPS:
                paused = True
                print("Shear demo finished. Press Space for manual control or R to replay.")

        time.sleep(max(0.0, period - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_arguments(parser)
    parser.add_argument("--seed", type=int, help="Seed used for the initial scene and every reset.")
    parser.add_argument("--device", default="cpu", help="Device for policy playback.")
    parser.add_argument("--no-obstacles", action="store_true", help="Use the open training course.")
    playback = parser.add_mutually_exclusive_group()
    playback.add_argument("--demo", action="store_true", help="Play a four-second shear maneuver on reset.")
    playback.add_argument("--model", type=Path, help="Replay a trained policy instead of keyboard input.")
    args = parser.parse_args()
    if args.no_obstacles:
        if args.stage == 2:
            parser.error("--no-obstacles cannot be combined with --stage 2.")
        args.stage = 1
    default = Path(__file__).resolve().parents[1] / "configs/experiments/formation_ppo.yaml"
    experiment = experiment_from_args(parser, args, default, model=args.model)
    policy = None
    if args.model:
        import torch

        from robot_rl.training.policy import load_policy

        torch.set_num_threads(min(torch.get_num_threads(), 4))
        policy = load_policy(args.model, device=args.device)
    env = FormationPyBulletEnv(render_mode="human", **experiment.environment.env_kwargs())
    try:
        run_viewer(env, experiment.training.seed, args.demo, policy=policy)
    except KeyboardInterrupt:
        pass
    except p.error:
        if p.isConnected(physicsClientId=env.physics_client_id):
            raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
