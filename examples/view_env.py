"""View and drive the training environment in PyBullet, without a PPO model."""

import argparse
import time

import numpy as np
import pybullet as p

from controllers.distributed import FOLLOWER_CONTROL_MODES
from POMDP_env import FormationPyBulletEnv


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


def run_viewer(env, seed, demo):
    period = 1 / env.metadata["render_fps"]
    client = env.physics_client_id
    n_leaders = env.n_leaders
    demo_command = shear_action(env.formation.nominal[:n_leaders])
    print(f"Follower control: {env.follower_control_mode}")
    print("W/A/S/D: move | 0: all leaders | 1/2/3: select leader")
    print("Space: pause/resume | R: reset with the same seed | Q: quit")
    print("Leaders are green, followers are blue, and the green sphere is the goal.")

    env.reset(seed=seed)
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
            env.reset(seed=seed)
            paused = finished = False
            steps = 0
            demo_remaining = _DEMO_STEPS if demo else 0
            episode_return = 0.0
            print(f"Reset (seed={seed}).")
        for index in range(n_leaders + 1):
            if pressed(events, str(index)):
                selection = index
                print(f"Selected leader {index}." if index else "Selected all leaders.")
                break
        if pressed(events, " ") and not finished:
            paused = not paused
            print("Paused." if paused else "Resumed.")

        if not (paused or finished):
            if demo_remaining:
                action = demo_command
                demo_remaining -= 1
            else:
                action = manual_action(events, n_leaders, selection)
            _, reward, terminated, truncated, info = env.step(action)
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
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the initial scene and every reset.")
    parser.add_argument("--no-obstacles", action="store_true", help="Use the open training course.")
    parser.add_argument("--demo", action="store_true", help="Play a four-second shear maneuver on reset.")
    parser.add_argument("--follower-control", choices=FOLLOWER_CONTROL_MODES, default="centralized",
                        help="Follower velocity controller (default: centralized).")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")

    env = FormationPyBulletEnv(render_mode="human", enable_obstacles=not args.no_obstacles,
                               follower_control_mode=args.follower_control)
    try:
        run_viewer(env, args.seed, args.demo)
    except KeyboardInterrupt:
        pass
    except p.error:
        if p.isConnected(physicsClientId=env.physics_client_id):
            raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
