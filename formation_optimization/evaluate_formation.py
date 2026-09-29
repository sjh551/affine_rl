"""Run the existing PyBullet environment with a saved formation configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from POMDP_env import FormationPyBulletEnv
from utils.formation_utils import (
    get_nominal_configuration,
    get_five_agent_topology,
    get_square_topology,
    get_topology,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "results" / "optimized_configuration.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved formation in PyBullet")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="运行指定控制周期；不指定时持续运行，按 Q 或 Esc 退出",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--no-walls",
        action="store_true",
        help="不生成四周实体墙，并关闭 Wall_Out 边界终止",
    )
    parser.add_argument("--control-mode", choices=("centralized", "distributed", "distributed_iterative"), default="distributed")
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到构型文件：{config_path}。请先运行 run_optimization.py。"
        )
    configuration = np.load(config_path)
    if configuration.ndim != 2 or configuration.shape[1] < 3:
        raise ValueError("构型文件必须是形状为 (n_agents, >=3) 的数组")
    if len(configuration) == len(get_nominal_configuration()):
        edges = get_topology()
    elif len(configuration) == 4:
        edges = get_square_topology()
    elif len(configuration) == 5:
        edges = get_five_agent_topology()
    else:
        raise ValueError(
            "无法自动选择该构型的拓扑；请在 evaluate_formation.py 中显式配置 formation_edges。"
        )
    env = FormationPyBulletEnv(
        render_mode=args.render,
        enable_obstacles=False,
        draw_line=True,
        follower_control_mode=args.control_mode,
        follower_inner_iterations=args.inner_iterations,
        nominal_configuration=configuration,
        formation_edges=edges,
        print_follower_spectrum=True,
        max_steps=None,
        enable_walls=not args.no_walls,
    )
    try:
        env.reset(seed=args.seed)
        print("================================================")
        print("方向键：控制 Leader A 的 x/y 运动")
        print("W/S：控制 Leader B 的 y 运动；A/D：控制 Leader B 的 x 运动")
        print("R：重置环境；Q 或 Esc：退出")
        print(f"已加载构型：{config_path}")
        print(f"智能体数量：{len(configuration)}（Leader=2，Follower={len(configuration) - 2}）")
        print("================================================")

        step = 0
        while args.steps is None or step < args.steps:
            keys = p.getKeyboardEvents() if args.render else {}
            if (
                ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED
            ) or (
                27 in keys and keys[27] & p.KEY_WAS_TRIGGERED
            ):
                break

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                action[1] = 0.5
            if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                action[1] = -0.5
            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                action[0] = -0.5
            if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                action[0] = 0.5
            if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
                action[3] = 0.5
            if ord("s") in keys and keys[ord("s")] & p.KEY_IS_DOWN:
                action[3] = -0.5
            if ord("a") in keys and keys[ord("a")] & p.KEY_IS_DOWN:
                action[2] = -0.5
            if ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
                action[2] = 0.5

            if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                env.reset(seed=args.seed)
                step = 0
                print("环境已重置")
                continue

            _, _, terminated, truncated, info = env.step(action)
            step += 1
            if step == 1 or step % 50 == 0:
                print(
                    f"step={step:04d}, reward={info.get('reward', float('nan')):.6f}, "
                    f"terminated={terminated}, truncated={truncated}"
                )
            if terminated or truncated:
                if args.steps is None:
                    print("当前回合结束，自动重置并继续运行")
                    env.reset(seed=args.seed)
                    step = 0
                else:
                    break
            if args.render:
                time.sleep(0.03)
    finally:
        env.close()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
