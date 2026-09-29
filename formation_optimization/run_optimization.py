"""Command-line entry point for offline formation optimization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formation_optimization.optimizer import optimize_nominal_configuration
from utils.formation_utils import get_nominal_configuration, get_square_configuration


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline formation configuration optimization")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="可选的 .npy 基准构型；不指定时使用论文基准构型",
    )
    parser.add_argument("--n-leaders", type=int, default=2)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--rho-limit", type=float, default=0.95)
    parser.add_argument("--search-radius", type=float, default=10.0)
    parser.add_argument("--coordinate-margin", type=float, default=8.0e-1)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--min-degree-singular-value", type=float, default=1.0e-3)
    parser.add_argument("--x-min", type=float, default=None)       # 跟随者世界坐标 \(x\) 的绝对范围
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=300)     # 差分进化算法的最大迭代代数
    parser.add_argument("--popsize", type=int, default=15)   # 差分进化算法的种群规模系数。
    parser.add_argument("--seed", type=int, default=0)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="显式指定多个随机种子，例如 --seeds 0 1 2 3",
    )
    seed_group.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="从 --seed 开始连续运行指定数量的随机种子",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--non-strict", action="store_true", help="保存最优不可行结果而不是报错")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.x_min is None) != (args.x_max is None):
        raise ValueError("--x-min and --x-max must be specified together")
    if (args.y_min is None) != (args.y_max is None):
        raise ValueError("--y-min and --y-max must be specified together")
    x_bounds = None if args.x_min is None else (args.x_min, args.x_max)
    y_bounds = None if args.y_min is None else (args.y_min, args.y_max)
    if args.num_seeds is not None and args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.seeds is not None:
        seeds = list(dict.fromkeys(args.seeds))
    elif args.num_seeds is not None:
        seeds = list(range(args.seed, args.seed + args.num_seeds))
    else:
        seeds = [args.seed]
    if any(seed < 0 for seed in seeds):
        raise ValueError("random seeds must be non-negative integers")

    if args.base_config is None:
        nominal = np.asarray(get_square_configuration(), dtype=float)
    else:
        base_path = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
        nominal = np.asarray(np.load(base_path), dtype=float)

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_output_dir = output_dir / "seeds"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    print(f"Running {len(seeds)} seed(s): {seeds}")
    for run_index, seed in enumerate(seeds, start=1):
        print(f"\n[{run_index}/{len(seeds)}] seed={seed}")
        optimized, result = optimize_nominal_configuration(
            nominal,
            n_leaders=args.n_leaders,
            theta=args.theta,
            spectral_radius_limit=args.rho_limit,
            search_radius=args.search_radius,
            min_degree_singular_value=args.min_degree_singular_value,
            coordinate_margin=args.coordinate_margin,
            minimum_pairwise_distance=args.min_distance,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=seed,
            # A failed seed must not abort the remaining batch.
            strict=False,
            return_result=True,
            verbose=True,
        )
        seed_config_path = seed_output_dir / f"seed_{seed}_configuration.npy"
        seed_summary_path = seed_output_dir / f"seed_{seed}_summary.json"
        np.save(seed_config_path, optimized)

        run_summary = asdict(result)
        run_summary["configuration"] = np.asarray(result.configuration).tolist()
        run_summary.update(
            {
                "seed": seed,
                "configuration_file": str(seed_config_path),
            }
        )
        seed_summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        runs.append((seed, optimized, result, run_summary))

    feasible_runs = [run for run in runs if run[2].success]
    if feasible_runs:
        best_seed, optimized, result, best_run_summary = min(
            feasible_runs,
            key=lambda run: (
                run[2].objective_value,
                run[2].optimized_spectral_radius,
            ),
        )
        selection_reason = "minimum objective among feasible runs"
    else:
        def normalized_violation(run) -> float:
            candidate = run[2]
            rho_scale = max(args.rho_limit, np.finfo(float).eps)
            coordinate_scale = max(
                args.coordinate_margin,
                np.finfo(float).eps,
            )
            distance_violation = 0.0
            if args.min_distance > 0.0:
                distance_violation = max(
                    0.0,
                    args.min_distance - candidate.minimum_pairwise_distance,
                ) / args.min_distance
            singular_value_scale = max(
                args.min_degree_singular_value,
                np.finfo(float).eps,
            )
            return float(
                max(
                    0.0,
                    candidate.optimized_spectral_radius - args.rho_limit,
                )
                / rho_scale
                + max(
                    0.0,
                    args.coordinate_margin
                    - candidate.min_signed_coordinate_separation,
                )
                / coordinate_scale
                + distance_violation
                + max(
                    0.0,
                    args.min_degree_singular_value
                    - candidate.min_degree_singular_value,
                )
                / singular_value_scale
            )

        best_seed, optimized, result, best_run_summary = min(
            runs,
            key=lambda run: (
                normalized_violation(run),
                run[2].objective_value,
            ),
        )
        selection_reason = "minimum normalized violation; no feasible run found"

    config_path = output_dir / "optimized_configuration.npy"
    summary_path = output_dir / "optimization_summary.json"
    np.save(config_path, optimized)

    summary = dict(best_run_summary)
    summary.update(
        {
            "best_seed": best_seed,
            "selection_reason": selection_reason,
            "number_of_runs": len(runs),
            "number_of_feasible_runs": len(feasible_runs),
            "requested_seeds": seeds,
            "rho_limit": args.rho_limit,
            "theta": args.theta,
            "search_radius": args.search_radius,
            "coordinate_margin": args.coordinate_margin,
            "minimum_pairwise_distance_limit": args.min_distance,
            "min_degree_singular_value_limit": (
                args.min_degree_singular_value
            ),
            "x_bounds": x_bounds,
            "y_bounds": y_bounds,
            "configuration_file": str(config_path),
            "runs": [run[3] for run in runs],
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved optimized configuration: {config_path}")
    print(f"Saved optimization summary:     {summary_path}")
    print(
        f"Best seed: {best_seed}; feasible runs: "
        f"{len(feasible_runs)}/{len(runs)}; selection: {selection_reason}"
    )
    if not feasible_runs and not args.non_strict:
        raise RuntimeError(
            "No random seed found a feasible configuration. All per-seed "
            f"results were saved under {seed_output_dir}. Use --non-strict "
            "to accept the least-infeasible main output."
        )


if __name__ == "__main__":
    main()
