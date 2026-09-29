"""Sweep position search radii while directly minimizing planar ``||T||_2``."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formation_optimization.spectral_norm_optimizer import (
    minimize_follower_planar_spectral_norm,
)
from utils.formation_utils import get_nominal_configuration


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "results" / "spectral_norm_sweep"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search whether a wider follower position box lowers ||T||_2"
    )
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--n-leaders", type=int, default=2)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument(
        "--search-radii",
        type=float,
        nargs="+",
        default=(3.0, 6.0, 12.0),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--min-degree-singular-value", type=float, default=1.0e-3)
    parser.add_argument("--coordinate-margin", type=float, default=8.0e-1)
    parser.add_argument("--min-pairwise-distance", type=float, default=1.0)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def path_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    if (args.x_min is None) != (args.x_max is None):
        raise ValueError("--x-min and --x-max must be specified together")
    if (args.y_min is None) != (args.y_max is None):
        raise ValueError("--y-min and --y-max must be specified together")
    x_bounds = None if args.x_min is None else (args.x_min, args.x_max)
    y_bounds = None if args.y_min is None else (args.y_min, args.y_max)
    if args.base_config is None:
        nominal = np.asarray(get_nominal_configuration(), dtype=float)
    else:
        base_path = (
            args.base_config
            if args.base_config.is_absolute()
            else ROOT / args.base_config
        )
        nominal = np.asarray(np.load(base_path), dtype=float)

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    best_result = None
    best_key = None
    for search_radius in args.search_radii:
        for seed in args.seeds:
            optimized, result = minimize_follower_planar_spectral_norm(
                nominal,
                n_leaders=args.n_leaders,
                theta=args.theta,
                search_radius=search_radius,
                min_degree_singular_value=args.min_degree_singular_value,
                min_pairwise_distance=args.min_pairwise_distance,
                coordinate_margin=args.coordinate_margin,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                maxiter=args.maxiter,
                popsize=args.popsize,
                seed=seed,
                polish=not args.no_polish,
                verbose=False,
                return_result=True,
            )
            config_name = (
                f"radius_{path_token(search_radius)}_seed_{seed}.npy"
            )
            config_path = output_dir / config_name
            np.save(config_path, optimized)

            record = asdict(result)
            record.pop("configuration")
            record.update({
                "search_radius": float(search_radius),
                "seed": int(seed),
                "configuration_file": str(config_path),
            })
            records.append(record)
            normalized_violation = (
                max(
                    0.0,
                    args.min_degree_singular_value
                    - result.min_degree_singular_value,
                )
                / max(args.min_degree_singular_value, np.finfo(float).eps)
                + max(
                    0.0,
                    args.coordinate_margin
                    - result.min_signed_coordinate_separation,
                )
                / max(args.coordinate_margin, np.finfo(float).eps)
                + (
                    max(
                        0.0,
                        args.min_pairwise_distance
                        - result.minimum_pairwise_distance,
                    )
                    / args.min_pairwise_distance
                    if args.min_pairwise_distance > 0.0
                    else 0.0
                )
            )
            candidate_key = (
                (0, result.optimized_spectral_norm)
                if result.feasible
                else (1, normalized_violation, result.optimized_spectral_norm)
            )
            if best_key is None or candidate_key < best_key:
                best_result = (optimized.copy(), result, record.copy())
                best_key = candidate_key

            print(
                f"radius={search_radius:>6g}, seed={seed:>3d}, "
                f"||T||_2={result.optimized_spectral_norm:.10f}, "
                f"rho(T)={result.optimized_spectral_radius:.10f}, "
                f"sigma_min(D)={result.min_degree_singular_value:.6g}, "
                f"coord_min={result.min_signed_coordinate_separation:.6g}, "
                f"d_min={result.minimum_pairwise_distance:.6g}, "
                f"active_bounds={result.active_bound_coordinates}, "
                f"solver_success={result.solver_success}"
            , flush=True)

    assert best_result is not None
    best_configuration, best_metrics, best_record = best_result
    best_path = output_dir / "best_configuration.npy"
    np.save(best_path, best_configuration)

    summary = {
        "objective": "minimize ||D_ff^-1 A_ff||_2",
        "n_leaders": args.n_leaders,
        "theta": args.theta,
        "search_radii": [float(value) for value in args.search_radii],
        "seeds": [int(value) for value in args.seeds],
        "min_degree_singular_value_limit": args.min_degree_singular_value,
        "coordinate_margin": args.coordinate_margin,
        "min_pairwise_distance_limit": args.min_pairwise_distance,
        "x_bounds": x_bounds,
        "y_bounds": y_bounds,
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "best_configuration_file": str(best_path),
        "best": best_record,
        "runs": records,
    }
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "Best ||T||_2       : "
        f"{best_metrics.optimized_spectral_norm:.10f}"
    )
    print(
        "Corresponding rho(T): "
        f"{best_metrics.optimized_spectral_radius:.10f}"
    )
    print(f"Best configuration  : {best_path}")
    print(f"Sweep summary       : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
