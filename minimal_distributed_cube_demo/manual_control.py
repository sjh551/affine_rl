"""Convenience entry point for the interactive minimal cube demo."""

import sys

from distributed_cube_demo import parse_args, run


if __name__ == "__main__":
    if "--manual" not in sys.argv:
        sys.argv.append("--manual")
    # Do not excite an unstable mode before the user has pressed a movement
    # key. The user can still request a perturbation explicitly on the CLI.
    if "--perturbation" not in sys.argv:
        sys.argv.extend(["--perturbation", "0"])
    raise SystemExit(run(parse_args()))
