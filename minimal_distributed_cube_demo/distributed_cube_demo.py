"""Minimal PyBullet demo for the paper's distributed follower controller.

This file intentionally does not import the RL environments, URDF models, or
wheel kinematics from the main project.  Every agent is a box created through
PyBullet's primitive-shape API and is commanded directly by world-frame planar
velocity [vx, vy, yaw_rate].

The control cycle is synchronous and causal:
  1. sample one immutable snapshot of every agent;
  2. each follower computes its own command from only its local triplet;
  3. apply all commands simultaneously for one control interval.

No M_ff inverse or centralized simultaneous velocity solve is used.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pybullet as pb


N_LEADERS = 2
STATE_DIM = 3  # g_i = [x_i, y_i, phi_i]


def nominal_configuration() -> np.ndarray:
    """The same 2-leader/2-follower nominal formation used by POMDP_env.py."""
    x, y = 3.0, 1.0
    return np.array(
        [
            [x, y, 0.0],
            [y, -x, np.pi / 2.0],
            [-y, x, -np.pi / 2.0],
            [-x, -y, np.pi],
        ],
        dtype=float,
    )


def compute_w_matrices(
    g_i_nom: np.ndarray,
    g_j_nom: np.ndarray,
    g_k_nom: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build W_jk and W_ki from equations (13)-(14) of the paper."""
    p_jk_nom = g_j_nom[:2] - g_k_nom[:2]
    phi_jk_nom = g_j_nom[2] - g_k_nom[2]
    p_ki_nom = g_k_nom[:2] - g_i_nom[:2]
    phi_ki_nom = g_k_nom[2] - g_i_nom[2]

    c, s = np.cos(theta), np.sin(theta)
    r_theta = np.array([[c, -s], [s, c]], dtype=float)
    theta_matrix = np.eye(STATE_DIM)
    theta_matrix[:2, :2] = r_theta

    p_jk_projected = r_theta.T @ p_jk_nom
    p_ki_projected = r_theta.T @ p_ki_nom
    w_jk = np.diag([p_jk_projected[0], p_jk_projected[1], phi_jk_nom])
    w_ki = np.diag([p_ki_projected[0], p_ki_projected[1], phi_ki_nom])
    return w_jk @ theta_matrix.T, w_ki @ theta_matrix.T


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable sampled states available at the start of a control cycle."""

    g: np.ndarray
    g_dot: np.ndarray


class DistributedFollower:
    """One local controller; it knows only follower k and its two neighbors."""

    def __init__(
        self,
        follower_id: int,
        neighbor_i: int,
        neighbor_j: int,
        g_nom: np.ndarray,
        theta: float,
    ) -> None:
        self.k = follower_id
        self.i = neighbor_i
        self.j = neighbor_j
        self.w_jk, self.w_ki = compute_w_matrices(
            g_nom[self.i], g_nom[self.j], g_nom[self.k], theta
        )
        self.w_kk = self.w_jk + self.w_ki
        if np.linalg.matrix_rank(self.w_kk) < STATE_DIM:
            raise ValueError(f"Follower {self.k}: W_kk is singular")
        self.w_kk_inv = np.linalg.inv(self.w_kk)

    def command(self, snapshot: StateSnapshot, gain: float) -> np.ndarray:
        """Paper equation (29), evaluated only from the sampled local data."""
        g_i, g_j, g_k = snapshot.g[self.i], snapshot.g[self.j], snapshot.g[self.k]
        g_dot_i = snapshot.g_dot[self.i]
        g_dot_j = snapshot.g_dot[self.j]

        local_input = (
            self.w_jk @ (gain * (g_i - g_k) + g_dot_i)
            + self.w_ki @ (gain * (g_j - g_k) + g_dot_j)
        )
        return self.w_kk_inv @ local_input

    def constraint_residual(self, g: np.ndarray) -> np.ndarray:
        return self.w_jk @ (g[self.i] - g[self.k]) + self.w_ki @ (
            g[self.j] - g[self.k]
        )


def make_local_controllers(g_nom: np.ndarray, theta: float) -> list[DistributedFollower]:
    # DEP: leader 0 <-> follower 2 <-> follower 3 <-> leader 1.
    # Each tuple is (neighbor i, neighbor j, follower k).
    triplets = [(0, 3, 2), (1, 2, 3)]
    return [DistributedFollower(k, i, j, g_nom, theta) for i, j, k in triplets]


def follower_iteration_matrix(
    controllers: Iterable[DistributedFollower],
) -> np.ndarray:
    """Construct T only for diagnostics; it is never used to compute commands."""
    controllers = list(controllers)
    follower_ids = [controller.k for controller in controllers]
    follower_index = {agent_id: idx for idx, agent_id in enumerate(follower_ids)}
    degree = np.zeros((STATE_DIM * len(controllers),) * 2)
    adjacency = np.zeros_like(degree)

    for row, controller in enumerate(controllers):
        row_slice = slice(row * STATE_DIM, (row + 1) * STATE_DIM)
        degree[row_slice, row_slice] = controller.w_kk
        if controller.i in follower_index:
            col = follower_index[controller.i]
            adjacency[row_slice, col * STATE_DIM : (col + 1) * STATE_DIM] += (
                controller.w_jk
            )
        if controller.j in follower_index:
            col = follower_index[controller.j]
            adjacency[row_slice, col * STATE_DIM : (col + 1) * STATE_DIM] += (
                controller.w_ki
            )
    return np.linalg.solve(degree, adjacency)


def create_box_agent(position: np.ndarray, color: list[float]) -> int:
    half_extents = [0.22, 0.22, 0.10]
    collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents)
    visual = pb.createVisualShape(
        pb.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )
    quaternion = pb.getQuaternionFromEuler([0.0, 0.0, float(position[2])])
    body_id = pb.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[float(position[0]), float(position[1]), 0.3],
        baseOrientation=quaternion,
    )
    # Agents are point-like for this controller-only experiment.  Disabling
    # contacts prevents collision impulses from being mistaken for instability.
    pb.setCollisionFilterGroupMask(body_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
    pb.changeDynamics(body_id, -1, linearDamping=0.0, angularDamping=0.0)
    return body_id


def read_snapshot(body_ids: list[int]) -> StateSnapshot:
    g = np.zeros((len(body_ids), STATE_DIM))
    g_dot = np.zeros_like(g)
    for index, body_id in enumerate(body_ids):
        position, quaternion = pb.getBasePositionAndOrientation(body_id)
        linear_velocity, angular_velocity = pb.getBaseVelocity(body_id)
        g[index] = [position[0], position[1], pb.getEulerFromQuaternion(quaternion)[2]]
        g_dot[index] = [linear_velocity[0], linear_velocity[1], angular_velocity[2]]
    return StateSnapshot(g=g, g_dot=g_dot)


def apply_planar_velocity(body_id: int, command: np.ndarray) -> None:
    """Ideal omnidirectional actuation in the world frame; no wheels or URDF."""
    pb.resetBaseVelocity(
        body_id,
        linearVelocity=[float(command[0]), float(command[1]), 0.0],
        angularVelocity=[0.0, 0.0, float(command[2])],
    )


def reset_box_agents(
    body_ids: list[int], initial_g: np.ndarray, initial_velocity: np.ndarray | None = None
) -> None:
    """Reset all primitive agents without rebuilding the PyBullet world."""
    if initial_velocity is None:
        initial_velocity = np.zeros_like(initial_g)
    for agent_id, body_id in enumerate(body_ids):
        quaternion = pb.getQuaternionFromEuler([0.0, 0.0, float(initial_g[agent_id, 2])])
        pb.resetBasePositionAndOrientation(
            body_id,
            [float(initial_g[agent_id, 0]), float(initial_g[agent_id, 1]), 0.3],
            quaternion,
        )
        apply_planar_velocity(body_id, initial_velocity[agent_id])


def manual_leader_commands(
    keys: dict[int, int], linear_speed: float, angular_speed: float
) -> np.ndarray:
    """Read the same leader key layout as debug_env_manual.py."""
    commands = np.zeros((N_LEADERS, STATE_DIM))

    # Leader A: arrow keys for world-frame translation; 1/2 for yaw.
    if keys.get(pb.B3G_UP_ARROW, 0) & pb.KEY_IS_DOWN:
        commands[0, 1] += linear_speed
    if keys.get(pb.B3G_DOWN_ARROW, 0) & pb.KEY_IS_DOWN:
        commands[0, 1] -= linear_speed
    if keys.get(pb.B3G_LEFT_ARROW, 0) & pb.KEY_IS_DOWN:
        commands[0, 0] -= linear_speed
    if keys.get(pb.B3G_RIGHT_ARROW, 0) & pb.KEY_IS_DOWN:
        commands[0, 0] += linear_speed
    if keys.get(ord("1"), 0) & pb.KEY_IS_DOWN:
        commands[0, 2] += angular_speed
    if keys.get(ord("2"), 0) & pb.KEY_IS_DOWN:
        commands[0, 2] -= angular_speed

    # Leader B: WASD for world-frame translation; Q/E for yaw.
    if keys.get(ord("w"), 0) & pb.KEY_IS_DOWN:
        commands[1, 1] += linear_speed
    if keys.get(ord("s"), 0) & pb.KEY_IS_DOWN:
        commands[1, 1] -= linear_speed
    if keys.get(ord("a"), 0) & pb.KEY_IS_DOWN:
        commands[1, 0] -= linear_speed
    if keys.get(ord("d"), 0) & pb.KEY_IS_DOWN:
        commands[1, 0] += linear_speed
    if keys.get(ord("q"), 0) & pb.KEY_IS_DOWN:
        commands[1, 2] += angular_speed
    if keys.get(ord("e"), 0) & pb.KEY_IS_DOWN:
        commands[1, 2] -= angular_speed
    return commands


def save_gui_screenshot(output_directory: Path) -> Path:
    """Save the current PyBullet debug camera image when P is pressed."""
    camera = pb.getDebugVisualizerCamera()
    width, height = int(camera[0]), int(camera[1])
    image = pb.getCameraImage(
        width,
        height,
        viewMatrix=camera[2],
        projectionMatrix=camera[3],
        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
    )
    rgba = np.asarray(image[2], dtype=np.uint8).reshape(height, width, 4)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"manual_{datetime.now():%Y%m%d_%H%M%S}.png"
    import matplotlib.pyplot as plt

    plt.imsave(path, rgba)
    return path


def apf_repulsion(
    position: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    influence_distance: float,
    gain: float,
    max_speed: float,
) -> np.ndarray:
    """Optional standard APF term, added after the formation command."""
    offset = position - obstacle_center
    center_distance = float(np.linalg.norm(offset))
    surface_distance = center_distance - obstacle_radius
    if center_distance < 1.0e-12 or surface_distance >= influence_distance:
        return np.zeros(2)

    safe_distance = max(surface_distance, 1.0e-3)
    direction = offset / center_distance
    magnitude = gain * (1.0 / safe_distance - 1.0 / influence_distance) / (
        safe_distance**2
    )
    repulsion = magnitude * direction
    norm = float(np.linalg.norm(repulsion))
    if norm > max_speed:
        repulsion *= max_speed / norm
    return repulsion


def draw_topology(body_ids: list[int]) -> None:
    for first, second in [(0, 2), (2, 3), (3, 1)]:
        first_pos = pb.getBasePositionAndOrientation(body_ids[first])[0]
        second_pos = pb.getBasePositionAndOrientation(body_ids[second])[0]
        pb.addUserDebugLine(first_pos, second_pos, [0.25, 0.25, 0.25], 2.0, 0.08)


def save_plot(
    output_path: Path,
    times: list[float],
    residuals: list[np.ndarray],
    speeds: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    residual_array = np.asarray(residuals)
    speed_array = np.asarray(speeds)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].semilogy(times, np.maximum(residual_array[:, 0], 1.0e-16), label="follower 2")
    axes[0].semilogy(times, np.maximum(residual_array[:, 1], 1.0e-16), label="follower 3")
    axes[0].set_ylabel("local constraint residual")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    for agent_id in range(speed_array.shape[1]):
        axes[1].semilogy(
            times,
            np.maximum(speed_array[:, agent_id], 1.0e-16),
            label=f"agent {agent_id}",
        )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("planar speed [m/s]")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(ncol=2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> int:
    use_gui = args.gui or args.manual
    client = pb.connect(pb.GUI if use_gui else pb.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet")

    try:
        pb.resetSimulation()
        pb.setGravity(0.0, 0.0, 0.0)
        pb.setTimeStep(args.physics_timestep)
        if use_gui:
            pb.resetDebugVisualizerCamera(
                cameraDistance=11.0,
                cameraYaw=35.0,
                cameraPitch=-70.0,
                cameraTargetPosition=[1.0, 0.0, 0.0],
            )

        g_nom = nominal_configuration()
        initial_g = g_nom.copy()
        initial_g[2, 0] += args.perturbation
        colors = [
            [0.90, 0.20, 0.20, 1.0],
            [0.95, 0.55, 0.10, 1.0],
            [0.15, 0.45, 0.95, 1.0],
            [0.10, 0.75, 0.35, 1.0],
        ]
        body_ids = [
            create_box_agent(initial_g[index], colors[index])
            for index in range(len(initial_g))
        ]

        obstacle_center = np.array([5.0, 0.0])
        obstacle_radius = 0.55
        if args.enable_apf:
            obstacle_visual = pb.createVisualShape(
                pb.GEOM_CYLINDER,
                radius=obstacle_radius,
                length=0.5,
                rgbaColor=[0.25, 0.25, 0.25, 1.0],
            )
            pb.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=obstacle_visual,
                basePosition=[obstacle_center[0], obstacle_center[1], 0.25],
            )

        controllers = make_local_controllers(g_nom, args.theta)
        iteration_matrix = follower_iteration_matrix(controllers)
        eigenvalues = np.linalg.eigvals(iteration_matrix)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        control_timestep = args.physics_timestep * args.substeps

        print("=" * 72)
        print("Minimal distributed cube demo (no URDF, no RL environment)")
        print(f"physics timestep       : {args.physics_timestep:.8f} s")
        print(f"physics substeps       : {args.substeps}")
        print(f"control timestep       : {control_timestep:.8f} s")
        print(f"formation gain         : {args.gain:.6g}")
        print(f"initial perturbation   : {args.perturbation:.3e} m")
        print(f"leader velocity        : [{args.leader_vx}, {args.leader_vy}, {args.leader_omega}]")
        print(f"leader input mode      : {'manual keyboard' if args.manual else 'constant'}")
        print(f"APF                    : {'enabled' if args.enable_apf else 'disabled'}")
        print("follower iteration eigenvalues:")
        print(np.array2string(eigenvalues, precision=8, suppress_small=True))
        print(f"spectral radius rho(T) : {spectral_radius:.8f}")
        print("=" * 72)

        if args.manual:
            print("Leader A: arrow keys move; [1]/[2] rotate")
            print("Leader B: [W][A][S][D] move; [Q]/[E] rotate")
            print("[R] reset formation; [P] save screenshot")
            print("Release all movement keys to command zero leader velocity.")
            print("=" * 72)

        leader_command = np.array(
            [args.leader_vx, args.leader_vy, args.leader_omega], dtype=float
        )
        times: list[float] = []
        residual_history: list[np.ndarray] = []
        speed_history: list[np.ndarray] = []
        n_control_steps = int(np.ceil(args.duration / control_timestep))
        diverged = False
        paused_after_divergence = False
        step = 0
        last_snapshot = read_snapshot(body_ids)

        while pb.isConnected() and (args.manual or step < n_control_steps):
            keys = pb.getKeyboardEvents() if args.manual else {}
            if args.manual and keys.get(ord("r"), 0) & pb.KEY_WAS_TRIGGERED:
                reset_box_agents(body_ids, initial_g)
                times.clear()
                residual_history.clear()
                speed_history.clear()
                step = 0
                diverged = False
                paused_after_divergence = False
                print("Formation reset.")
                continue
            if args.manual and keys.get(ord("p"), 0) & pb.KEY_WAS_TRIGGERED:
                screenshot = save_gui_screenshot(Path(__file__).parent / "screenshots")
                print(f"Screenshot saved: {screenshot.resolve()}")
            if paused_after_divergence:
                time.sleep(0.03)
                continue

            snapshot = read_snapshot(body_ids)
            last_snapshot = snapshot
            commands = np.zeros_like(snapshot.g_dot)
            if args.manual:
                commands[:N_LEADERS] = manual_leader_commands(
                    keys, args.manual_speed, args.manual_angular_speed
                )
            else:
                commands[:N_LEADERS] = leader_command

            # Each object executes an independent local computation against the
            # same sampled state.  A newly computed follower command is never
            # visible to the other follower in this cycle.
            for controller in controllers:
                commands[controller.k] = controller.command(snapshot, args.gain)

            if args.enable_apf:
                for follower_id in range(N_LEADERS, len(body_ids)):
                    commands[follower_id, :2] += apf_repulsion(
                        snapshot.g[follower_id, :2],
                        obstacle_center,
                        obstacle_radius,
                        args.apf_influence,
                        args.apf_gain,
                        args.apf_max_speed,
                    )

            command_magnitudes = np.linalg.norm(commands, axis=1)
            residual_norms = np.array(
                [
                    np.linalg.norm(controller.constraint_residual(snapshot.g))
                    for controller in controllers
                ]
            )
            times.append(step * control_timestep)
            residual_history.append(residual_norms)
            speed_history.append(np.linalg.norm(snapshot.g_dot[:, :2], axis=1))

            if not np.all(np.isfinite(commands)) or np.max(command_magnitudes) > args.divergence_threshold:
                diverged = True
                print(
                    f"DIVERGENCE threshold reached at t={step * control_timestep:.6f} s; "
                    f"max |command|={np.max(command_magnitudes):.6g}"
                )
                if args.manual:
                    for body_id in body_ids:
                        apply_planar_velocity(body_id, np.zeros(STATE_DIM))
                    paused_after_divergence = True
                    print("Simulation paused. Press [R] to reset and continue.")
                    continue
                break

            for body_id, command in zip(body_ids, commands):
                apply_planar_velocity(body_id, command)

            for _ in range(args.substeps):
                pb.stepSimulation()
                if use_gui:
                    draw_topology(body_ids)
                if args.realtime or args.manual:
                    time.sleep(args.physics_timestep)
            step += 1

        final_snapshot = read_snapshot(body_ids) if pb.isConnected() else last_snapshot
        final_residuals = [
            np.linalg.norm(controller.constraint_residual(final_snapshot.g))
            for controller in controllers
        ]
        print(f"result                  : {'diverged' if diverged else 'completed'}")
        print(f"simulated time          : {times[-1] if times else 0.0:.6f} s")
        print(f"final residuals         : {np.asarray(final_residuals)}")
        print(f"final planar speeds     : {np.linalg.norm(final_snapshot.g_dot[:, :2], axis=1)}")

        if args.plot is not None and times:
            save_plot(Path(args.plot), times, residual_history, speed_history)
            print(f"plot saved              : {Path(args.plot).resolve()}")

        if use_gui and not args.manual:
            print("GUI run finished. Close the window or press Ctrl+C to exit.")
            try:
                while pb.isConnected():
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass

        # Divergence is an experimental outcome rather than a program error.
        # The textual result remains machine-readable without making ordinary
        # command-line runs look as if Python/PyBullet itself failed.
        return 0
    finally:
        if pb.isConnected():
            pb.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated distributed formation test with primitive PyBullet boxes."
    )
    parser.add_argument("--gui", action="store_true", help="Open the PyBullet GUI")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Open GUI and control both leaders with the keyboard until the window closes",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep after each physics step (useful with --gui)",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--physics-timestep", type=float, default=1.0 / 240.0)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--perturbation", type=float, default=1.0e-5)
    parser.add_argument("--leader-vx", type=float, default=0.20)
    parser.add_argument("--leader-vy", type=float, default=0.0)
    parser.add_argument("--leader-omega", type=float, default=0.0)
    parser.add_argument("--manual-speed", type=float, default=0.5)
    parser.add_argument("--manual-angular-speed", type=float, default=0.5)
    parser.add_argument("--divergence-threshold", type=float, default=20.0)
    parser.add_argument("--enable-apf", action="store_true")
    parser.add_argument("--apf-influence", type=float, default=1.2)
    parser.add_argument("--apf-gain", type=float, default=0.10)
    parser.add_argument("--apf-max-speed", type=float, default=1.0)
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional output path for the residual/speed figure",
    )
    args = parser.parse_args()
    if args.physics_timestep <= 0.0:
        parser.error("--physics-timestep must be positive")
    if args.substeps < 1:
        parser.error("--substeps must be at least 1")
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
