import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def draw_environment_footprints(ax, obstacle_footprints, wall_footprints=None):
    obstacle_label_used = False
    for aabb_min, aabb_max in obstacle_footprints:
        x0, y0 = aabb_min[0], aabb_min[1]
        w = aabb_max[0] - aabb_min[0]
        h = aabb_max[1] - aabb_min[1]
        rect = Rectangle(
            (x0, y0),
            w,
            h,
            facecolor="dimgray",
            edgecolor="black",
            alpha=0.55,
            label="障碍物" if not obstacle_label_used else None,
            zorder=1,
        )
        ax.add_patch(rect)
        obstacle_label_used = True

    if wall_footprints:
        for aabb_min, aabb_max in wall_footprints:
            x0, y0 = aabb_min[0], aabb_min[1]
            w = aabb_max[0] - aabb_min[0]
            h = aabb_max[1] - aabb_min[1]
            rect = Rectangle(
                (x0, y0),
                w,
                h,
                facecolor="lightgray",
                edgecolor="gray",
                alpha=0.35,
                zorder=0,
            )
            ax.add_patch(rect)


def _build_formation_edges(env, num_robots):
    if hasattr(env, "edges") and env.edges:
        edges = [(u, v) for (u, v) in env.edges if u < num_robots and v < num_robots]
    else:
        edges = [(i, i + 1) for i in range(num_robots - 1)]

    if num_robots >= 2:
        edges.append((1, 0))
    return edges


def plot_trajectory_and_formation(
    env,
    all_trajectories,
    obstacle_footprints,
    wall_footprints=None,
    output_path=None,
    show=True,
    title="Trajectory",
):
    num_robots = len(all_trajectories)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    plt.scatter(env.target_pos[0], env.target_pos[1], c="red", marker="*", s=300, label="目标点", zorder=5)

    colors = ["r", "b", "g", "c", "m", "y", "k", "#7f7f7f", "#CE782C"]
    labels = ["领导者 1", "领导者 2", "跟随者 1", "跟随者 2", "跟随者 3", "跟随者 4", "跟随者 5", "跟随者 6", "跟随者 7"]

    for i in range(num_robots):
        xs, ys = all_trajectories.get(i, ([], []))
        if not xs or not ys:
            continue

        color = colors[i % len(colors)]
        plt.plot(xs, ys, color=color, label=labels[i] if i < len(labels) else f"Robot {i}", linewidth=2)
        plt.scatter(xs[0], ys[0], c="white", edgecolors=color, s=50)
        plt.scatter(xs[-1], ys[-1], c=color, s=100)

    draw_environment_footprints(ax, obstacle_footprints, wall_footprints)

    valid_lengths = [len(all_trajectories[i][0]) for i in range(num_robots) if len(all_trajectories[i][0]) > 0]
    if valid_lengths:
        total_steps = min(valid_lengths)
        if total_steps > 0:
            snap_indices = sorted(set([0, total_steps // 3, (2 * total_steps) // 3, total_steps - 1]))
            formation_edges = _build_formation_edges(env, num_robots)
            snap_styles = ["--", "-.", ":", "-"]
            snap_alphas = [0.35, 0.5, 0.65, 0.9]

            for k, step_idx in enumerate(snap_indices):
                style = snap_styles[k % len(snap_styles)]
                alpha = snap_alphas[k % len(snap_alphas)]
                for u, v in formation_edges:
                    if step_idx >= len(all_trajectories[u][0]) or step_idx >= len(all_trajectories[v][0]):
                        continue
                    x_u = all_trajectories[u][0][step_idx]
                    y_u = all_trajectories[u][1][step_idx]
                    x_v = all_trajectories[v][0][step_idx]
                    y_v = all_trajectories[v][1][step_idx]
                    plt.plot(
                        [x_u, x_v],
                        [y_u, y_v],
                        color="black",
                        linestyle=style,
                        linewidth=2.0,
                        alpha=alpha,
                        zorder=4,
                    )

    wall_len = getattr(env, "wall_length", 10.0)
    wall_wid = getattr(env, "wall_width", 5.0)
    plt.plot([-wall_len, wall_len, wall_len, -wall_len, -wall_len], [wall_wid, wall_wid, -wall_wid, -wall_wid, wall_wid], "k--", linewidth=2)

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    ax.tick_params(axis="both", direction="in", length=6, width=1.5)
    plt.legend(loc="upper right", fontsize=24, framealpha=0.9)
    plt.xlabel("X 位置 (m)", fontsize=24)
    plt.ylabel("Y 位置 (m)", fontsize=24)
    # plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
