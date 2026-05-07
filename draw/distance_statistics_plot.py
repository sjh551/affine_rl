import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


def _point_to_rect_distance(point, rect_min, rect_max):
    px, py = point
    x0, y0 = rect_min
    x1, y1 = rect_max
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return float(np.hypot(dx, dy))


def _topology_robot_distances(step_positions_dict, topology_edges):
    distances = []
    for u, v in topology_edges:
        if u in step_positions_dict and v in step_positions_dict:
            distances.append(float(np.linalg.norm(step_positions_dict[u] - step_positions_dict[v])))
    return distances


def compute_distance_metrics(all_trajectories, obstacle_footprints, topology_edges=None):
    robot_indices = sorted(all_trajectories.keys())
    valid_lengths = [len(all_trajectories[i][0]) for i in robot_indices if len(all_trajectories[i][0]) > 0]
    if not valid_lengths:
        return {
            "steps": np.array([]),
            "min_robot_distance": np.array([]),
            "max_robot_distance": np.array([]),
            "min_robot_obstacle_distance": np.array([]),
        }

    total_steps = min(valid_lengths)
    min_robot_distance = []
    max_robot_distance = []
    min_robot_obstacle_distance = []

    # 仅保留当前机器人集合内的拓扑边，并去重 (u,v) 与 (v,u)
    edge_set = set()
    if topology_edges:
        for u, v in topology_edges:
            if u in robot_indices and v in robot_indices and u != v:
                edge_set.add(tuple(sorted((u, v))))
    filtered_edges = sorted(edge_set)

    for step_idx in range(total_steps):
        step_positions = []
        step_positions_dict = {}
        for rid in robot_indices:
            xs, ys = all_trajectories[rid]
            if step_idx < len(xs):
                pos = np.array([xs[step_idx], ys[step_idx]], dtype=np.float32)
                step_positions.append(pos)
                step_positions_dict[rid] = pos

        if len(step_positions) < 2:
            continue

        topo_dists = _topology_robot_distances(step_positions_dict, filtered_edges)
        if not topo_dists:
            # 若未提供有效拓扑边，则本步智能体-智能体距离记为 NaN
            min_robot_distance.append(np.nan)
            max_robot_distance.append(np.nan)
        else:
            min_robot_distance.append(min(topo_dists))
            max_robot_distance.append(max(topo_dists))

        if obstacle_footprints:
            obstacle_dists = []
            for pos in step_positions:
                for rect_min, rect_max in obstacle_footprints:
                    obstacle_dists.append(_point_to_rect_distance(pos, rect_min[:2], rect_max[:2]))
            min_robot_obstacle_distance.append(min(obstacle_dists) if obstacle_dists else np.nan)
        else:
            min_robot_obstacle_distance.append(np.nan)

    steps = np.arange(len(min_robot_distance))
    return {
        "steps": steps,
        "min_robot_distance": np.asarray(min_robot_distance, dtype=np.float32),
        "max_robot_distance": np.asarray(max_robot_distance, dtype=np.float32),
        "min_robot_obstacle_distance": np.asarray(min_robot_obstacle_distance, dtype=np.float32),
    }


def plot_distance_statistics(
    all_trajectories,
    obstacle_footprints,
    topology_edges=None,
    output_path=None,
    show=True,
    title="Distance Statistics During Formation",
):
    metrics = compute_distance_metrics(all_trajectories, obstacle_footprints, topology_edges=topology_edges)
    steps = metrics["steps"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, metrics["min_robot_distance"], color="#d62728", lw=2.5, label="最小 智能体-智能体 距离")
    ax.plot(steps, metrics["max_robot_distance"], color="#1f77b4", lw=2.5, label="最大 智能体-智能体 距离")
    ax.plot(steps, metrics["min_robot_obstacle_distance"], color="#2ca02c", lw=2.5, label="最小 智能体-障碍物 距离")

    # ax.set_title(title, fontsize=16)
    ax.set_xlabel("步数", fontsize=18)
    ax.set_ylabel("距离 (m)", fontsize=18, )
    ax.tick_params(axis="both", which="major", labelsize=18, direction="in", length=6, width=1.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=18, loc="upper right", framealpha=0.9)
    ax.set_xlim(0, max(len(steps) - 1, 1))
    ax.set_ylim(0, 10)  # y轴从0开始，自动调整上限

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return metrics
