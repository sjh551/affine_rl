"""
DWA (Dynamic Window Approach) 动态窗口法 完整演示
=================================================
算法核心步骤：
  1. 计算动态窗口：根据当前速度 (v,w) 和运动学约束，确定下一步
     可行的速度范围 [v_min, v_max] × [w_min, w_max]
  2. 采样候选轨迹：在窗口内均匀采样 (v,w)，模拟前向轨迹
  3. 评价函数（三项）：
       • heading_cost  — 轨迹末端朝向与目标方向的角度差
       • speed_cost    — 鼓励高速运动（max_speed - v）
       • obstacle_cost — 最近障碍物距离的倒数（越近代价越大）
  4. 选出综合代价最小的 (v*,w*)，执行后循环迭代

依赖：numpy, matplotlib（标准科学计算包）
运行：python DWA.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ========================
# DWA 类
# ========================
class DWA:
    def __init__(self, config):
        self.cfg = config

    def motion(self, state, control):
        """
        计算经过时间dt后, what is the next state
        state: 位置、角度、速度
        """
        dt = self.cfg.dt
        s = state.copy()
        # control 現在是 [vx, vy]
        s[0] += control[0] * dt  # x 變化
        s[1] += control[1] * dt  # y 變化
        s[3] = control[0]        # 更新 vx 狀態
        s[4] = control[1]        # 更新 vy 狀態
        # s[2] (yaw) 在全向平移模型中可以保持不變
        return s
    
    def calc_min_obstacle_dist(self, traj, obstacles):
        dx = traj[:, 0:1] - obstacles[:, 0]
        dy = traj[:, 1:2] - obstacles[:, 1]
        r = np.hypot(dx, dy)
        return np.min(r)  # 返回轨迹上所有点到障碍物的【最小距离】

    def calc_dynamic_window(self, state, obstacles):
        """
        计算速度的上下限
        """
        vx, vy = state[3], state[4]   # 两轴的速度
        Vs = [self.cfg.min_speed, self.cfg.max_speed,
              self.cfg.min_speed, self.cfg.max_speed]
        
        Vd = [vx - self.cfg.max_accel * self.cfg.dt,
              vx + self.cfg.max_accel * self.cfg.dt,
              vy - self.cfg.max_accel * self.cfg.dt,
              vy + self.cfg.max_accel * self.cfg.dt]
        
        # 先预测一条以当前速度直行的轨迹，计算离障碍物多近
        traj = self.predict_trajectory(state, vx, vy) # 这里多计算了一次，懒得改了
        dist = self.calc_min_obstacle_dist(traj, obstacles)
        v_max_safe = np.sqrt(2 * dist * self.cfg.max_accel)
        w_max_safe = np.sqrt(2 * dist * self.cfg.max_accel)


        dw = [
            max(Vs[0], Vd[0], -v_max_safe),   # 最小vx
            min(Vs[1], Vd[1],  v_max_safe),   # 最大vx
            max(Vs[2], Vd[2], -w_max_safe),   # 最小vy
            min(Vs[3], Vd[3],  w_max_safe)    # 最大vy
        ]

        return dw

    def predict_trajectory(self, state, vx, vy):
        """
        state:  current state
        vx:     linear velocity in x
        vy:     linear velocity in y
        return: the trajectory of current state within the predicted time period
        """
        s = state.copy()
        traj = [s.copy()]   # 记录以当前的速度前进预设时间后的轨迹
        t = 0.0
        while t <= self.cfg.predict_time:
            s = self.motion(s, [vx, vy])
            traj.append(s.copy())
            t += self.cfg.dt
        return np.array(traj)

    def calc_obstacle_cost(self, traj, obstacles, config_id=None):
        """
        计算当前轨迹与障碍物之间的代价
        若最小距离小于机器人半径则代价无限大, 否则为最小距离的倒数
        """
        min_r = float('inf')

        # --- 1. 计算到点状障碍物的距离 (所有配置通用) ---
        if obstacles.size > 0:
            for point in traj:
                dx = point[0] - obstacles[:, 0]
                dy = point[1] - obstacles[:, 1]
                r_points = np.hypot(dx, dy)
                min_r = min(min_r, np.min(r_points))

        # --- 2. 计算到四周边界墙的距离 (所有配置通用) ---
        dist_boundary = np.min([
            self.cfg.wall_length - np.abs(traj[:, 0]),
            self.cfg.wall_width - np.abs(traj[:, 1])
        ])
        min_r = min(min_r, dist_boundary)

        # --- 3. 只有当配置 ID 为 1 时，才计算中间「带门的墙」 ---
        if config_id == 1:
            y_limits = [-self.cfg.wall_width + 0.5]
            for dy in self.cfg.doors_y:
                y_limits.append(dy - self.cfg.gap_half_width)
                y_limits.append(dy + self.cfg.gap_half_width)
            y_limits.append(self.cfg.wall_width - 0.5)
            
            for point in traj:
                px, py = point[0], point[1]
                for i in range(0, len(y_limits), 2):
                    y_start, y_end = y_limits[i], y_limits[i+1]
                    dx = px - self.cfg.wall_x
                    if py < y_start:
                        dist_to_seg = np.hypot(dx, py - y_start)
                    elif py > y_end:
                        dist_to_seg = np.hypot(dx, py - y_end)
                    else:
                        dist_to_seg = abs(dx) - self.cfg.wall_thickness / 2.0
                    min_r = min(min_r, dist_to_seg)

        # --- 4. 最终判定 ---
        if min_r <= self.cfg.robot_radius:
            return float('inf')
            
        return 1.0 / min_r

    def calc_to_goal_cost(self, traj, goal):
        dist = np.hypot(goal[0] - traj[-1, 0], goal[1] - traj[-1, 1])
        return dist

    def plan(self, state, goal, obstacles, config_id=None):
        """
        state: 当前机器人的位姿和速度
        goal:  目标位置
        obs:   positions of obstacles
        config_id: 配置 ID

        """
        dw = self.calc_dynamic_window(state, obstacles)
        best_u = [0.0, 0.0]
        min_cost = float('inf')
        all_traj = []

        for v in np.arange(dw[0], dw[1] + self.cfg.v_resolution, self.cfg.v_resolution):
            for w in np.arange(dw[2], dw[3] + self.cfg.yaw_rate_resolution, self.cfg.yaw_rate_resolution):
                traj = self.predict_trajectory(state, v, w)
                oc = self.calc_obstacle_cost(traj, obstacles, config_id)
                if oc == float('inf'):
                    continue

                vx = traj[-1, 3]
                vy = traj[-1, 4]
                current_v = np.hypot(vx, vy)

                gc = self.cfg.to_goal_cost_gain * self.calc_to_goal_cost(traj, goal)
                sc = self.cfg.speed_cost_gain * (self.cfg.max_speed - current_v)  # 跑得越快 lower cost
                cost = gc + sc + self.cfg.obstacle_cost_gain * oc

                all_traj.append((traj, cost))
                if cost < min_cost:
                    min_cost = cost
                    best_u = [v, w]

        return best_u, all_traj

# ========================
# 配置
# ========================
class Config:
    max_speed           = 1.0   # 最大速度
    min_speed           = -1.0  # 最小速度
    max_yaw_rate        = 60.0 * np.pi / 180.0  # 最大角速度（每秒转60度）
    max_accel           = 0.2   # 最大加速度（车不能瞬间猛加速）
    max_delta_yaw_rate  = 80.0 * np.pi / 180.0  # 最大角速度变化（转向快慢）
    v_resolution        = 0.05  # 速度采样步长（越小轨迹越多，越慢）
    yaw_rate_resolution = 2.0 * np.pi / 180.0   # 角度采样步长（2度一步）
    dt                  = 0.1   
    predict_time        = 3.0   # 向前看 3 秒的轨迹
    robot_radius        = 0.6   # 机器人半径
    to_goal_cost_gain   = 0.4  # 朝向目标权重（越大越拼命朝终点）
    speed_cost_gain     = 0.6   # 速度权重（越大越想快跑）
    obstacle_cost_gain  = 2.0   # 避障权重（越大越怕撞）
    wall_length         = 20.0  # 對應環境中的 self.wall_length
    wall_width          = 10.0  # 對應環境中的 self.wall_width

    wall_x = 0.0
    wall_thickness = 1.0
    doors_y = [-5.0, 0.0, 5.0]
    gap_half_width = 0.8

cfg = Config()


# ========================
# 仿真（调用 DWA 类）
# ========================
def simulate():
    state = np.array([0.0, 0.0, np.pi / 8.0, 0.0, 0.0]) # 位姿 + 速度
    goal  = np.array([10.0, 10.0])
    obs   = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 0.],
                      [5., 5.], [6., 3.], [6., 7.], [8., 1.],
                      [9., 6.], [8., 8.], [11., 5.]])

    dwa = DWA(cfg)  # 实例化 DWA

    history    = [state.copy()]
    controls   = []
    all_trajs  = []
    best_trajs = []
    reached    = False

    for _ in range(500):
        dist = np.hypot(state[0] - goal[0], state[1] - goal[1]) # 计算目标距离
        if dist <= cfg.robot_radius:
            reached = True
            break

        u, trajs = dwa.plan(state, goal, obs)   # 返回最优输入以及所有的可行轨迹
        state = dwa.motion(state, u)

        history.append(state.copy())
        controls.append(u)
        all_trajs.append(trajs)
        best_trajs.append(min(trajs, key=lambda t: t[1])[0] if trajs else None)

    return np.array(history), controls, all_trajs, best_trajs, goal, obs, reached, dwa

# ========================
# 可视化
# ========================
def run_demo():
    print("正在运行 DWA 仿真，请稍候...")
    history, controls, all_trajs, best_trajs, goal, obs, reached, dwa = simulate()
    total = len(history) - 1
    print(f"仿真完成！共 {total} 步，{'成功到达目标！' if reached else '未到达（增加 max_iter）'}")

    fig = plt.figure(figsize=(16, 9), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
    ax      = fig.add_subplot(gs[:, :2])
    ax_v    = fig.add_subplot(gs[0, 2])
    ax_w    = fig.add_subplot(gs[1, 2])
    ax_dist = fig.add_subplot(gs[2, 2])

    for a in [ax, ax_v, ax_w, ax_dist]:
        a.set_facecolor('#0f0f23')
        a.tick_params(colors='#aaaaaa', labelsize=8)
        for sp in a.spines.values():
            sp.set_color('#333355')

    ax.set_xlim(-1, 12); ax.set_ylim(-1, 12); ax.set_aspect('equal')
    ax.set_title('DWA Dynamic Window Approach  —  Path Planning Demo',
                 color='white', fontsize=13, pad=8)
    ax.set_xlabel('X (m)', color='#aaaaaa')
    ax.set_ylabel('Y (m)', color='#aaaaaa')

    for v in range(12):
        ax.axvline(v, color='#1a1a3e', lw=0.4, zorder=0)
        ax.axhline(v, color='#1a1a3e', lw=0.4, zorder=0)

    for o in obs:
        ax.add_patch(plt.Circle(o, cfg.robot_radius * 1.1,
                                color='#E24B4A', alpha=0.75, zorder=3))
        ax.plot(o[0], o[1], 'x', color='#ff6b6b', ms=6, zorder=4)

    ax.plot(history[:, 0], history[:, 1], color='#223355', lw=1.5, zorder=2)
    ax.plot(history[0, 0], history[0, 1], 's', color='#1D9E75', ms=12, zorder=8, label='Start')
    ax.plot(goal[0], goal[1], '*', color='#FAC775', ms=18, zorder=8, label='Goal')
    ax.add_patch(plt.Circle(goal, 0.4, color='#FAC775', fill=False, ls='--', alpha=0.5))

    trail_line,  = ax.plot([], [], color='#3B8BD4', lw=2.0, alpha=0.9, zorder=4, label='Robot path')
    best_line,   = ax.plot([], [], color='#1D9E75', lw=2.5, alpha=0.85, zorder=5,
                           label='Best predicted traj')
    cand_lines   = [ax.plot([], [], color='#0F6E56', lw=0.6, alpha=0.2, zorder=2)[0]
                    for _ in range(80)]
    robot_patch  = plt.Circle((0, 0), cfg.robot_radius, color='#3B8BD4', alpha=0.9, zorder=6)
    ax.add_patch(robot_patch)
    dir_line,    = ax.plot([], [], color='white', lw=2, zorder=7)
    dw_box       = patches.FancyBboxPatch(
        (0, 0), 0, 0, boxstyle="round,pad=0.05",
        lw=1, edgecolor='#AFA9EC', facecolor='#AFA9EC', alpha=0.12, zorder=1)
    ax.add_patch(dw_box)
    dw_lbl       = ax.text(0, 0, '', color='#AFA9EC', fontsize=7)
    info_box     = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                           color='white', fontsize=9, va='top',
                           bbox=dict(facecolor='#0f0f23', alpha=0.8,
                                     edgecolor='#333355', boxstyle='round'))
    ax.legend(loc='lower right', fontsize=8, facecolor='#1a1a2e',
              labelcolor='white', framealpha=0.7)

    def init_sub(axis, title, ylabel, color, extra=None):
        axis.set_title(title, color='white', fontsize=9, pad=4)
        axis.set_ylabel(ylabel, color='#aaaaaa', fontsize=8)
        axis.set_xlabel('Step', color='#aaaaaa', fontsize=8)
        axis.axhline(0, color='#333355', lw=0.8)
        if extra:
            extra(axis)
        ln,  = axis.plot([], [], color=color, lw=1.5)
        dot, = axis.plot([], [], 'o', color=color, ms=4)
        vl   = axis.axvline(0, color='white', lw=0.8, ls=':', alpha=0.5)
        return ln, dot, vl

    v_ln, v_dot, v_vl = init_sub(ax_v, 'Linear Velocity (m/s)', 'm/s', '#3B8BD4',
                                  extra=lambda a: a.axhline(cfg.max_speed, color='#E24B4A',
                                                            lw=0.8, ls='--', alpha=0.6))
    w_ln, w_dot, w_vl = init_sub(ax_w, 'Angular Velocity (deg/s)', 'deg/s', '#EF9F27')
    d_ln, d_dot, d_vl = init_sub(ax_dist, 'Distance to Goal (m)', 'm', '#1D9E75',
                                  extra=lambda a: a.axhline(cfg.robot_radius, color='#FAC775',
                                                            lw=0.8, ls='--', alpha=0.6))

    v_data    = [c[0] for c in controls]
    w_data    = [np.degrees(c[1]) for c in controls]
    dist_data = [np.hypot(history[i, 0] - goal[0], history[i, 1] - goal[1])
                 for i in range(len(history))]

    def update(frame):
        if frame >= total:
            return
        s = history[frame]
        x, y, yaw = s[0], s[1], s[2]

        robot_patch.center = (x, y)
        dir_line.set_data([x, x + cfg.robot_radius * np.cos(yaw)],
                          [y, y + cfg.robot_radius * np.sin(yaw)])
        trail_line.set_data(history[:frame + 1, 0], history[:frame + 1, 1])

        trajs = all_trajs[frame] if frame < len(all_trajs) else []
        for i, ln in enumerate(cand_lines):
            if i < len(trajs):
                ln.set_data(trajs[i][0][:, 0], trajs[i][0][:, 1])
            else:
                ln.set_data([], [])

        if frame < len(best_trajs) and best_trajs[frame] is not None:
            bt = best_trajs[frame]
            best_line.set_data(bt[:, 0], bt[:, 1])
        else:
            best_line.set_data([], [])

        dw = dwa.calc_dynamic_window(s, obs)

        vr = dw[1] - dw[0]; wr = dw[3] - dw[2]
        dw_box.set_bounds(x - vr, y - wr, vr * 2, wr * 2)
        dw_lbl.set_position((x - vr, y + wr + 0.1))
        dw_lbl.set_text(f'DW v=[{dw[0]:.2f},{dw[1]:.2f}]')

        vc = controls[frame][0] if frame < len(controls) else 0
        wc = np.degrees(controls[frame][1]) if frame < len(controls) else 0
        info_box.set_text(
            f'Step  : {frame}/{total}\n'
            f'v     : {vc:.2f} m/s\n'
            f'ω     : {wc:.1f} °/s\n'
            f'Dist  : {dist_data[frame]:.2f} m'
        )

        for ln, dot, vl, axis, data in [
            (v_ln, v_dot, v_vl, ax_v, v_data),
            (w_ln, w_dot, w_vl, ax_w, w_data),
            (d_ln, d_dot, d_vl, ax_dist, dist_data),
        ]:
            xs = list(range(frame + 1))
            ln.set_data(xs, data[:frame + 1])
            if xs:
                dot.set_data([xs[-1]], [data[frame]])
            vl.set_xdata([frame, frame])
            axis.set_xlim(0, max(frame + 1, 20))
            visible = data[:frame + 1]
            if visible:
                ymin, ymax = min(visible), max(visible)
                m = max(abs(ymax - ymin) * 0.2, 0.15)
                axis.set_ylim(ymin - m, ymax + m)

    plt.suptitle('DWA (Dynamic Window Approach)  Path Planning Demo',
                 color='white', fontsize=14, y=0.99)

    skip   = max(1, total // 150)
    frames = list(range(0, total, skip)) + [total - 1]
    anim   = FuncAnimation(fig, update, frames=frames,
                           interval=60, blit=False, repeat=False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return anim

if __name__ == '__main__':
    anim = run_demo()