import time
import importlib.util
from pathlib import Path

import numpy as np


TEST_ENV_PATH = Path(__file__).with_name("POMDP_env test.py")
TEST_ENV_SPEC = importlib.util.spec_from_file_location(
    "pomdp_env_single_integrator",
    TEST_ENV_PATH,
)
TEST_ENV_MODULE = importlib.util.module_from_spec(TEST_ENV_SPEC)
TEST_ENV_SPEC.loader.exec_module(TEST_ENV_MODULE)
FormationPyBulletEnv = TEST_ENV_MODULE.FormationPyBulletEnv


# ============================================================
# 运动领导者控制律最小测试
# ============================================================
# 两名领导者保持完全相同的恒定前向动作，直到某台机器人越过墙体边界。
# 关闭 APF 只是为了隔离公式（29）本身的速度反馈行为；墙体和越界检测
# 仍然保留。centralized distributed
LEADER_FORWARD_ACTION = 0.35
FOLLOWER_CONTROL_MODE = "distributed"
RENDER_MODE = True
PRINT_INTERVAL = 10
STEP_SLEEP_SECONDS = 0.03
# 本测试把积分步长缩小 4 倍，同时保持 0.1 s 的控制周期不变。
PHYSICS_SUBSTEPS_PER_CONTROL = 24
PHYSICS_TIMESTEP = 1.0 / 240.0
TEST_RANDOM_SEED = 0
SELF_EXCITATION_MARGIN = 0.20
SELF_EXCITATION_REQUIRED_STEPS = 5


def disable_follower_apf(env):
    """让跟随者控制中的 APF 恒为零，避免墙面斥力干扰本测试。"""

    def zero_repulsion(_pos_2d, safe_margin=1.0, num_rays=90):
        return safe_margin, np.zeros(2, dtype=float)

    env._get_obstacle_repulsion = zero_repulsion


def move_goal_beyond_right_wall(env):
    """
    将任务目标放到右侧墙外，使回合先因撞墙/越界结束，而不是因到达
    reset() 随机生成的目标点而提前 Success。
    """
    center = env._get_center_pos()[:2]
    target_x = env.wall_length + 50.0

    env.target_pos = np.array([target_x, center[1], 0.0], dtype=float)
    env.virtual_target_pos = np.array(
        [target_x + 10.0, center[1]],
        dtype=float,
    )
    env.L = env.target_pos[:2] - env.start_center[:2]
    env.L_norm = np.linalg.norm(env.L)
    env.last_center_dist = np.linalg.norm(env.target_pos[:2] - center)

    # 重新计算目标对齐坐标系，使正的 action[0]、action[2] 对应世界系
    # 正 x 方向。
    return env._get_obs(env._get_center_pos())


def get_planar_speeds(env):
    """读取两名领导者和两名跟随者的世界系平移速度模长。"""
    states = env._get_all_robot_states()
    leader_speeds = np.array([
        np.linalg.norm(states[i]["vel"][:2])
        for i in range(env.n_leaders)
    ])
    follower_speeds = np.array([
        np.linalg.norm(states[i]["vel"][:2])
        for i in range(env.n_leaders, env.num_robots)
    ])
    return leader_speeds, follower_speeds


def main():
    np.random.seed(TEST_RANDOM_SEED)
    env = FormationPyBulletEnv(
        render_mode=RENDER_MODE,
        enable_obstacles=False,
        follower_control_mode=FOLLOWER_CONTROL_MODE,
        physics_substeps_per_control=PHYSICS_SUBSTEPS_PER_CONTROL,
        physics_timestep=PHYSICS_TIMESTEP,
    )

    result = None
    try:
        _, _ = env.reset()
        disable_follower_apf(env)
        obs = move_goal_beyond_right_wall(env)

        # 两名领导者使用完全相同的恒定前向动作：
        # [Leader A forward, Leader A lateral,
        #  Leader B forward, Leader B lateral]
        action = np.array([
            LEADER_FORWARD_ACTION,
            0.0,
            LEADER_FORWARD_ACTION,
            0.0,
        ], dtype=np.float32)

        max_leader_speed = 0.0
        max_follower_speed = 0.0
        excitation_counter = 0
        excitation_detected = False

        print("=" * 72)
        print("运动领导者控制律最小测试")
        print(f"Follower mode       : {env.follower_control_mode}")
        print(f"Leader action       : {action.tolist()}")
        print(f"Leader target speed : {LEADER_FORWARD_ACTION * env.max_vel:.3f} m/s")
        print(
            "Control interval    : "
            f"{PHYSICS_SUBSTEPS_PER_CONTROL * PHYSICS_TIMESTEP:.6f} s "
            f"({PHYSICS_SUBSTEPS_PER_CONTROL} x {PHYSICS_TIMESTEP:.8f} s)"
        )
        print("APF                 : disabled for isolation")
        print("Stop condition      : Wall_Out / Crash_Team / Timeout")
        print("=" * 72)

        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            leader_speeds, follower_speeds = get_planar_speeds(env)

            current_leader_max = float(np.max(leader_speeds))
            current_follower_max = float(np.max(follower_speeds))
            max_leader_speed = max(max_leader_speed, current_leader_max)
            max_follower_speed = max(max_follower_speed, current_follower_max)

            # 两名领导者执行相同平移时，理想跟随者速度应与领导者接近。
            # 若跟随者连续多个周期显著快于领导者，则标记为疑似自激。
            if (
                current_follower_max
                > current_leader_max + SELF_EXCITATION_MARGIN
            ):
                excitation_counter += 1
            else:
                excitation_counter = 0

            if (
                not excitation_detected
                and excitation_counter >= SELF_EXCITATION_REQUIRED_STEPS
            ):
                excitation_detected = True
                print(
                    f"[SELF-EXCITATION SUSPECTED] step={env.current_step}, "
                    f"leader={leader_speeds.round(3).tolist()}, "
                    f"follower={follower_speeds.round(3).tolist()}"
                )

            if env.current_step % PRINT_INTERVAL == 0:
                print(
                    f"step={env.current_step:4d} | "
                    f"leader speed={leader_speeds.round(3).tolist()} | "
                    f"follower speed={follower_speeds.round(3).tolist()} | "
                    f"max follower={max_follower_speed:.3f}"
                )

            if terminated or truncated:
                print("=" * 72)
                print(f"Test finished        : {info.get('reason', 'Unknown')}")
                print(f"Total steps          : {env.current_step}")
                print(f"Max leader speed     : {max_leader_speed:.3f} m/s")
                print(f"Max follower speed   : {max_follower_speed:.3f} m/s")
                print(f"Self-excitation flag : {excitation_detected}")
                print("=" * 72)
                result = {
                    "physics_substeps_per_control": PHYSICS_SUBSTEPS_PER_CONTROL,
                    "control_interval_s": (
                        PHYSICS_SUBSTEPS_PER_CONTROL * PHYSICS_TIMESTEP
                    ),
                    "physics_timestep_s": PHYSICS_TIMESTEP,
                    "total_control_steps": env.current_step,
                    "simulated_time_s": (
                        env.current_step
                        * PHYSICS_SUBSTEPS_PER_CONTROL
                        * PHYSICS_TIMESTEP
                    ),
                    "reason": info.get("reason", "Unknown"),
                    "max_leader_speed": max_leader_speed,
                    "max_follower_speed": max_follower_speed,
                    "self_excitation": excitation_detected,
                }
                break

            time.sleep(STEP_SLEEP_SECONDS)
    finally:
        env.close()
    return result


if __name__ == "__main__":
    main()
