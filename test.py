from evaluation_common import run_evaluation
from POMDP_env import FormationPyBulletEnv
from train import STAGE_CONFIGS


def main():
    run_evaluation(
        env_class=FormationPyBulletEnv,
        stage_configs=STAGE_CONFIGS,
        expected_kinematics_dim=22,
        observation_name="partial_state_pomdp",
        description="Evaluate PPO in the partial-state POMDP environment.",
    )


if __name__ == "__main__":
    main()
