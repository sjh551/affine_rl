from evaluation_common import run_evaluation
from MDP_env import FormationPyBulletEnv
from train_mdp import STAGE_CONFIGS


def main():
    run_evaluation(
        env_class=FormationPyBulletEnv,
        stage_configs=STAGE_CONFIGS,
        expected_kinematics_dim=36,
        observation_name="full_state_mdp",
        description="Evaluate PPO in the full-state MDP environment.",
    )


if __name__ == "__main__":
    main()
