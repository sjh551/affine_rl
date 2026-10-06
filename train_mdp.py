from MDP_env import FormationPyBulletEnv
from project_paths import (
    CHECKPOINTS_DIR,
    MODEL_FINAL_NAME,
    MODEL_STAGE1_NAME,
    TB_STAGE1_DIR,
    TB_STAGE2_DIR,
)
from training_common import (
    CustomCombinedExtractor,
    TensorboardRewardCallback,
    build_stage_configs,
    policy_kwargs,
    ppo_training_kwargs,
    run_training,
)


MDP_STAGE1_MODEL_NAME = f"mdp_{MODEL_STAGE1_NAME}"
MDP_STAGE2_MODEL_NAME = "mdp_ppo2_stage2_obstacles_no_followers"
MDP_STAGE3_MODEL_NAME = f"mdp_{MODEL_FINAL_NAME}"
MDP_FINAL_MODEL_NAME = MDP_STAGE3_MODEL_NAME

MDP_STAGE1_TB_DIR = TB_STAGE1_DIR / "mdp_full_state"
MDP_STAGE2_TB_DIR = TB_STAGE2_DIR / "mdp_full_state"
MDP_STAGE3_TB_DIR = TB_STAGE2_DIR.parent / "stage3" / "mdp_full_state"
MDP_CHECKPOINT_DIR = CHECKPOINTS_DIR / "mdp_full_state"

STAGE_CONFIGS = build_stage_configs(
    prefix="mdp",
    stage1_model=MDP_STAGE1_MODEL_NAME,
    stage2_model=MDP_STAGE2_MODEL_NAME,
    stage3_model=MDP_STAGE3_MODEL_NAME,
    stage1_tensorboard_dir=MDP_STAGE1_TB_DIR,
    stage2_tensorboard_dir=MDP_STAGE2_TB_DIR,
    stage3_tensorboard_dir=MDP_STAGE3_TB_DIR,
)


def main():
    run_training(
        env_class=FormationPyBulletEnv,
        stage_configs=STAGE_CONFIGS,
        expected_kinematics_dim=40,
        checkpoint_dir=MDP_CHECKPOINT_DIR,
        description="Train PPO with full follower-state observations.",
    )


if __name__ == "__main__":
    main()
