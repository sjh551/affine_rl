from POMDP_env import FormationPyBulletEnv
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


POMDP_STAGE1_MODEL_NAME = MODEL_STAGE1_NAME
POMDP_STAGE2_MODEL_NAME = "ppo2_stage2_obstacles_no_followers"
POMDP_STAGE3_MODEL_NAME = MODEL_FINAL_NAME
POMDP_FINAL_MODEL_NAME = POMDP_STAGE3_MODEL_NAME

POMDP_STAGE1_TB_DIR = TB_STAGE1_DIR / "pomdp"
POMDP_STAGE2_TB_DIR = TB_STAGE2_DIR / "pomdp"
POMDP_STAGE3_TB_DIR = TB_STAGE2_DIR.parent / "stage3" / "pomdp"
POMDP_CHECKPOINT_DIR = CHECKPOINTS_DIR / "pomdp"

STAGE_CONFIGS = build_stage_configs(
    prefix="pomdp",
    stage1_model=POMDP_STAGE1_MODEL_NAME,
    stage2_model=POMDP_STAGE2_MODEL_NAME,
    stage3_model=POMDP_STAGE3_MODEL_NAME,
    stage1_tensorboard_dir=POMDP_STAGE1_TB_DIR,
    stage2_tensorboard_dir=POMDP_STAGE2_TB_DIR,
    stage3_tensorboard_dir=POMDP_STAGE3_TB_DIR,
)


def main():
    run_training(
        env_class=FormationPyBulletEnv,
        stage_configs=STAGE_CONFIGS,
        expected_kinematics_dim=22,
        checkpoint_dir=POMDP_CHECKPOINT_DIR,
        description="Train PPO with partial follower-state observations.",
    )


if __name__ == "__main__":
    main()
