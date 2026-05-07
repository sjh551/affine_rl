from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR = OUTPUTS_DIR / "models"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
TB_STAGE1_DIR = OUTPUTS_DIR / "tensorboard" / "stage1"
TB_STAGE2_DIR = OUTPUTS_DIR / "tensorboard" / "stage2"
FIGURES_DIR = OUTPUTS_DIR / "figures"
AUTOENCODER_DIR = OUTPUTS_DIR / "autoencoder"

DATASET_100K_PATH = DATA_DIR / "lidar_dataset_100k.npy"
PRETRAINED_ENCODER_PATH = AUTOENCODER_DIR / "pretrained_encoder.pth"

MODEL_STAGE1_NAME = "ppo2_stage1_nav"
MODEL_FINAL_NAME = "ppo2_final_leader-stage3-1100w" #"ppo2_final_leader2阶段-300wsteps" #"ppo2_final_leader-best"


def ensure_output_dirs() -> None:
    for folder in [
        OUTPUTS_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        TB_STAGE1_DIR,
        TB_STAGE2_DIR,
        FIGURES_DIR,
        AUTOENCODER_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def model_save_base(model_name: str) -> str:
    return str((MODELS_DIR / model_name).resolve())


def model_load_path(model_name_or_path: str) -> str:
    raw = Path(model_name_or_path)
    if raw.exists():
        return str(raw.resolve())

    in_models = MODELS_DIR / model_name_or_path
    if in_models.exists():
        return str(in_models.resolve())

    zip_candidate = in_models.with_suffix(".zip")
    if zip_candidate.exists():
        return str(zip_candidate.resolve())

    # 兼容老路径（根目录下的历史模型）
    legacy = PROJECT_ROOT / model_name_or_path
    if legacy.exists():
        return str(legacy.resolve())
    legacy_zip = legacy.with_suffix(".zip")
    if legacy_zip.exists():
        return str(legacy_zip.resolve())

    return model_name_or_path
