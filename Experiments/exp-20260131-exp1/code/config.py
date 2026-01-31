from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path
    exp_dir: Path
    data_dir: Path
    artifacts_dir: Path
    tf_log_dir: Path
    cache_dir: Path

    window_size: int = 400
    resample_minutes: int = 5
    horizon_days: int = 7

    min_move_pct: float = 0.0005 * 2.5  # 0.0500% * 2.5
    atr_window: int = 14
    std_window: int = 20

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 2048
    num_epochs: int = 5
    learning_rate: float = 1e-3
    critic_weight: float = 0.5
    grad_accum_steps: int = 1
    parallel_prep: bool = True
    max_workers: int | None = None
    label_log_transform: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0

    seed: int = 42


def build_config() -> ExperimentConfig:
    code_dir = Path(__file__).resolve().parent
    exp_dir = code_dir.parent
    root_dir = exp_dir.parent.parent
    data_dir = root_dir / "Data"
    artifacts_dir = exp_dir / "artifacts"
    tf_log_dir = artifacts_dir / "tf_logs"
    cache_dir = artifacts_dir / "cache"
    return ExperimentConfig(
        root_dir=root_dir,
        exp_dir=exp_dir,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        tf_log_dir=tf_log_dir,
        cache_dir=cache_dir,
    )
