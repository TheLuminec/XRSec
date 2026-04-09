"""
Hydra entry point for XR biometric identification project.

Usage examples:
    python model/main.py mode=train
    python model/main.py mode=train data_dirs=[/abs/path/users_a,/abs/path/users_b]
    python model/main.py mode=test test_dirs=[/abs/path/users_eval]
"""

import re
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, ListConfig

from train import train
from eval import evaluate_model
from utils import plot_training_history

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower() or "dataset"


def _dataset_tag(paths) -> str:
    names = [Path(p).name for p in _as_list(paths)]
    if not names:
        return "none"
    if len(names) == 1:
        return _slug(names[0])
    return f"multi{len(names)}"


def _normalize_paths(cfg: DictConfig) -> None:
    """Convert configured filesystem paths to absolute paths for Hydra run dirs."""
    cfg.data_dirs = [to_absolute_path(p) for p in _as_list(cfg.data_dirs)]
    cfg.test_dirs = [to_absolute_path(p) for p in _as_list(cfg.test_dirs)]
    cfg.exclude_users = [to_absolute_path(p) for p in _as_list(cfg.exclude_users)]


def _artifact_stem(cfg: DictConfig) -> str:
    active_dirs = cfg.test_dirs if cfg.mode == "test" and cfg.test_dirs else cfg.data_dirs
    tag = _dataset_tag(active_dirs)
    return (
        f"{_slug(cfg.experiment_name)}_{tag}_"
        f"{cfg.sample_time}s_{cfg.sample_rate}hz_emb{cfg.embedding_dim}_{cfg.mode}"
    )


def _resolve_output_path(path_value: str, default_path: Path) -> str:
    if path_value == "auto":
        return str(default_path)

    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    return str(Path(get_original_cwd()) / path)


def _resolve_artifact_paths(cfg: DictConfig) -> None:
    stem = _artifact_stem(cfg)
    checkpoint_path = Path("checkpoints") / f"{stem}.pth"
    graph_path = Path("plots") / f"{stem}.png"

    cfg.save_path = _resolve_output_path(cfg.save_path, checkpoint_path)
    cfg.model_path = _resolve_output_path(cfg.model_path, checkpoint_path)
    cfg.graph_path = _resolve_output_path(cfg.graph_path, graph_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _normalize_paths(cfg)
    _resolve_artifact_paths(cfg)

    if cfg.mode == "train":
        print("=== Starting Training Mode ===")
        result = train(cfg)
        if cfg.graph and isinstance(result, dict) and "train_loss" in result and "test_loss" in result:
            print("Generating training graph...")
            plot_training_history(result, save_path=cfg.graph_path)
        elif cfg.graph and getattr(cfg, "boosting", None) and cfg.boosting.enabled:
            print("Skipping training graph generation for boosted mode.")

    elif cfg.mode == "test":
        print("=== Starting Testing Mode ===")
        evaluate_model(cfg)

    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
