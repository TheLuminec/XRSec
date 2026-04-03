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
from hydra.utils import to_absolute_path
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


def _standardize_artifact_paths(cfg: DictConfig) -> None:
    """
    Standard naming convention for milestone A artifacts.

    - checkpoints/{experiment}_{dataset}_{sample}s_{rate}hz_emb{dim}_{mode}.pth
    - plots/{experiment}_{dataset}_{sample}s_{rate}hz_emb{dim}_{mode}.png
    """
    active_eval_dirs = cfg.test_dirs if cfg.test_dirs else cfg.data_dirs
    tag = _dataset_tag(active_eval_dirs if cfg.mode == "test" else cfg.data_dirs)
    stem = (
        f"{_slug(cfg.experiment_name)}_{tag}_"
        f"{cfg.sample_time}s_{cfg.sample_rate}hz_emb{cfg.embedding_dim}_{cfg.mode}"
    )

    if cfg.save_path == "auto":
        cfg.save_path = str(Path("checkpoints") / f"{stem}.pth")
    if cfg.model_path == "auto":
        cfg.model_path = str(Path("checkpoints") / f"{stem}.pth")
    if cfg.graph_path == "auto":
        cfg.graph_path = str(Path("plots") / f"{stem}.png")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _normalize_paths(cfg)
    _standardize_artifact_paths(cfg)

    if cfg.mode == "train":
        print("=== Starting Training Mode ===")
        history = train(cfg)
        if cfg.graph:
            print("Generating training graph...")
            plot_training_history(history, save_path=cfg.graph_path)

    elif cfg.mode == "test":
        print("=== Starting Testing Mode ===")
        evaluate_model(cfg)

    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
