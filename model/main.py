"""
Hydra entry point for XR biometric identification project.

Usage examples:
    python model/main.py mode=train
    python model/main.py mode=train data_dirs=[/abs/path/users_a,/abs/path/users_b]
    python model/main.py mode=test test_dirs=[/abs/path/users_eval]
"""

import hashlib
import re
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

import results_log
from sweep import run_sweep
from train import train
from eval import evaluate_model
from utils import plot_boosted_training_history, plot_training_history

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

    # Hydra chdirs into a fresh run directory, so a relative sweep root would be
    # unreachable next time and resume would never find prior state.
    sweep = getattr(cfg, "sweep", None)
    if sweep is not None and getattr(sweep, "artifact_root", None):
        cfg.sweep.artifact_root = to_absolute_path(str(cfg.sweep.artifact_root))


def _artifact_stem(cfg: DictConfig) -> str:
    active_dirs = cfg.test_dirs if cfg.mode == "test" and cfg.test_dirs else cfg.data_dirs
    tag = _dataset_tag(active_dirs)
    # The extractor name is part of the stem so runs that differ only by extractor
    # write to distinct checkpoints and plots. Non-default hyperparameters add a short
    # digest, so a sweep does not produce many identically named artifacts.
    extractor = _slug(str(getattr(cfg, "extractor", "paper_gnn_bilstm")))
    params = getattr(cfg, "extractor_params", None)
    if params:
        digest = hashlib.sha1(
            ";".join(f"{key}={params[key]}" for key in sorted(params)).encode("utf-8")
        ).hexdigest()[:6]
        extractor = f"{extractor}-{digest}"

    return (
        f"{_slug(cfg.experiment_name)}_{tag}_{extractor}_"
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

    OmegaConf.set_struct(cfg, False)
    cfg._dataset_tag = _dataset_tag(cfg.data_dirs)
    OmegaConf.set_struct(cfg, True)

    if cfg.mode == "train":
        print("=== Starting Training Mode ===")
        result = train(cfg)
        if cfg.graph and isinstance(result, dict) and "train_loss" in result and "test_loss" in result:
            print("Generating training graph...")
            plot_training_history(result, save_path=cfg.graph_path)
        elif cfg.graph and getattr(cfg, "boosting", None) and cfg.boosting.enabled:
            print("Generating boosted training graphs...")
            plot_boosted_training_history(
                result.get("round_histories", []),
                save_path=cfg.graph_path,
            )

    elif cfg.mode == "test":
        print("=== Starting Testing Mode ===")
        result = evaluate_model(cfg)

    elif cfg.mode == "sweep":
        print("=== Starting Sweep Mode ===")
        # run_sweep logs each configuration to the results table itself, so the
        # single summary row that append_run would write here is skipped.
        run_sweep(cfg)
        return

    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    results_log.append_run(cfg, result, dataset_tag=_dataset_tag(
        cfg.test_dirs if cfg.mode == "test" and cfg.test_dirs else cfg.data_dirs
    ))


if __name__ == "__main__":
    main()
