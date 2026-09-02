"""
Sweep mode: train many configurations in one command and rank them.

The intended workflow is that feature extractors are written independently (including
by a model), declare the knobs worth varying in ``search_space()``, and are then
compared here on identical data, splits and seeds.

Design decisions worth knowing:

- **One process, not one per configuration.** Sampled windows are cached, so each
  extra configuration pays about a second of loading rather than a full CSV parse.
- **A failing configuration never aborts the sweep.** Generated extractors will
  sometimes blow up on a particular combination; that run is recorded with its error
  and the sweep continues.
- **Every run is appended to results/runs.csv** exactly like a normal run, tagged
  with a ``sweep_id``, so sweep and non-sweep results stay comparable in one table.
- **Resume is on by default** and keyed by a digest of the configuration, so an
  interrupted sweep re-runs only what it has not finished.

Axes are namespaced. ``extractor_params.<name>`` varies an extractor hyperparameter;
any other key varies a top-level config value:

    sweep:
      grid:
        lr: [0.001, 0.0003]
        extractor_params.lstm_hidden: [32, 64, 128]

``grid: auto`` uses each extractor's own declared ``search_space()``.

Cross-validation
----------------
``sweep.folds: K`` runs every configuration against K disjoint held-out user groups
and ranks by the mean. This is not optional rigour on this data: measured on the
default dataset, swapping which 5 users are held out moves a training-free position
probe from 0.631 to 0.746 - a 0.114 spread, against a binomial error bar of +/-0.019
on 2560 pairs. The effective sample size is the number of held-out *users*, not the
number of pairs, so a single fixed split cannot separate configurations that differ
by a few points.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

import feature_extractor as fe
import results_log


EXTRACTOR_PREFIX = "extractor_params."


def _plain(value):
    """OmegaConf containers -> plain Python, so they can be hashed and serialised."""
    if isinstance(value, (DictConfig, type(OmegaConf.create([])))):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _sweep_setting(cfg, key: str, default=None):
    sweep = getattr(cfg, "sweep", None)
    if sweep is None:
        return default
    value = getattr(sweep, key, default)
    return default if value is None else value


def resolve_extractors(cfg) -> list[str]:
    """Which extractors this sweep covers."""
    selected = _sweep_setting(cfg, "extractors", "config")
    if isinstance(selected, str):
        if selected == "all":
            return fe.available()
        if selected == "config":
            return [str(getattr(cfg, "extractor", "paper_gnn_bilstm"))]
        return [selected]
    return [str(name) for name in _plain(selected)]


def resolve_axes(cfg, extractor: str) -> dict[str, list]:
    """
    Axes for one extractor, as {namespaced_key: [values]}.

    ``auto`` defers to the extractor's declared search space, which is the whole
    point of ``search_space()``: the extractor's author decides what is worth varying.
    """
    grid = _plain(_sweep_setting(cfg, "grid", "auto"))

    if grid in (None, "auto", {}):
        return {f"{EXTRACTOR_PREFIX}{key}": list(values)
                for key, values in fe.search_space(extractor).items()}

    axes = {}
    for key, values in grid.items():
        values = _plain(values)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"sweep.grid['{key}'] must be a non-empty list, got {values!r}.")
        axes[str(key)] = list(values)
    return axes


def build_folds(cfg, folds: int, seed: int) -> list[list[str]]:
    """
    Partition every user across all data_dirs into `folds` disjoint held-out groups.

    The configured `exclude_users` is ignored: cross-validation defines its own
    held-out groups, and honouring both would silently shrink the training set.
    """
    users = []
    for directory in _plain(cfg.data_dirs) or []:
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isdir(path):
                users.append(path)

    if folds < 2 or len(users) < folds:
        raise ValueError(f"sweep.folds={folds} needs at least that many users; found {len(users)}.")

    order = np.random.default_rng(int(seed)).permutation(len(users))
    return [[users[int(i)] for i in order[fold::folds]] for fold in range(folds)]


def _configuration_id(extractor: str, overrides: dict) -> str:
    payload = json.dumps({"extractor": extractor, "overrides": overrides}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def build_configurations(cfg) -> list[dict]:
    """
    Expand the sweep into a concrete, ordered list of configurations.

    Each entry is {id, extractor, overrides}, where overrides are namespaced keys.
    """
    strategy = str(_sweep_setting(cfg, "strategy", "grid")).lower()
    if strategy not in {"grid", "random"}:
        raise ValueError(f"sweep.strategy must be 'grid' or 'random', got {strategy!r}.")

    max_runs = _sweep_setting(cfg, "max_runs")
    max_runs = int(max_runs) if max_runs else None

    configurations = []
    for extractor in resolve_extractors(cfg):
        axes = resolve_axes(cfg, extractor)
        if not axes:
            # No declared search space: still worth running once, at defaults.
            configurations.append({
                "id": _configuration_id(extractor, {}),
                "extractor": extractor,
                "overrides": {},
            })
            continue

        keys = list(axes)
        for combination in itertools.product(*(axes[key] for key in keys)):
            overrides = dict(zip(keys, combination))
            configurations.append({
                "id": _configuration_id(extractor, overrides),
                "extractor": extractor,
                "overrides": overrides,
            })

    if strategy == "random" and max_runs is not None and max_runs < len(configurations):
        # Sample without replacement so a capped sweep still covers every axis,
        # rather than varying only the fastest-moving one as a truncated grid would.
        seed = _sweep_setting(cfg, "seed") or getattr(cfg, "seed", 0)
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(configurations), size=max_runs, replace=False)
        configurations = [configurations[int(index)] for index in sorted(chosen)]
    elif max_runs is not None:
        configurations = configurations[:max_runs]

    return configurations


def describe_configuration(configuration: dict) -> str:
    overrides = configuration["overrides"]
    if not overrides:
        return f"{configuration['extractor']} (defaults)"
    settings = ", ".join(
        f"{key[len(EXTRACTOR_PREFIX):] if key.startswith(EXTRACTOR_PREFIX) else key}={value}"
        for key, value in sorted(overrides.items())
    )
    return f"{configuration['extractor']} [{settings}]"


def apply_configuration(base_cfg, configuration: dict, artifact_root: Path, sweep_id: str,
                        fold: int | None = None, held_out_users: list[str] | None = None):
    """Build the per-run config for one configuration (optionally for one CV fold)."""
    run_cfg = deepcopy(base_cfg)
    OmegaConf.set_struct(run_cfg, False)

    run_cfg.mode = "train"
    if held_out_users is not None:
        run_cfg.exclude_users = list(held_out_users)
        run_cfg.swap_data = False
        run_cfg.test_on_excluded = True
        run_cfg.fold = fold
    run_cfg.extractor = configuration["extractor"]
    run_cfg.sweep_id = sweep_id

    extractor_params = dict(_plain(getattr(base_cfg, "extractor_params", None)) or {})
    for key, value in configuration["overrides"].items():
        if key.startswith(EXTRACTOR_PREFIX):
            extractor_params[key[len(EXTRACTOR_PREFIX):]] = value
        else:
            OmegaConf.update(run_cfg, key, value, merge=False)
    run_cfg.extractor_params = extractor_params or None

    epochs = _sweep_setting(base_cfg, "epochs")
    if epochs:
        run_cfg.epochs = int(epochs)

    suffix = "" if fold is None else f"_fold{fold}"
    run_dir = artifact_root / "runs" / f"{configuration['extractor']}_{configuration['id']}{suffix}"
    run_cfg.save_path = str(run_dir / "best.pth")
    run_cfg.graph_path = str(run_dir / "history.png")
    # Boosted rounds inside a sweep must not share one artifact root.
    if getattr(run_cfg, "boosting", None) is not None:
        run_cfg.boosting.artifact_root = str(run_dir / "boosting")

    return run_cfg


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _best_accuracy(result) -> float | None:
    if not isinstance(result, dict):
        return None
    if result.get("mode") == "boosted":
        summaries = result.get("round_summaries") or []
        return max((float(s["best_test_acc"]) for s in summaries), default=None)
    value = result.get("best_test_acc")
    return float(value) if value is not None else None


def _aggregate_folds(configurations: list[dict], records: dict) -> list[dict]:
    """
    Collapse per-fold records into one row per configuration, as mean and spread.

    The spread is reported because it is large on this data: swapping which users are
    held out moves a training-free position probe by 0.114 accuracy. Two
    configurations whose means differ by less than the fold standard deviation have
    not been separated by the experiment.
    """
    aggregated = []
    for configuration in configurations:
        fold_records = [r for r in records.values() if r.get("id") == configuration["id"]]
        scores = [r["best_test_acc"] for r in fold_records
                  if r.get("status") == "ok" and r.get("best_test_acc") is not None]
        failures = [r for r in fold_records if r.get("status") == "failed"]

        entry = {
            "id": configuration["id"],
            "extractor": configuration["extractor"],
            "overrides": configuration["overrides"],
            "description": describe_configuration(configuration),
            "folds_completed": len(scores),
            "fold_scores": [round(score, 4) for score in scores],
        }
        if scores:
            entry.update({
                "status": "ok",
                "best_test_acc": float(np.mean(scores)),
                "fold_std": float(np.std(scores)),
                "checkpoint": fold_records[0].get("checkpoint", ""),
            })
        else:
            entry.update({
                "status": "failed",
                "error": failures[0].get("error", "no folds completed") if failures else "no folds completed",
            })
        aggregated.append(entry)
    return aggregated


def _print_ranking(records: list[dict]) -> None:
    completed = [r for r in records if r.get("status") == "ok" and r.get("best_test_acc") is not None]
    failed = [r for r in records if r.get("status") == "failed"]

    print("\n" + "=" * 78)
    print(f"SWEEP RESULTS  ({len(completed)} completed, {len(failed)} failed)")
    print("=" * 78)

    if completed:
        cross_validated = any("fold_std" in record for record in completed)
        if cross_validated:
            print(f"{'rank':>4}  {'mean acc':>9}  {'sd':>7}  {'folds':>6}  configuration")
        else:
            print(f"{'rank':>4}  {'best acc':>9}  configuration")
        print("-" * 78)
        for rank, record in enumerate(sorted(completed, key=lambda r: -r["best_test_acc"]), start=1):
            if cross_validated:
                print(f"{rank:>4}  {record['best_test_acc']:>8.2%}  {record.get('fold_std', 0.0):>7.3f}  "
                      f"{record.get('folds_completed', 0):>6}  {record['description']}")
            else:
                print(f"{rank:>4}  {record['best_test_acc']:>8.2%}  {record['description']}")

        if cross_validated and len(completed) > 1:
            ranked = sorted((r["best_test_acc"] for r in completed), reverse=True)
            spread = max(r.get("fold_std", 0.0) for r in completed)
            if (ranked[0] - ranked[1]) < spread:
                print(f"\nNOTE: the top two configurations differ by {ranked[0] - ranked[1]:.3f}, less than "
                      f"the fold spread ({spread:.3f}). This experiment has not separated them.")

    for record in failed:
        print(f"  FAILED  {record['description']}: {record.get('error', '')}")
    print("=" * 78)


def run_sweep(cfg, train_fn=None) -> dict:
    """
    Run every configuration and return a ranked summary.

    Args:
        cfg: The full Hydra config, including the ``sweep`` block.
        train_fn: Training entry point, injected for testing. Defaults to train.train.
    """
    if train_fn is None:
        from train import train as train_fn

    configurations = build_configurations(cfg)

    folds = _sweep_setting(cfg, "folds")
    folds = int(folds) if folds else None
    fold_seed = _sweep_setting(cfg, "seed") or getattr(cfg, "seed", 0)
    fold_users = build_folds(cfg, folds, fold_seed) if folds else None
    sweep_id = hashlib.sha1(
        json.dumps([c["id"] for c in configurations], sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]

    artifact_root = Path(str(_sweep_setting(cfg, "artifact_root", "sweeps"))) / sweep_id
    state_path = artifact_root / "sweep_state.json"
    resume = bool(_sweep_setting(cfg, "resume", True))
    state = _load_state(state_path) if resume else {}
    completed = state.get("records", {}) if resume else {}

    print(f"\n=== Sweep {sweep_id}: {len(configurations)} configuration(s) ===")
    for configuration in configurations:
        marker = "done" if configuration["id"] in completed else "todo"
        print(f"  [{marker}] {describe_configuration(configuration)}")
    print(f"Artifacts: {artifact_root}")

    if _sweep_setting(cfg, "dry_run", False):
        print("\nsweep.dry_run=true - nothing was trained.")
        return {"mode": "sweep", "sweep_id": sweep_id, "dry_run": True,
                "configurations": configurations, "records": []}

    # Each unit of work is one (configuration, fold) pair. Without folds there is
    # exactly one unit per configuration and the resume key is unchanged.
    units = []
    for configuration in configurations:
        if fold_users is None:
            units.append((configuration, None, None, configuration["id"]))
        else:
            for fold_index, held_out in enumerate(fold_users):
                units.append((configuration, fold_index, held_out, f"{configuration['id']}_f{fold_index}"))

    records = dict(completed)
    for index, (configuration, fold_index, held_out, key) in enumerate(units, start=1):
        description = describe_configuration(configuration)
        label = description if fold_index is None else f"{description}  [fold {fold_index + 1}/{len(fold_users)}]"

        if key in records and records[key].get("status") == "ok":
            print(f"\n--- [{index}/{len(units)}] skipping (already complete): {label}")
            continue

        print(f"\n--- [{index}/{len(units)}] {label}")
        run_cfg = apply_configuration(cfg, configuration, artifact_root, sweep_id,
                                      fold=fold_index, held_out_users=held_out)
        record = {
            "id": configuration["id"],
            "key": key,
            "fold": fold_index,
            "extractor": configuration["extractor"],
            "overrides": configuration["overrides"],
            "description": description,
        }

        try:
            result = train_fn(run_cfg)
            record.update({
                "status": "ok",
                "best_test_acc": _best_accuracy(result),
                "checkpoint": str(run_cfg.save_path),
            })
            results_log.append_run(run_cfg, result, dataset_tag=str(getattr(cfg, "_dataset_tag", "") or "sweep"))
        except Exception as exc:
            # One bad configuration must not cost the whole sweep.
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        records[key] = record
        _write_state(state_path, {"sweep_id": sweep_id, "records": records})

    if fold_users is None:
        ordered = [records[c["id"]] for c in configurations if c["id"] in records]
    else:
        ordered = _aggregate_folds(configurations, records)
    _print_ranking(ordered)
    _write_summary_csv(artifact_root / "summary.csv", ordered)

    best = max(
        (r for r in ordered if r.get("status") == "ok" and r.get("best_test_acc") is not None),
        key=lambda r: r["best_test_acc"],
        default=None,
    )
    if best:
        print(f"Best: {best['description']} at {best['best_test_acc']:.2%}")
        print(f"Checkpoint: {best.get('checkpoint')}")

    return {
        "mode": "sweep",
        "sweep_id": sweep_id,
        "records": ordered,
        "best": best,
        "summary_path": str(artifact_root / "summary.csv"),
    }


def _write_summary_csv(path: Path, records: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "best_test_acc", "fold_std", "folds_completed", "fold_scores",
                         "extractor", "overrides", "status", "checkpoint", "error"])
        ranked = sorted(
            records,
            key=lambda r: (r.get("best_test_acc") is None, -(r.get("best_test_acc") or 0)),
        )
        for rank, record in enumerate(ranked, start=1):
            writer.writerow([
                rank,
                record.get("best_test_acc", ""),
                record.get("fold_std", ""),
                record.get("folds_completed", ""),
                ";".join(str(v) for v in record.get("fold_scores", [])),
                record.get("extractor", ""),
                ";".join(f"{key}={value}" for key, value in sorted(record.get("overrides", {}).items())),
                record.get("status", ""),
                record.get("checkpoint", ""),
                record.get("error", ""),
            ])
    print(f"Sweep summary written to {path}")
