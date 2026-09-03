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


def announce_corpus(cfg) -> None:
    """
    Say which corpus this sweep is about to run on, before it runs.

    `data_dirs` defaults to a single dataset. A sweep launched with every other
    override passed explicitly and this one left alone runs on 48 identities while its
    author believes it is running on 343 - which happened, and invalidated a pilot
    whose whole premise was a property of the pooled corpus. The only tell was a user
    count buried in the loader's output.
    """
    dirs = list(getattr(cfg, "data_dirs", None) or [])
    print("=" * 78)
    print(f"CORPUS: {len(dirs)} dataset director{'y' if len(dirs) == 1 else 'ies'}")
    for directory in dirs:
        print(f"  {Path(directory).parent.name}")
    if len(dirs) == 1:
        print("  NOTE: one dataset. normalize=per_dataset and within_dataset_negatives")
        print("        are no-ops here, and this is the config default - pass data_dirs")
        print("        explicitly if you meant the pooled corpus.")
    print("=" * 78)


def build_folds(cfg, folds: int, seed: int) -> list[list[str]]:
    """
    Partition every user across all data_dirs into `folds` disjoint held-out groups,
    stratified by dataset.

    Stratification matters once several datasets are pooled. The corpus is very
    uneven - 100 / 99 / 48 / 35 / 22 / 21 / 18 users - and the datasets differ in
    difficulty: ViewGauss is 10Hz native and half its frames are duplicates at
    sample_rate=20, NJIT is room-scale walking with one session per user. Partitioning
    the pooled user list at random lets fold composition vary binomially, so a fold
    heavy in one dataset is measuring something different from its neighbours. That
    shows up as inflated fold variance, which is the one thing this project cannot
    afford: the fold spread is already what decides whether a difference is real.

    Assigning each dataset's users round-robin across folds keeps every fold's
    composition proportional to within one user per dataset. The starting offset is
    randomised per dataset so the remainder does not always land in fold 0.

    The configured `exclude_users` is ignored: cross-validation defines its own
    held-out groups, and honouring both would silently shrink the training set.
    """
    # Folds must partition whatever the run actually trains on. With max_users set,
    # that is the subsample, not the full corpus.
    from dataset import select_user_subset

    subset = select_user_subset(_plain(cfg.data_dirs), getattr(cfg, "max_users", None),
                                getattr(cfg, "seed", 0))
    subset = set(subset) if subset else None

    by_dataset: dict[str, list[str]] = {}
    total_users = 0
    for directory in _plain(cfg.data_dirs) or []:
        users = [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if os.path.isdir(os.path.join(directory, name))
            and (subset is None or os.path.join(directory, name) in subset)
        ]
        if users:
            by_dataset[directory] = users
            total_users += len(users)

    if folds < 2 or total_users < folds:
        raise ValueError(f"sweep.folds={folds} needs at least that many users; found {total_users}.")

    rng = np.random.default_rng(int(seed))
    groups: list[list[str]] = [[] for _ in range(folds)]
    for directory in sorted(by_dataset):
        users = by_dataset[directory]
        order = rng.permutation(len(users))
        offset = int(rng.integers(folds))
        for position, index in enumerate(order):
            groups[(position + offset) % folds].append(users[int(index)])

    return [sorted(group) for group in groups]


def describe_fold_composition(fold_users: list[list[str]]) -> str:
    """Users per dataset per fold, so an unbalanced split is visible not assumed."""
    datasets = sorted({Path(user).parent.parent.name for group in fold_users for user in group})
    lines = [f"Fold composition ({len(fold_users)} folds, "
             f"{sum(len(g) for g in fold_users)} users):"]
    for index, group in enumerate(fold_users):
        counts = {name: 0 for name in datasets}
        for user in group:
            counts[Path(user).parent.parent.name] += 1
        detail = "  ".join(f"{name[:22]}={counts[name]}" for name in datasets)
        lines.append(f"  fold {index}: {len(group):>4} users   {detail}")
    return "\n".join(lines)


#: Keys that describe HOW a sweep is run rather than WHAT it measures. Changing one
#: should reuse existing results, not invalidate them.
_IDENTITY_IGNORED_TOP = {"hydra", "sweep_id", "_dataset_tag", "save_path", "model_path",
                         "graph_path", "graph", "num_workers", "mode"}
_IDENTITY_IGNORED_SWEEP = {"artifact_root", "resume", "dry_run"}


def _portable(value):
    """
    Reduce absolute paths to their last three components.

    Without this the same experiment gets a different identity on every machine,
    because data_dirs are absolutised at startup against different repo roots.
    """
    if isinstance(value, str) and ("/" in value or "\\" in value):
        parts = Path(value).parts
        return "/".join(parts[-3:]) if len(parts) >= 3 else value
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    return value


def config_identity(cfg, configurations: list[dict]) -> str:
    """
    Digest of everything that defines what this sweep measures.

    This exists because the previous identity was the extractor names plus the grid
    overrides and nothing else. Every top-level key - max_users, objective, normalize,
    sample_time, data_dirs, val_user_fraction - was invisible, so two sweeps differing
    only in one of them shared an artifact root and a state file. The second would
    find the first's records, skip every run as already complete, and print the first
    sweep's numbers under the second's banner. That is a silent wrong answer rather
    than a crash, and it is worse still when the stale numbers happen to agree with
    the hypothesis being tested.
    """
    try:
        payload = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        payload = {key: value for key, value in vars(cfg).items()} if hasattr(cfg, "__dict__") else dict(cfg)

    payload = {key: value for key, value in payload.items() if key not in _IDENTITY_IGNORED_TOP}
    sweep_section = payload.get("sweep") or {}
    if isinstance(sweep_section, dict):
        payload["sweep"] = {key: value for key, value in sweep_section.items()
                            if key not in _IDENTITY_IGNORED_SWEEP}
    payload["__configurations__"] = [c["id"] for c in configurations]

    canonical = json.dumps(_portable(payload), sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


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


#: Ranking metric, best first.
#:   selected_test_auc  AUC at the validation-chosen epoch. Threshold-free and
#:                      insensitive to pair balance, which is why it leads: accuracy
#:                      has broken twice on this data, once through selection
#:                      inflation and once through a 69/31 pair set where a constant
#:                      predictor outscored every real configuration.
#:   selected_test_acc  accuracy at that same epoch. Meaningful, but only on a set
#:                      whose balance is what was requested.
#:   best_test_acc      max over epochs of the reported set. Inflated by ~+0.02 by
#:                      construction; the last resort, and flagged as such.
_RANKING_METRICS = ("selected_test_auc", "selected_test_acc", "best_test_acc")


def _best_accuracy(result) -> float | None:
    if not isinstance(result, dict):
        return None
    if result.get("mode") == "boosted":
        summaries = result.get("round_summaries") or []
        return max((float(s["best_test_acc"]) for s in summaries), default=None)
    for key in _RANKING_METRICS:
        value = result.get(key)
        if value is not None:
            return float(value)
    return None


def ranking_metric_name(result) -> str:
    """Which metric a result was ranked on, so the report can say so."""
    if isinstance(result, dict):
        for key in _RANKING_METRICS:
            if result.get(key) is not None:
                return key
    return "unknown"


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
                "metric": next((r.get("metric") for r in fold_records if r.get("metric")), None),
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
    metric = next((r.get("metric") for r in completed if r.get("metric")), None)
    if metric:
        # Both selected_* metrics are chosen on validation users; only the best_* ones
        # are a max over evaluations of the set they report.
        inflated = not str(metric).startswith("selected_")
        note = "  <- inflated; set val_user_fraction" if inflated else ""
        print(f"ranked on: {metric}{note}")
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

    announce_corpus(cfg)

    folds = _sweep_setting(cfg, "folds")
    folds = int(folds) if folds else None
    fold_seed = _sweep_setting(cfg, "seed") or getattr(cfg, "seed", 0)
    fold_users = build_folds(cfg, folds, fold_seed) if folds else None
    if fold_users:
        print(describe_fold_composition(fold_users))
    sweep_id = config_identity(cfg, configurations)

    artifact_root = Path(str(_sweep_setting(cfg, "artifact_root", "sweeps"))) / sweep_id
    state_path = artifact_root / "sweep_state.json"
    resume = bool(_sweep_setting(cfg, "resume", True))
    state = _load_state(state_path) if resume else {}

    # Second line of defence. The identity above should make a collision impossible,
    # but a state file written by an older version of this code carries an identity
    # computed a different way. Reusing it would silently skip every run and report
    # someone else's numbers, so refuse rather than trust the directory name.
    stored_identity = state.get("config_identity")
    stored_code = state.get("code_identity")
    current_code = results_log.code_identity()

    if state and stored_identity is not None and stored_identity != sweep_id:
        print(f"WARNING: state at {state_path} was written for configuration "
              f"{stored_identity}, not {sweep_id}. Ignoring it and starting fresh.")
        state = {}
    elif state and stored_identity is None:
        print(f"WARNING: state at {state_path} predates configuration identities and "
              "cannot be verified. Ignoring it and starting fresh.")
        state = {}
    elif state and stored_code != current_code:
        # The config is identical but the code is not. Reusing these records would
        # report pre-change numbers as though they came from the current code - which
        # is how three launches were lost in one day, each time silently confirming
        # what had been measured before the fix.
        print(f"WARNING: state at {state_path} was written by code {stored_code or 'unknown'}, "
              f"not {current_code}. The configuration matches but the implementation "
              "changed, so those results are not this experiment's. Starting fresh.")
        state = {}

    completed = state.get("records", {}) if resume else {}

    plan = f"{len(configurations)} configuration(s)"
    if fold_users:
        plan += f" x {len(fold_users)} folds = {len(configurations) * len(fold_users)} runs"
    print(f"\n=== Sweep {sweep_id}: {plan} ===")
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
            # Which users this fold held out. A fold number alone does not identify
            # them, so without this every later evaluation of the fold's checkpoint
            # silently depends on build_folds producing the same partition forever.
            "held_out_users": list(held_out or []),
            "extractor": configuration["extractor"],
            "overrides": configuration["overrides"],
            "description": description,
        }

        try:
            result = train_fn(run_cfg)
            record.update({
                "status": "ok",
                "best_test_acc": _best_accuracy(result),
                "metric": ranking_metric_name(result),
                "checkpoint": str(run_cfg.save_path),
            })
            results_log.append_run(run_cfg, result, dataset_tag=str(getattr(cfg, "_dataset_tag", "") or "sweep"))
        except Exception as exc:
            # One bad configuration must not cost the whole sweep.
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        records[key] = record
        _write_state(state_path, {"sweep_id": sweep_id, "config_identity": sweep_id,
                            "code_identity": results_log.code_identity(), "records": records})

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
