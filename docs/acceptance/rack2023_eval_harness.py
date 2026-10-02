"""Evaluation harness for Rack et al. 2023 — their model, their protocol, our plumbing.

WHY THIS EXISTS. Their published numbers are not reachable from their published code:
there is no `test_step` anywhere (so the 27 test subjects are never scored), use-time is
hardcoded to [5, 10, 15] minutes in `similarity_module.py:27`, and enrolment is a single
fixed `session_1_embeddings[::150]` at line 99. Figure 3 needs the test split, an
enrolment axis and a 1-minute use-time. None is present.

WHAT IS THEIRS AND WHAT IS MINE — this distinction is the whole point.
  THEIRS: the trained model; `SimilarityDatamodule` (including the 27/9/27 split); the
          `WindowDataset`; and `MotionAccuracyCalculator`, which computes every metric.
  MINE:   the loop that drives them, and the two knobs they hardcoded — which split to
          score, and how much enrolment to use.

The metric is **re-driven, not reimplemented**. Reimplementing it would put two
independent implementations of the same arithmetic in play, and a disagreement could then
come from the metric rather than from the thing being measured — the exact defect this
project's cross-machine gate exists to avoid. So `MotionAccuracyCalculator` is imported
and called, never copied.

ENROLMENT, and this is an INFERENCE that is labelled as one. Their `[::150]` takes every
150th window of session 1; `frame_step_size = original_fps // fps = 1`, so windows step one
frame and every 150th window is one every 10 seconds across the WHOLE session — i.e. all
available enrolment data, subsampled for kNN tractability. Limiting enrolment to N minutes
is therefore: keep session-1 windows whose `frame_id` falls in the first N minutes, then
subsample at the SAME 150-frame spacing. Duration varies; gallery density is held fixed.
Varying both at once would confound enrolment duration with gallery density.

Supporting evidence for the inference: `return_frame_ids = True` is set on the **test**
dataset and not on validation (`similarity_datamodule.py`), which is what an
enrolment-limited analysis needs and nothing else in the shipped code uses.

THE GATE. Run with `--split validation --enrol-minutes 0 --gate`: that reproduces their
hardcoded configuration exactly, and the 5-minute figure must match
`sequence_top_1_accuracy_5_mins/validation/mean` as their own module computes it. Until
that passes, no number from this harness means anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "Versatile-XR-User-Identification" / "machine_learning"
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

from src.custom_metrics.accuracy_calculator import MotionAccuracyCalculator
from src.lightning_module.similarity_module import SimilarityModule

FPS = 15
THEIR_GALLERY_SPACING_FRAMES = 150   # similarity_module.py:99 -> [::150]


def build_datamodule(experiment: str, data_path: str | None, stats_path: str):
    """Instantiate THEIR datamodule through HYDRA'S OWN COMPOSITION, so the split is theirs.

    The first version did OmegaConf.load() on the experiment YAML alone and died on
    `${data_dir}`, which only the root config defines (2026-09-17, rc=1 before scoring).
    compose() is what run.py does; the split seed is the datamodule's own `seed` key.
    """
    from hydra import initialize_config_dir, compose
    data_path = data_path or f"{REPO}/data/15_fps-63_subjects-metric_learning_movement.hdf5"
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.1"):
        cfg = compose(config_name="config",
                      overrides=[f"+experiments/{experiment}",
                                 f"datamodule.data_path={data_path}",
                                 f"datamodule.data_stats_path={stats_path}"])
    print(f"datamodule seed {cfg.datamodule.seed}; split {dict(cfg.datamodule.split)}; stats {stats_path}")
    return hydra.utils.instantiate(cfg.datamodule, _recursive_=True)


@torch.no_grad()
def embed(module, loader, device) -> dict:
    """Forward every window once; keep embeddings, labels, session and frame ids."""
    module.eval().to(device)
    embs, ys, sess, frames = [], [], [], []
    for i, batch in enumerate(loader):
        h = module(batch["data"].to(device)).cpu()
        embs.append(h)
        # .clone(), not .cpu(): on a CPU tensor .cpu() returns the SAME object, so keeping it kept the
        # worker's whole shared-memory batch alive (~14 MB/batch). Measured 2026-10-02 on the seed-42
        # gate: 18.7 GB at 72 s and OOM at the 32 GB cap unpatched, against 3.0 GB flat with .clone()
        # (no_grad was tested separately and is not needed). Values are copied, not changed, so the
        # embeddings and every metric are unaffected; the gate's exact reproduction is the proof.
        ys.append(batch["targets"].clone())
        sess.append(batch["session_idx"].clone())
        frames.append(batch["frame_id"].clone() if "frame_id" in batch
                      else torch.full_like(batch["targets"], -1))
        if i % 200 == 0:
            print(f"    batch {i}: {sum(e.shape[0] for e in embs):,} windows", flush=True)
    return {"emb": torch.cat(embs), "y": torch.cat(ys),
            "session": torch.cat(sess), "frame": torch.cat(frames)}


def score(packed: dict, enrol_minutes: float, use_time_minutes: list[int]) -> dict:
    """Gallery from session 0, probe from session 1, scored by THEIR calculator."""
    emb, y, sess, frame = packed["emb"], packed["y"], packed["session"], packed["frame"]

    g_mask = sess == 0
    if enrol_minutes and enrol_minutes > 0:
        if int(frame.max()) < 0:
            raise SystemExit("enrolment limiting needs frame_id, which this split did not return")
        limit = int(round(enrol_minutes * 60 * FPS))
        g_mask = g_mask & (frame < limit)

    g_idx = torch.nonzero(g_mask).flatten()[::THEIR_GALLERY_SPACING_FRAMES]
    p_idx = torch.nonzero(sess == 1).flatten()

    gallery, gallery_y = emb[g_idx].contiguous(), y[g_idx].contiguous()
    probe, probe_y = emb[p_idx].contiguous(), y[p_idx].contiguous()

    # Their calculator asserts query labels are a dense 0..n-1 range.
    uniq = torch.unique(probe_y)
    remap = {int(v): i for i, v in enumerate(uniq.tolist())}
    probe_y = torch.tensor([remap[int(v)] for v in probe_y])
    keep = torch.tensor([int(v) in remap for v in gallery_y])
    gallery, gallery_y = gallery[keep], torch.tensor([remap[int(v)] for v in gallery_y[keep]])

    print(f"    gallery {gallery.shape[0]:,} windows / {len(torch.unique(gallery_y))} subjects"
          f"   probe {probe.shape[0]:,} / {len(uniq)} subjects", flush=True)

    evaluator = MotionAccuracyCalculator(
        sequence_lengths_minutes=list(use_time_minutes),
        sliding_window_step_size_seconds=1, k="max_bin_count", device=torch.device("cpu"))
    with torch.cuda.device(-1):
        acc = evaluator.get_accuracy(probe, gallery, probe_y, gallery_y,
                                     embeddings_come_from_same_source=False)
    return {"n_gallery_windows": int(gallery.shape[0]), "n_probe_windows": int(probe.shape[0]),
            "n_subjects": int(len(uniq)), "metrics": {k: float(v) for k, v in acc.items()}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--experiment", default="tvcg_2024_paper=winner_similarity_model")
    p.add_argument("--stats", required=True, help="the train_stats.json written by the run that made the checkpoint")
    p.add_argument("--data-path", default=None)
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--enrol-minutes", type=float, default=0, help="0 = all available (their [::150])")
    p.add_argument("--use-time", type=int, nargs="+", default=[1, 5, 10, 15])
    p.add_argument("--gate", type=float, default=None,
                   help="their recorded sequence_top_1_accuracy_5_mins/validation/mean to reproduce")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}; split {a.split}; enrol {'ALL' if not a.enrol_minutes else str(a.enrol_minutes)+' min'}")

    dm = build_datamodule(a.experiment, a.data_path, str(Path(a.stats).resolve()))
    stage = "test" if a.split == "test" else "validate"
    dm.setup(stage=stage)
    if a.split == "validation":
        # THEIR stage="validate" branch builds the dataset WITHOUT return_session_idx, so
        # validation_step (batch["session_idx"]) cannot run on it. During training the val
        # dataset comes from the fit branch, which sets exactly these two flags and enforces
        # the training stats - the same stats that run saved to --stats. Mirror the fit branch.
        dm.val_dataset.return_session_idx = True
        dm.val_dataset.return_take_id = True
    loader = dm.test_dataloader() if a.split == "test" else dm.val_dataloader()
    subject_ids = dm.test_subject_ids if a.split == "test" else dm.validation_subject_ids
    print(f"{a.split} subjects ({len(subject_ids)}): {sorted(map(str, subject_ids))}")

    module = SimilarityModule.load_from_checkpoint(a.checkpoint, map_location=device)
    packed = embed(module, loader, device)
    result = score(packed, a.enrol_minutes, a.use_time)
    result["split"] = a.split
    result["subject_ids"] = sorted(map(str, subject_ids))
    result["enrol_minutes"] = a.enrol_minutes or "all"
    result["checkpoint"] = a.checkpoint

    print("\n=== metrics ===")
    for k in sorted(result["metrics"]):
        print(f"  {k:45} {result['metrics'][k]!r}")

    rc = 0
    if a.gate is not None:
        key = "sequence_top_1_accuracy_5_mins"
        if key not in result["metrics"]:
            print(f"\nGATE CANNOT RUN: {key} not computed (use-time must include 5)")
            rc = 1
        else:
            got, want = result["metrics"][key], a.gate
            gap = abs(got - want)
            # Registered before the gate was first run: their own number is computed on the
            # same embeddings by the same calculator, so only batching/dtype order differs.
            verdict = "PASS" if gap < 1e-4 else ("PASS (loose, investigate)" if gap < 1e-2 else "FAIL")
            print(f"\n=== HARNESS GATE vs their own module ===")
            print(f"  theirs {want!r}\n  ours   {got!r}\n  gap    {gap:.3e}  -> {verdict}")
            result["gate"] = {"theirs": want, "ours": got, "gap": gap, "verdict": verdict}
            rc = 0 if verdict.startswith("PASS") else 1

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
