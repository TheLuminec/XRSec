"""
Coordinate-frame and position-semantics audit of processed datasets.

    .venv/Scripts/python audit_frames.py                 # every dataset with cached windows
    .venv/Scripts/python audit_frames.py who_is_alyx     # one or more dataset directory names

Run this on any newly converted dataset BEFORE it is trained or tested on. It reads the
5s@20Hz sample cache (so run a training or `mode=sweep sweep.dry_run=true` load first, or
point it at another resolution with --resolution), and reports per dataset:

  up axis      which world axis the head's local up vector (0,1,0) points along after
               rotation by the quaternion. Every Unity-family corpus is Y-up; Nymeria's
               raw frame is Z-up; NJIT's quaternion rotates about Z while its position is
               Y-up (a parser slip, see docs/GENERALISATION_PROPOSAL.md section 1).
  facing       mean world forward vector (0,0,1) rotated by the quaternion, and its
               concentration (1 = everyone faces the same way all the time). This is the
               content-referenced yaw offset, which differs per corpus (+Z, +X, -X, none)
               and which per-channel standardisation cannot remove.
  |pos|        mean norm of the position vector per frame. Exactly 1.000 with sd 0 means
               the dataset stores a unit viewing-direction vector in HmdPosition, not a
               position: PanoSaliency, Panonut360, Head_and_Gaze V1 and 360_em do this.
  position     mean, between-user sd of the per-user mean, and within-window sd, per axis.
  dup frames   fraction of consecutive frames that are bit-identical - the native-rate
               artefact (ViewGauss 10Hz native is ~50% at 20Hz).
  |q|          mean quaternion norm; anything far from 1.0000 is a conversion fault.

Everything is computed from a subsample of users and windows, so it runs in seconds on a
CPU and needs no checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))

import sample_cache  # noqa: E402

PROCESSED = ROOT / "processed_datasets"


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N,3) by unit quaternions q (N,4) in x,y,z,w order."""
    u, w = q[:, :3], q[:, 3:4]
    c = np.cross(u, v)
    return v + 2.0 * (w * c + np.cross(u, c))


def _cache_entry(dataset: str, user: str, resolution: str):
    """Newest cache file for one user at one resolution, trying older signatures too."""
    for suffix in (f"{resolution}_full-nearest-s{resolution[0]}", f"{resolution}_full-nearest", f"{resolution}_full"):
        pattern = str(sample_cache.cache_dir() / f"{sample_cache._slug(dataset)}__{sample_cache._slug(user)}__{suffix}__*.pt")
        hits = sorted(glob.glob(pattern), key=os.path.getmtime)
        if hits:
            return hits[-1]
    return None


def audit(dataset: str, resolution: str, max_users: int, windows_per_user: int, seed: int) -> dict | None:
    rng = np.random.default_rng(seed)
    users = sorted(d for d in os.listdir(PROCESSED / dataset / "users") if (PROCESSED / dataset / "users" / d).is_dir())
    if len(users) > max_users:
        users = [users[i] for i in np.sort(rng.choice(len(users), max_users, replace=False))]

    user_means, ups, forwards, duplicates, within, norms, position_norms = [], [], [], [], [], [], []
    for user in users:
        path = _cache_entry(dataset, user, resolution)
        if path is None:
            continue
        windows = torch.load(path, map_location="cpu", weights_only=True)["samples"].numpy()  # (n, C, T)
        if windows.shape[0] == 0:
            continue
        keep = np.sort(rng.choice(windows.shape[0], min(windows_per_user, windows.shape[0]), replace=False))
        windows = windows[keep]
        has_quaternion = windows.shape[1] >= 7
        position = np.transpose(windows[:, -3:, :], (0, 2, 1)).reshape(-1, 3)
        position_norms.append(np.linalg.norm(position, axis=1))
        user_means.append(position.mean(axis=0))
        within.append(windows[:, -3:, :].std(axis=2).mean(axis=0))
        frames = np.transpose(windows, (0, 2, 1))
        duplicates.append(np.mean(np.all(frames[:, 1:, :] == frames[:, :-1, :], axis=2)))
        if has_quaternion:
            q = np.transpose(windows[:, :4, :], (0, 2, 1)).reshape(-1, 4)
            norms.append(np.linalg.norm(q, axis=1).mean())
            ups.append(_rotate(q, np.tile([0.0, 1.0, 0.0], (q.shape[0], 1))).mean(axis=0))
            forwards.append(_rotate(q, np.tile([0.0, 0.0, 1.0], (q.shape[0], 1))).mean(axis=0))

    if not user_means:
        return None
    means = np.array(user_means)
    all_norms = np.concatenate(position_norms)
    result = {
        "dataset": dataset, "users": len(means),
        "pos_norm_mean": float(all_norms.mean()), "pos_norm_sd": float(all_norms.std()),
        "mean_pos": means.mean(axis=0), "between_user_sd": means.std(axis=0),
        "within_window_sd": np.array(within).mean(axis=0),
        "dup_frames": float(np.mean(duplicates)),
    }
    if ups:
        up = np.array(ups).mean(axis=0)
        forward = np.array(forwards)
        axis = int(np.argmax(np.abs(up)))
        # A real head keeps its up vector near one world axis. If the mean cancels to a
        # short vector the quaternion is rotating about the wrong axis (NJIT: yaw about Z
        # with a Y-up position), and naming an up axis from it would be a guess.
        up_axis = "xyz"[axis] + ("+" if up[axis] > 0 else "-") if np.linalg.norm(up) >= 0.5 else "??"
        result.update({
            "head_up": up, "up_axis": up_axis,
            "facing_mean": forward.mean(axis=0),
            "facing_concentration": float(np.linalg.norm(forward.mean(axis=0))),
            "qnorm": float(np.mean(norms)),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help="dataset directory names under processed_datasets/ (default: all)")
    parser.add_argument("--resolution", default="5s20hz", help="cache resolution to read, e.g. 5s20hz (default)")
    parser.add_argument("--max-users", type=int, default=60)
    parser.add_argument("--windows-per-user", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = args.datasets or sorted(d for d in os.listdir(PROCESSED) if (PROCESSED / d / "users").is_dir())
    rows = [r for r in (audit(n, args.resolution, args.max_users, args.windows_per_user, args.seed) for n in names) if r]
    if not rows:
        print("No cached windows found. Load the datasets once (any training run) so the cache is populated.")
        return 1

    f3 = lambda v: "[" + " ".join(f"{x:+.3f}" for x in v) + "]"  # noqa: E731
    print(f"{'dataset':<30} {'n':>3}  {'up':<4} {'head-up (world xyz)':<24} {'facing':>7} {'|pos|':>13}  "
          f"{'mean pos xyz':<26} {'between-user sd':<26} {'within-window sd':<26} {'dup':>6} {'|q|':>7}")
    for r in rows:
        if "head_up" in r:
            up, headup, facing, qn = r["up_axis"], f3(r["head_up"]), f"{r['facing_concentration']:.3f}", f"{r['qnorm']:.4f}"
        else:
            up, headup, facing, qn = "-", "(no quaternion)", "-", "-"
        norm = f"{r['pos_norm_mean']:.3f}+-{r['pos_norm_sd']:.3f}"
        flag = "  <- UNIT VECTOR, not a position" if r["pos_norm_sd"] < 1e-3 and abs(r["pos_norm_mean"] - 1.0) < 1e-3 else ""
        if r.get("up_axis") == "??":
            flag += "  <- head-up vector cancels: quaternion frame suspect"
        print(f"{r['dataset'][:30]:<30} {r['users']:>3}  {up:<4} {headup:<24} {facing:>7} {norm:>13}  "
              f"{f3(r['mean_pos']):<26} {f3(r['between_user_sd']):<26} {f3(r['within_window_sd']):<26} "
              f"{r['dup_frames']:>6.3f} {qn:>7}{flag}")
    print("\nfacing mean (world xyz), i.e. the corpus's yaw reference:")
    for r in rows:
        if "facing_mean" in r:
            print(f"  {r['dataset'][:30]:<30} {f3(r['facing_mean'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
