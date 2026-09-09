"""Cross-machine gate: do two machines compute the SAME training-free numbers?

WHY. Two gaps in this project's record have no check at all. `code_identity` hashes
`model/*.py` only, so it cannot see a corpus difference (the converters sit at the repo
root) and cannot see a stack difference (no row records python, torch, cuda or device).
The AVALON/DESKTOP-C size manifest explicitly did NOT establish content equality, and
sha256 over 42.8 GB would cost ~2 h a side and still say nothing about the environment.

The mean-position lookup and the amplitude baseline need no model, no GPU and no
training. Given the same users, resolution and seed they are pure arithmetic over the
CSVs, so two machines MUST agree - and a disagreement can only come from the corpus, the
manifest RNG, or the environment. What matters is not whether the disks agree but whether
the arithmetic does.

THREE OUTPUTS, NOT ONE, SO A FAILURE LOCALISES INSTEAD OF MERELY FIRING:

    counts differ                       -> CORPUS content or user selection
    counts agree, manifest hash differs -> numpy Generator stream (NOT a finding:
                                           numpy freezes RandomState, not Generator, and
                                           generate_pair_manifest calls rng.choice)
    counts + hash agree, AUC differs    -> ENVIRONMENT / numerics, the gap we care about
    all three agree                     -> both open gaps closed in one run

It calls the pipeline's own `static_position_lookup`, `amplitude_lookup` and `roc_auc`
rather than reimplementing them: a gate that reimplements the thing it is gating tests
the reimplementation.

Registered tolerance (fixed BEFORE any number was seen):
    |delta| < 1e-12   pass, arithmetic agreement
    < 1e-9            pass with a note, float association order
    < 1e-6            investigate
    >= 1e-6           fail
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# model/ is not a package - its modules import each other flat (`import boost_train`),
# so the directory goes on the path and the imports are top-level, exactly as the
# pipeline does it. Importing `model.dataset` instead fails with "not a package".
sys.path.insert(0, str(REPO_ROOT / "model"))

import numpy as np
import torch

from dataset import build_sample_index, generate_pair_manifest
from metrics import roc_auc, static_position_lookup, amplitude_lookup


def stack_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "host": platform.node(),
    }


def sha256_tensor(*tensors) -> str:
    """Hash raw bytes of the manifest index tensors, in a fixed dtype and order.

    Cast to int64/float64 explicitly: a dtype difference between machines would otherwise
    change the hash for a reason that has nothing to do with which pairs were drawn.
    """
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().numpy()
        array = array.astype(np.int64 if np.issubdtype(array.dtype, np.integer) else np.float64)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run(args) -> dict:
    users_root = Path(args.data_dir)
    if not users_root.is_dir():
        raise SystemExit(f"not a directory: {users_root}")

    # sorted() traversal is the property the whole pipeline's reproducibility rests on,
    # so the user list is derived the same way rather than by a glob whose order is
    # filesystem-dependent.
    all_users = sorted(p.name for p in users_root.iterdir() if p.is_dir())
    keep = all_users[: args.n_users] if args.n_users else all_users
    keep_paths = [str(users_root / name) for name in keep]
    print(f"users: {len(keep)} of {len(all_users)} available (first {len(keep)} in sorted order)")
    print(f"first/last kept: {keep[0]} .. {keep[-1]}")

    index = build_sample_index(
        str(users_root),
        sample_time=args.sample_time,
        sample_rate=args.sample_rate,
        exclude_users=None,
        swap_data=False,
        channels=args.channels,
        resample=args.resample,
        window_stride=args.window_stride,
        center_position=False,
        encoding=args.encoding,
        keep_users=keep_paths,
    )
    n_windows = int(index.samples.shape[0])
    print(f"windows: {n_windows:,}")

    # Generate our own manifest ALWAYS - its hash is layer 2 of the diagnosis and we want
    # it even when we go on to score someone else's pairs.
    manifest = generate_pair_manifest(
        index,
        pairs_per_user=args.pairs_per_user,
        match_ratio=0.5,
        seed=args.seed,
        within_dataset_negatives=False,
        cross_session_positives=args.cross_session_positives,
    )
    x1, x2 = manifest["x1_indices"].view(-1), manifest["x2_indices"].view(-1)
    labels = manifest["labels"].view(-1)
    manifest_hash = sha256_tensor(x1, x2, labels)
    print(f"pairs: {x1.numel():,}   own manifest sha256: {manifest_hash}")

    if args.emit_manifest:
        args.emit_manifest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.emit_manifest,
                            x1=x1.cpu().numpy().astype(np.int64),
                            x2=x2.cpu().numpy().astype(np.int64),
                            labels=labels.cpu().numpy().astype(np.float64))
        print(f"emitted manifest -> {args.emit_manifest}")

    scored_manifest = "own"
    if args.use_manifest:
        # THE AMENDMENT THAT MAKES LAYER 3 REACHABLE. numpy freezes RandomState, not
        # Generator, so two machines on different numpy minors draw different pairs from
        # the same seed - DESKTOP-C 2.4.2 against Miami 2.5.3. Without this the gate stops
        # at "the streams diverged" and never tests the environment, which is the layer it
        # exists for. Scoring a SHARED manifest isolates the arithmetic from the draw.
        loaded = np.load(args.use_manifest)
        x1 = torch.from_numpy(loaded["x1"].astype(np.int64))
        x2 = torch.from_numpy(loaded["x2"].astype(np.int64))
        labels = torch.from_numpy(loaded["labels"].astype(np.float32))
        limit = int(index.samples.shape[0])
        # A shared manifest indexes into OUR index, so out-of-range means the corpora or
        # the user selection differ - catch it here rather than as a confusing IndexError.
        if int(x1.max()) >= limit or int(x2.max()) >= limit:
            raise SystemExit(
                f"shared manifest indexes window {max(int(x1.max()), int(x2.max()))} but this "
                f"index holds only {limit} - the corpora or user lists differ, so the pairs "
                f"are not comparable. That is a LAYER 1 (corpus) failure, not a stack one.")
        scored_manifest = sha256_tensor(x1, x2, labels)
        print(f"scoring SHARED manifest instead: {x1.numel():,} pairs, sha256 {scored_manifest}")

    positions = getattr(index, "window_mean_positions", None)
    amplitudes = getattr(index, "window_amplitudes", None)
    if positions is None or positions.numel() == 0:
        raise SystemExit("index carries no window_mean_positions - cannot gate")
    if amplitudes is None or amplitudes.numel() == 0:
        raise SystemExit("index carries no window_amplitudes - cannot gate")

    # CPU explicitly: the gate compares arithmetic, not two cuDNN versions.
    pos_scores = static_position_lookup(positions[x1].cpu(), positions[x2].cpu())
    amp_scores = amplitude_lookup(amplitudes[x1].cpu(), amplitudes[x2].cpu()).float()
    position_auc = float(roc_auc(pos_scores, labels.cpu()))
    amplitude_auc = float(roc_auc(amp_scores, labels.cpu()))

    report = {
        "spec": {
            "data_dir": str(users_root), "n_users": len(keep),
            "first_user": keep[0], "last_user": keep[-1],
            "sample_time": args.sample_time, "sample_rate": args.sample_rate,
            "window_stride": args.window_stride, "channels": args.channels,
            "encoding": args.encoding, "resample": args.resample,
            "seed": args.seed, "pairs_per_user": args.pairs_per_user,
            "cross_session_positives": args.cross_session_positives,
            "device_for_metrics": "cpu",
        },
        "counts": {"users": len(keep), "windows": n_windows, "pairs": int(x1.numel())},
        "manifest_sha256": manifest_hash,
        "scored_manifest_sha256": scored_manifest,
        "position_lookup_auc": position_auc,
        "amplitude_auc": amplitude_auc,
        "stack": stack_fingerprint(),
        "tolerance_registered_before_any_number": {
            "pass": 1e-12, "pass_with_note": 1e-9, "investigate": 1e-6},
    }
    print(f"\nposition_lookup_auc  {position_auc!r}")
    print(f"amplitude_auc        {amplitude_auc!r}")
    print(f"stack                {report['stack']}")
    return report


def compare(mine: dict, theirs: dict) -> int:
    """Localise a disagreement to corpus, RNG or environment - in that order, because
    each later layer is only meaningful if the earlier ones agree."""
    print("=== cross-machine comparison ===")
    ok = True
    if mine["counts"] != theirs["counts"]:
        print(f"COUNTS DIFFER -> CORPUS content or user selection")
        print(f"  mine   {mine['counts']}")
        print(f"  theirs {theirs['counts']}")
        return 1
    print(f"counts agree: {mine['counts']}")

    if mine["manifest_sha256"] != theirs["manifest_sha256"]:
        print("own-manifest hashes differ -> numpy Generator stream, NOT a corpus or stack finding")
        print(f"  mine   {mine['manifest_sha256'][:16]}...  numpy {mine['stack']['numpy']}")
        print(f"  theirs {theirs['manifest_sha256'][:16]}...  numpy {theirs['stack']['numpy']}")
    else:
        print(f"own-manifest hashes agree: {mine['manifest_sha256'][:16]}...")

    # Layer 3 is only meaningful if BOTH sides scored the same pairs. Comparing AUCs from
    # two different draws would report an environment difference that is really a draw
    # difference - the failure the shared manifest exists to prevent.
    mine_scored = mine.get("scored_manifest_sha256", "own")
    theirs_scored = theirs.get("scored_manifest_sha256", "own")
    if mine_scored == "own" or theirs_scored == "own" or mine_scored != theirs_scored:
        print("\nCANNOT TEST THE ENVIRONMENT LAYER: the two sides did not score the same pairs.")
        print(f"  mine scored   {mine_scored}")
        print(f"  theirs scored {theirs_scored}")
        print("  Re-run both with --use-manifest pointing at ONE shared .npz.")
        return 1
    print(f"both scored the SAME shared manifest: {mine_scored[:16]}...")

    for key in ("position_lookup_auc", "amplitude_auc"):
        delta = abs(mine[key] - theirs[key])
        if delta < 1e-12:
            verdict = "PASS (arithmetic agreement)"
        elif delta < 1e-9:
            verdict = "PASS with a note (float association order)"
        elif delta < 1e-6:
            verdict = "INVESTIGATE"; ok = False
        else:
            verdict = "FAIL"; ok = False
        print(f"{key}: |delta| = {delta:.3e}  -> {verdict}")
        print(f"   mine {mine[key]!r}   theirs {theirs[key]!r}")

    print(f"\nstacks: mine {mine['stack']}\n        theirs {theirs['stack']}")
    print("\nGATE PASSES: same corpus content and equivalent arithmetic on two stacks."
          if ok else "\nGATE FAILS - see the localisation above.")
    return 0 if ok else 1



def selftest() -> int:
    """Verify `compare` localises every failure class - needs no data, so DESKTOP-C can
    run it before it has anything to compare.

    A gate that only ever passes is decoration. The direction that matters most is the
    fifth: own-manifest hashes DIFFERING (numpy 2.4.2 against 2.5.3) while both sides
    scored a shared manifest must still reach layer 3 and pass. If that regressed, the
    gate would refuse to test the environment on exactly the machines it was built for.
    """
    import copy, io, contextlib

    base = {
        "counts": {"users": 12, "windows": 12508, "pairs": 384},
        "manifest_sha256": "a" * 64,
        "scored_manifest_sha256": "b" * 64,
        "position_lookup_auc": 0.5399576822916666,
        "amplitude_auc": 0.5455729166666666,
        "stack": {"python": "3.13.15", "numpy": "2.5.3", "torch": "2.14.0+cu130",
                  "cuda": "13.0", "device": "cpu", "host": "selftest"},
    }

    def case(label, mutate, want_rc, want_text):
        theirs = copy.deepcopy(base); mutate(theirs)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = compare(copy.deepcopy(base), theirs)
        ok = rc == want_rc and want_text in buf.getvalue()
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            print(f"    rc={rc} want={want_rc}; looked for {want_text!r}")
            print("    " + buf.getvalue().replace("\n", "\n    "))
        return ok

    results = [
        case("identical -> passes", lambda d: None, 0, "GATE PASSES"),
        case("counts differ -> CORPUS", lambda d: d["counts"].update(windows=999), 1, "CORPUS"),
        case("different shared pairs -> env layer unreachable",
             lambda d: d.__setitem__("scored_manifest_sha256", "deadbeef"), 1, "CANNOT TEST THE ENVIRONMENT"),
        case("one side never shared -> env layer unreachable",
             lambda d: d.__setitem__("scored_manifest_sha256", "own"), 1, "CANNOT TEST THE ENVIRONMENT"),
        case("own hashes differ, shared pairs -> STILL reaches layer 3",
             lambda d: d.__setitem__("manifest_sha256", "f" * 64), 0, "numpy Generator stream"),
        case("AUC +1e-13 -> pass",
             lambda d: d.__setitem__("position_lookup_auc", d["position_lookup_auc"] + 1e-13), 0, "PASS (arithmetic"),
        case("AUC +1e-10 -> pass with note",
             lambda d: d.__setitem__("position_lookup_auc", d["position_lookup_auc"] + 1e-10), 0, "association order"),
        case("AUC +1e-7 -> investigate",
             lambda d: d.__setitem__("position_lookup_auc", d["position_lookup_auc"] + 1e-7), 1, "INVESTIGATE"),
        case("AUC +1e-4 -> fail",
             lambda d: d.__setitem__("position_lookup_auc", d["position_lookup_auc"] + 1e-4), 1, "FAIL"),
    ]
    print()
    if all(results):
        print("SELFTEST PASSES: every failure class localises, and a numpy-stream")
        print("                 divergence does not block the environment measurement.")
        return 0
    print(f"SELFTEST FAILED: {results.count(False)} direction(s) broken - do not trust this gate.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="processed_datasets/BOXRR-23_Dataset/users")
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--sample-time", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=20)
    parser.add_argument("--window-stride", type=float, default=5)
    parser.add_argument("--channels", default="full")
    parser.add_argument("--encoding", default="raw")
    parser.add_argument("--resample", default="nearest")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--pairs-per-user", type=int, default=64)
    # BooleanOptionalAction, not store_true: `store_true` with `default=True` yields a
    # flag that can never be turned off, which is misleading on a script another machine
    # runs from a spec. --no-cross-session-positives now works.
    parser.add_argument("--cross-session-positives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare-with", type=Path, help="the other machine's JSON")
    parser.add_argument("--emit-manifest", type=Path, help="write our pairs as .npz for the other machine")
    parser.add_argument("--use-manifest", type=Path, help="score THEIR pairs instead of ours")
    parser.add_argument("--selftest", action="store_true", help="verify compare() needs no data")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    report = run(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    if args.compare_with:
        return compare(report, json.loads(args.compare_with.read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
