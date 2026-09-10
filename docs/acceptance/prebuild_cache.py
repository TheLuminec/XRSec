"""
Pre-build the sample cache for a training configuration, on CPU, ahead of the GPU job.

Why: on this node the GPU is shared with the Rack-2023 reproduction, so the training job
should hold the card for training, not for parsing 42 GB of BOXRR CSVs. The cache entry a
training run reads is keyed on `{channels}-{resample}-s{stride}` with the stride formatted
as passed, so this script passes exactly what `model/main.py` passes (an int stride from
the Hydra override); a float 5.0 writes a different key (`s5-0`) that training never reads,
which is what the 200 BOXRR entries left by the cross-machine gate script are.

    XRSEC_SAMPLE_CACHE_DIR=<repo>/.cache/samples python docs/acceptance/prebuild_cache.py \
        --dirs <dataset>/users ... --sample-time 10 --sample-rate 20 --window-stride 5

Clause 15: every BOXRR entry this writes is a derived copy inside the DUA's destruction
scope. `docs/acceptance/boxrr_inventory.py` derives the list from the cache filenames, so
nothing here needs to be recorded separately - but the resolution must be one the training
actually uses, or it is a cache set that only ever costs.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))

from dataset import build_sample_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--sample-time", type=int, required=True)
    ap.add_argument("--sample-rate", type=int, required=True)
    ap.add_argument("--window-stride", type=int, default=None)
    ap.add_argument("--channels", default="full")
    ap.add_argument("--resample", default="nearest")
    args = ap.parse_args()
    for d in args.dirs:
        t0 = time.time()
        index = build_sample_index(
            d, sample_time=args.sample_time, sample_rate=args.sample_rate,
            channels=args.channels, resample=args.resample,
            window_stride=args.window_stride,
        )
        print(f"{d}: {index.num_users} users, {index.sample_count} windows, "
              f"{time.time() - t0:.0f}s", flush=True)
        del index
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
