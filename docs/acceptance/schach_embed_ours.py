"""
Stage 1 of the Schach paired comparison (Amendment 8): embed Across-XR with our gated
checkpoints on the CPU and dump what the scoring stage needs, so the scoring stage can run in
an environment that has their metric library and none of the pipeline's dependencies.

For each checkpoint: re-gate on CPU through the pipeline's own loader (tolerance 2e-3, the
documented CPU/GPU cuDNN gap), then embed all 49 users x 5 applications at the checkpoint's
own resolution and write an .npz with embeddings, user id, application index (1..5 = game_id,
the same numbering as their `comment`) and window start time.

    XRSEC_SAMPLE_CACHE_DIR=/run/media/feng/Data/CalebProject/XRSec/.cache/samples \
      .venv313/bin/python docs/acceptance/schach_embed_ours.py --arm zero_shot \
      --checkpoints runs/.../best.pth ... --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import across_xr_alignment as axa  # noqa: E402

GAME_ID = {g: i + 1 for i, g in enumerate(axa.GAMES)}  # 1 Superhot .. 5 Social VR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = torch.device("cpu")
    torch.set_num_threads(4)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = []
    for k, ck_path in enumerate(args.checkpoints, start=1):
        print(f"\n=== {args.arm} seed-slot {k}: {ck_path} ===", flush=True)
        g, model, ck = axa.gate(ck_path, device)
        gates.append(g)
        assert g["passed"], g
        assert g["eval_users"] == 17, g["eval_users"]
        t0 = time.time()
        corpus = axa.Corpus(ck, device, model)
        print(f"  embedded {len(corpus.embeddings)} windows ({corpus.encoding}, {corpus.sample_time}s, "
              f"stride {corpus.stride}) in {time.time() - t0:.0f}s", flush=True)
        assert corpus.sample_time == 10 and int(corpus.stride) == 5, (corpus.sample_time, corpus.stride)
        assert corpus.encoding in ("dyn", "raw"), corpus.encoding
        user_id = corpus.user_ids[corpus.window_user]
        app = np.array([GAME_ID[a] for a in corpus.window_app], dtype=np.int64)
        path = out_dir / f"ours_{args.arm}_seed{int(ck.get('seed', k))}.npz"
        np.savez_compressed(path, embeddings=corpus.embeddings.astype(np.float32), user_id=user_id,
                            app=app, start=corpus.window_start.astype(np.float32),
                            checkpoint=np.array(g["checkpoint"]), run_id=np.array(g["run_id"]),
                            seed=np.array(int(ck.get("seed", k))), encoding=np.array(corpus.encoding))
        print(f"  wrote {path}", flush=True)
    (out_dir / f"ours_{args.arm}_gate_cpu.json").write_text(json.dumps(gates, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
