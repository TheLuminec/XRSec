"""
The second column: per-corpus LODO checkpoints, which saw six OTHER seated corpora.

The 9.3 five saw only Beat Saber and Half-Life Alyx, so a seated corpus is doubly foreign
to them - unseen users AND an unseen activity. The LODO checkpoints held out one seated
corpus and trained on the rest, so if seated viewing has a transferable dynamic style, this
column should find it and the 9.3 column should not.

Prediction registered before running: the two columns land within 0.05 of each other,
because content-driven motion is a property of the video rather than of the viewer and
training on other people's video-watching therefore teaches little that transfers.

Gated like everything else: each checkpoint reproduces its own recorded metric on its own
recorded users before it is allowed to produce a rank-1.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import build_sample_index              # noqa: E402
from normalization import ChannelNormalizer         # noqa: E402
from utils import load_checkpoint                   # noqa: E402
from score_nymeria import gate                      # noqa: E402
from step6_seated_dyn import (CORPORA, DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, euclid, cosine, rank1, zscore)

LODO = {  # corpus -> the dyn checkpoint that held exactly this corpus out
    "ViewGauss":        "runs/2026-09-04/11-12-25_train/checkpoints/lodo_multi7_bilstm_5s_20hz_emb128_train.pth",
    "Head_and_Gaze":    "runs/2026-09-04/11-18-26_train/checkpoints/lodo_multi7_bilstm_5s_20hz_emb128_train.pth",
    "VR_User_Behavior": "runs/2026-09-04/11-07-03_train/checkpoints/lodo_multi7_bilstm_5s_20hz_emb128_train.pth",
}
NINE_THREE = {"ViewGauss": 0.187, "Head_and_Gaze": 0.179, "VR_User_Behavior": 0.245}

if __name__ == "__main__":
    print(f"device {DEVICE}; chance at N={N_SMALL} is {1/N_SMALL:.4f}\n")

    gates = [dict(gate(p), corpus=name) for name, p in LODO.items()]
    pathlib.Path("docs/acceptance/step6_lodo_dyn_gate.json").write_text(json.dumps(gates, indent=1),
                                                                        encoding="utf-8")
    if not all(g["passed"] for g in gates):
        print("\n*** GATE FAILED - report the mismatch, do not compute rank-1 ***")
        sys.exit(1)
    print("\ngate: 3/3 passed\n")

    results = {}
    for name, ckpt in LODO.items():
        dirname, k, _ = CORPORA[name]
        users_dir = str((ROOT / "processed_datasets" / dirname / "users").resolve())
        model, ck = load_checkpoint(ckpt, DEVICE, 100, return_checkpoint=True)
        dyn = quiet(build_sample_index, users_dir, sample_time=5, sample_rate=20, encoding="dyn")
        raw = quiet(build_sample_index, users_dir, sample_time=5, sample_rate=20)
        assert torch.equal(dyn.window_session_ids, raw.window_session_ids)
        quiet(ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit").transform, dyn)

        pop = population(raw, k)
        pos = raw.samples[:, 4:7, :].mean(dim=2).numpy()
        emb = embed(model, dyn.samples)
        g_e, p_e, o = templates(emb, pop, k, np.random.default_rng(SEED))
        g_p, p_p, _ = templates(pos, pop, k, np.random.default_rng(SEED))
        d_dyn, d_y = cosine(p_e, g_e), euclid(p_p, g_p, [1])
        row = {lab: rank1(d, o, N_SMALL, np.random.default_rng(SEED))
               for lab, d in (("dyn", d_dyn), ("y", d_y), ("y+dyn", zscore(d_dyn) + zscore(d_y)))}
        row.update(k=k, users=len(pop), nine_three_dyn=NINE_THREE[name],
                   delta=row["dyn"] - NINE_THREE[name])
        results[name] = row
        print(f"  {name:<20} k={k:<3} users {len(pop):>3}   dyn {row['dyn']:.3f}   "
              f"y {row['y']:.3f}   y+dyn {row['y+dyn']:.3f}   "
              f"vs 9.3 dyn {NINE_THREE[name]:.3f}  delta {row['delta']:+.3f}", flush=True)
    pathlib.Path("docs/acceptance/step6_lodo_dyn.json").write_text(json.dumps(results, indent=1),
                                                                   encoding="utf-8")
