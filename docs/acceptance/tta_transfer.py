"""Experiment 3: label-free test-time adaptation of the embedding (tta_transfer_REGISTERED.md).
For each unseen corpus: the pipeline's own pairs and embeddings, then AUC under none / centre / CORAL.
`none` must reproduce evaluate()'s AUC on the same manifest exactly (the gate for the embedding path).

    ONLY=<corpus> .venv/bin/python docs/acceptance/tta_transfer.py treatment=<ckpt> control=<ckpt>
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model")); sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
from dataset import build_sample_index  # noqa: E402
from eval import evaluate  # noqa: E402
from metrics import roc_auc  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402
from e240_transfer import loader, SEATED, CROSS_APP, PD, DEVICE, _here  # noqa: E402
from nymeria_script_pair import embed_all  # noqa: E402
OUT = ROOT / "docs" / "acceptance" / "tta_transfer.json"
SOURCE_USERS = 100   # BOXRR training users, first in sorted order, for the source covariance

def cosine_auc(E, ds):
    m = ds.manifest; a, b = m["x1_indices"].view(-1), m["x2_indices"].view(-1)
    s = torch.nn.functional.cosine_similarity(E[a], E[b], dim=1)
    return float(roc_auc(s, m["labels"].view(-1).float()))

def coral(Et, Es):
    """Whiten target embeddings with their covariance, re-colour with the source covariance."""
    def cov(X): Xc = X - X.mean(0, keepdim=True); return Xc.T @ Xc / (X.shape[0] - 1) + 1e-4 * torch.eye(X.shape[1])
    def sqrt_inv(C):
        w, U = torch.linalg.eigh(C); return U @ torch.diag(w.clamp_min(1e-8) ** -0.5) @ U.T
    def sqrt(C):
        w, U = torch.linalg.eigh(C); return U @ torch.diag(w.clamp_min(0) ** 0.5) @ U.T
    Ct, Cs = cov(Et.double()), cov(Es.double())
    return ((Et.double() - Et.double().mean(0, keepdim=True)) @ sqrt_inv(Ct) @ sqrt(Cs) + Es.double().mean(0, keepdim=True)).float()

def source_embeddings(model, ck, seed):
    es = ck["eval_split"]; boxrr = _here([d for d in es["data_dirs"] if "BOXRR" in d][0])
    users = sorted(p for p in Path(boxrr).iterdir() if p.is_dir())[:SOURCE_USERS]
    idx = build_sample_index([str(boxrr)], sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
                             channels=ck.get("channels", "full"), window_stride=es.get("window_stride"), encoding=es.get("encoding", "raw"),
                             keep_users=[str(u) for u in users])
    ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit").transform(idx)
    return embed_all(model, idx.samples)

def main():
    ckpts = {a: Path(p) for a, p in (x.split("=", 1) for x in sys.argv[1:])}; only = os.environ.get("ONLY")
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arm, ckpt in ckpts.items():
        model, ck = load_checkpoint(str(ckpt), DEVICE, 200, return_checkpoint=True); seed = int(ck.get("seed", 1))
        Es = source_embeddings(model, ck, seed); entry = results.setdefault(arm, {}); entry.update(checkpoint="/".join(ckpt.resolve().parts[-4:]), source_windows=int(Es.shape[0]), corpora={})
        corpora = [c for c in SEATED + CROSS_APP if (PD / c / "users").is_dir()]
        targets = [(c, [PD / c / "users"]) for c in corpora if not only or c == only] + ([("SEATED_SEVEN_POOLED", [PD / c / "users" for c in SEATED])] if not only else [])
        for name, dirs in targets:
            t0 = time.time(); ld = loader(ck, dirs, seed); ds = ld.dataset
            _, _, m = evaluate(model, ld, nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
            E = embed_all(model, ds.sample_index.samples)
            none = cosine_auc(E, ds); gate_gap = abs(none - float(m["auc"]))
            centre = cosine_auc(E - E.mean(0, keepdim=True), ds); cor = cosine_auc(coral(E, Es), ds)
            rec = {"users": int(ds.sample_index.num_users), "pairs": int(ds.manifest["labels"].numel()), "evaluate_auc": float(m["auc"]),
                   "none": none, "gate_gap": gate_gap, "gate_passed": gate_gap <= 1e-6, "centre": centre, "coral": cor,
                   "d_centre": centre - none, "d_coral": cor - none, "seconds": round(time.time() - t0, 1)}
            entry["corpora"][name] = rec; OUT.write_text(json.dumps(results, indent=1))
            print(f"  {arm:9s} {name:44s} none {none:.4f} (gate gap {gate_gap:.1e} {'PASS' if rec['gate_passed'] else 'FAIL'}) | centre {centre:.4f} ({centre-none:+.4f}) | CORAL {cor:.4f} ({cor-none:+.4f})", flush=True)

if __name__ == "__main__": main()
