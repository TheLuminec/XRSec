"""Score checkpoints on Nymeria_LSO_test (nymeria_lso_REGISTERED.md): the standard figure (cross-recording
positives, within-corpus negatives) and the constrained one (cross-script positives / same-script
negatives) on the same embeddings, after each checkpoint reproduces its own row's figure on its own
evaluation set. Rows -> this machine's shard (experiment lso_reference, mode rescore); certificate ->
docs/acceptance/nymeria_lso_score.json.

    DEVICE=cpu .venv/bin/python docs/acceptance/nymeria_lso_score.py --shard <shard.jsonl> label=path [...]
labels: control_s1 treatment_s1 lso_s1 ...  (the label names the arm; the row is matched by path tail)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model")); sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
from dataset import SiameseDataset, _seed_value  # noqa: E402
from eval import evaluate  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402
from nymeria_script_pair import (build_eval, _here, scripts_by_sequence, window_scripts, embed_all,  # noqa: E402
                                 constrained_pairs, unconstrained_pairs, auc_of)
from e240_transfer import log_row  # noqa: E402

DEVICE = torch.device(os.environ.get("DEVICE", "cpu")); GATE_TOL = {"cuda": 1e-4, "cpu": 1e-3}[DEVICE.type]
LSO_TEST = ROOT / "processed_datasets" / "Nymeria_LSO_test" / "users"
OUT = ROOT / "docs" / "acceptance" / "nymeria_lso_score.json"

def own_eval(ck, seed):
    """The checkpoint's own evaluation set: test_dirs if it trained with them (the LSO arm), else the
    excluded users of data_dirs (the in-domain arms)."""
    es = ck["eval_split"]
    if es.get("test_dirs"):
        return loader_for(ck, [_here(d) for d in es["test_dirs"]], seed, cross_session=True)
    return DataLoader(build_eval(ck, seed), batch_size=256, shuffle=False)

def loader_for(ck, dirs, seed, cross_session=True):
    es = ck["eval_split"]
    ds = SiameseDataset(dirs, samples_per_user=512, sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
                        exclude_users=[], swap_data=False, seed=_seed_value(seed, 4), within_dataset_negatives=True,
                        channels=ck.get("channels", "full"), window_stride=es.get("window_stride"),
                        cross_session_positives=cross_session, encoding=es.get("encoding", "raw"))
    norm = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if norm.enabled: norm.transform(ds.sample_index)
    ds.unseen_datasets = dict(norm.unseen_datasets)
    return DataLoader(ds, batch_size=256, shuffle=False)

def main():
    args = sys.argv[1:]; shards = []
    while args and args[0] == "--shard": shards.append(args[1]); args = args[2:]
    rows = []
    for s in shards: rows += [json.loads(l) for l in Path(s).read_text().splitlines() if l.strip()]
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    table = scripts_by_sequence()
    for spec in args:
        label, path = spec.split("=", 1); ckpt = Path(path); t0 = time.time()
        tail = "/".join(ckpt.resolve().parts[-4:])
        hits = [r for r in rows if (r.get("checkpoint") or "").replace("\\", "/").endswith(tail) and r.get("mode") in (None, "train")]
        assert len(hits) == 1, f"{label}: {len(hits)} rows match {tail}"
        row = hits[-1]; model, ck = load_checkpoint(str(ckpt), DEVICE, 200, return_checkpoint=True); seed = int(ck.get("seed", row["seed"]))
        _, _, m = evaluate(model, own_eval(ck, seed), nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
        gap = abs(float(m["auc"]) - float(row["selected_test_auc"]))
        rec = {"label": label, "checkpoint": tail, "seed": seed, "row_auc": row["selected_test_auc"], "gate_rescored": float(m["auc"]),
               "gate_gap": gap, "gate_tol": GATE_TOL, "device": DEVICE.type, "gate_passed": gap <= GATE_TOL}
        print(f"gate {label}: row {row['selected_test_auc']:.6f} rescored {m['auc']:.6f} gap {gap:.1e} {'PASS' if rec['gate_passed'] else 'FAIL'}", flush=True)
        if rec["gate_passed"]:
            ld = loader_for(ck, [str(LSO_TEST)], seed); ds = ld.dataset
            _, acc, mt = evaluate(model, ld, nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
            scripts = window_scripts(ds.sample_index, table); assert all(s is not None for s in scripts)
            E = embed_all(model, ds.sample_index.samples); rng = np.random.default_rng(_seed_value(seed, 2))
            pos, neg, skipped = constrained_pairs(ds.sample_index, scripts, rng)
            rec.update(lso_users=int(ds.sample_index.num_users), lso_windows=int(ds.sample_index.sample_count),
                       lso_auc=float(mt["auc"]), lso_eer=float(mt["eer"]), lso_acc=float(acc),
                       position_lookup_auc=mt.get("position_lookup_auc"), amplitude_auc=mt.get("amplitude_auc"),
                       constrained_auc=auc_of(model, E, pos, neg), constrained_pairs=[len(pos), len(neg)], skipped_users=skipped,
                       same_session_fallback=None, row=log_row(ckpt, ck, seed, [LSO_TEST], f"lso_reference:{label}", mt, acc, ds, experiment="nymeria_lso_reference"))
            print(f"  {label}: LSO_test users {rec['lso_users']} windows {rec['lso_windows']} | AUC {rec['lso_auc']:.4f} EER {rec['lso_eer']:.4f} | "
                  f"constrained {rec['constrained_auc']:.4f} on {len(pos)}+{len(neg)} | pos_lookup {rec['position_lookup_auc']:.4f} amplitude {rec['amplitude_auc']:.4f}", flush=True)
        rec["seconds"] = round(time.time() - t0, 1); results[label] = rec; OUT.write_text(json.dumps(results, indent=1))

if __name__ == "__main__": main()
