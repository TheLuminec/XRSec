"""Score the e240 pair on every corpus it never saw (e240_transfer_REGISTERED.md). Gate first on each
checkpoint's own 48 held-out users (twice-deterministic CPU figure), then one loader per corpus plus the
pooled seated seven, plus a within-application pass on Across-XR and Questset. Rows -> this machine's
shard (experiment e240_transfer, mode rescore); certificate -> docs/acceptance/e240_transfer.json."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
import results_log  # noqa: E402
from dataset import SiameseDataset, _seed_value  # noqa: E402
from eval import evaluate  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402
sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
from nymeria_script_pair import build_eval, _here  # noqa: E402  (the gated held-out-set builder)

DEVICE = torch.device("cpu")
PD = ROOT / "processed_datasets"
SEATED = ["ViewGauss_Head-Movement_Dataset", "Head_and_Gaze_Behavior_Dataset",
          "VR_User_Behavior_Dataset_(Spherical_Video_Streaming)", "NJIT_6DOF_VR_Navigation_Dataset",
          "EyeNavGS_6-DoF_Navigation_Dataset", "Panonut360_Dataset", "360-degree_Saliency_Dataset_(PanoSaliency)"]
CROSS_APP = ["CrossApplicationXR_Dataset", "Questset"]
KNOWN = {"treatment": (0.732900792, 0.7232780555884043), "control": (0.540551503, 0.7232780555884043)}
OUT = ROOT / "docs" / "acceptance" / "e240_transfer.json"

def loader(ck, dirs, seed, cross_session=True):
    es = ck["eval_split"]
    ds = SiameseDataset([str(d) for d in dirs], samples_per_user=512, sample_time=int(es["sample_time"]),
                        sample_rate=int(es["sample_rate"]), exclude_users=[], swap_data=False, seed=_seed_value(seed, 4),
                        within_dataset_negatives=True, channels=ck.get("channels", "full"), window_stride=es.get("window_stride"),
                        cross_session_positives=cross_session, encoding=es.get("encoding", "raw"))
    norm = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if norm.enabled: norm.transform(ds.sample_index)
    ds.unseen_datasets = dict(norm.unseen_datasets)
    return DataLoader(ds, batch_size=256, shuffle=False)

def score(model, ck, dirs, seed, label, cross_session=True):
    ld = loader(ck, dirs, seed, cross_session); ds = ld.dataset
    _, acc, m = evaluate(model, ld, nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
    labels = ds.manifest["labels"].view(-1)
    rec = {"label": label, "cross_session_positives": cross_session, "users": int(ds.sample_index.num_users),
           "pairs": int(labels.numel()), "positive_fraction": float(labels.float().mean()), "auc": float(m["auc"]),
           "eer": float(m["eer"]), "acc": float(acc), "lookup_auc_encoded": m.get("lookup_auc"),
           "position_lookup_auc": m.get("position_lookup_auc"), "amplitude_auc": m.get("amplitude_auc"),
           "by_dataset": m.get("by_dataset") or {}, "same_session_fallback": None}
    print(f"  {label:44s} users {rec['users']:4d} pairs {rec['pairs']:6d}  AUC {rec['auc']:.4f}  EER {rec['eer']:.4f}  "
          f"pos_lookup {rec['position_lookup_auc'] or float('nan'):.4f}  amplitude {rec['amplitude_auc'] or float('nan'):.4f}", flush=True)
    return rec, ds, m, acc

def log_row(ckpt, ck, seed, dirs, label, m, acc, ds, experiment="e240_transfer"):
    es = ck["eval_split"]; labels = ds.manifest["labels"].view(-1)
    history = {"selected_test_auc": m["auc"], "selected_test_eer": m["eer"], "best_test_auc": m["auc"], "best_test_eer": m["eer"],
               "selected_test_acc": acc, "lookup_auc": m.get("lookup_auc"), "position_lookup_auc": m.get("position_lookup_auc"),
               "amplitude_auc": m.get("amplitude_auc"), "selected_test_by_dataset": m.get("by_dataset") or {},
               "unseen_datasets": getattr(ds, "unseen_datasets", {}), "eval_positive_fraction": float(labels.float().mean()),
               "best_epoch": int(ck.get("epoch", 0))}
    cfg = SimpleNamespace(mode="rescore", experiment_name=experiment, extractor=ck.get("extractor"), extractor_params=None,
                          objective=ck.get("objective"), identity_margin=0.35, identity_scale=30.0, balance_identities=False, balance_cap=None,
                          head=ck.get("head"), channels=ck.get("channels", "full"), encoding=es.get("encoding", "raw"), resample="nearest",
                          window_stride=es.get("window_stride"), sweep_id="", fold=label, normalize="per_dataset", within_dataset_negatives=True,
                          cross_session_positives=True, center_position=False, max_users=None, eval_normalize="target_fit", seed=seed,
                          sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]), embedding_dim=int(ck.get("embedding_dim", 128)),
                          samples_per_user=512, batch_size=256, lr=0.001, weight_decay=0.0, val_user_fraction=0.25, epochs=None,
                          early_stopping_patience=None, data_dirs=list(es.get("data_dirs") or []), test_dirs=[str(d) for d in dirs],
                          exclude_users=[], validation_users=[], drop_users=[], swap_data=False, test_on_excluded=False,
                          model_path=str(ckpt), save_path=str(ckpt), boosting=None)
    return str(results_log.append_run(cfg, history, dataset_tag=label))

def main():
    ckpts = {a: Path(p) for a, p in (x.split("=", 1) for x in sys.argv[1:])}
    only = os.environ.get("ONLY")  # smoke: one corpus name
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arm, ckpt in ckpts.items():
        t0 = time.time(); model, ck = load_checkpoint(str(ckpt), DEVICE, 200, return_checkpoint=True); seed = int(ck.get("seed", 1))
        # gate: own held-out users, known CPU figure
        ds = build_eval(ck, seed); _, _, m = evaluate(model, DataLoader(ds, batch_size=256, shuffle=False), nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
        want_auc, want_lookup = KNOWN[arm]; gap = abs(float(m["auc"]) - want_auc)
        gate = {"rescored": float(m["auc"]), "known_cpu": want_auc, "gap": gap, "position_lookup": m.get("position_lookup_auc"),
                "passed": gap <= 1e-6 and abs(float(m.get("position_lookup_auc")) - want_lookup) <= 1e-12}
        print(f"gate {arm}: rescored {m['auc']:.9f} known {want_auc:.9f} gap {gap:.1e} lookup {m.get('position_lookup_auc'):.13f} {'PASS' if gate['passed'] else 'FAIL'}", flush=True)
        entry = results.setdefault(arm, {}); entry.update(checkpoint="/".join(ckpt.resolve().parts[-4:]), gate=gate, corpora={})
        OUT.write_text(json.dumps(results, indent=1))
        if not gate["passed"]: print(f"  {arm}: gate failed, nothing read", flush=True); continue
        corpora = [c for c in SEATED + CROSS_APP if (PD / c / "users").is_dir()]
        if only: corpora = [c for c in corpora if c == only]
        for c in corpora:
            rec, ds, m, acc = score(model, ck, [PD / c / "users"], seed, c)
            rec["row"] = log_row(ckpt, ck, seed, [PD / c / "users"], c, m, acc, ds); entry["corpora"][c] = rec
            if c in CROSS_APP:
                rec2, ds2, m2, acc2 = score(model, ck, [PD / c / "users"], seed, c + " [within-application]", cross_session=False)
                entry["corpora"][c + " [within-application]"] = rec2
            OUT.write_text(json.dumps(results, indent=1))
        if not only:
            rec, ds, m, acc = score(model, ck, [PD / c / "users" for c in SEATED], seed, "SEATED_SEVEN_POOLED")
            rec["row"] = log_row(ckpt, ck, seed, [PD / c / "users" for c in SEATED], "seated_seven_pooled", m, acc, ds); entry["corpora"]["SEATED_SEVEN_POOLED"] = rec
            OUT.write_text(json.dumps(results, indent=1))
        entry["seconds"] = round(time.time() - t0, 1); OUT.write_text(json.dumps(results, indent=1)); print(f"{arm} done in {entry['seconds']} s", flush=True)

if __name__ == "__main__":
    main()
