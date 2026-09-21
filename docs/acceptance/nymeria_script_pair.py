"""Script-pair follow-up for the Nymeria in-domain arm (nymeria_in_domain_REGISTERED.md, Amendment 10).

Does the treatment's gain read the PERSON's motion or WHICH ACTIVITIES the person did? On the same 48
held-out users and the same checkpoint, score positives only ACROSS scripts (same person, different
script) and negatives only WITHIN a script (two people, same script): an activity cue has nothing to
read there. Gate first: the harness rebuilds the row's held-out evaluation set exactly as the pipeline
did (same dirs, exclusions, swap, seed, normaliser) and must reproduce the row's selected_test_auc
within GATE_TOL on CPU (CLAUDE.md: CPU-vs-GPU differs by up to 7e-4; below ~1e-3 is arithmetic) before
a constrained figure is read from that checkpoint.

    .venv/bin/python docs/acceptance/nymeria_script_pair.py --shard <miami shard.jsonl> <ckpt.pth> [...]

Writes docs/acceptance/nymeria_script_pair.json (one entry per checkpoint) and prints each row's
verdict against the registered partition.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
from dataset import SiameseDataset, _seed_value  # noqa: E402
from eval import evaluate  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402
from metrics import roc_auc  # noqa: E402

DEVICE = torch.device("cpu")
GATE_TOL = 1e-3
PAIRS_PER_USER = 256           # per class per anchor user; the pipeline's manifest is 512 at 0.5
SCRIPTS = ROOT / "docs" / "acceptance" / "nymeria_sequence_scripts.csv"
OUT = ROOT / "docs" / "acceptance" / "nymeria_script_pair.json"
BAND, FALSIFIER = 0.65, 0.58   # registered: >= band credited; < falsifier activity-mix; between: partly

def _here(p: str) -> str:
    q = str(p).replace("\\", "/"); i = q.find("processed_datasets/")
    return str(ROOT / q[i:]) if i >= 0 else str(p)

def load_rows(shards) -> list[dict]:
    rows = []
    for s in shards:
        rows += [json.loads(l) for l in Path(s).read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("experiment") == "nymeria_in_domain" and int(r.get("sample_time", 0)) == 10
            and r.get("encoding") == "dyn"]

def match_row(ckpt: Path, rows: list[dict]) -> dict:
    tail = "/".join(ckpt.resolve().parts[-4:])          # 2026-09-21/<run>/checkpoints/<base>
    hits = [r for r in rows if (r.get("checkpoint") or "").replace("\\", "/").endswith(tail)]
    assert len(hits) == 1, f"{len(hits)} rows match {tail}"
    return hits[0]

def scripts_by_sequence() -> dict[tuple[str, str], str]:
    with open(SCRIPTS) as f:
        return {(r["participant"], r["act"]): r["script"] for r in csv.DictReader(f)}

def build_eval(ck: dict, seed: int) -> SiameseDataset:
    es = ck["eval_split"]
    ds = SiameseDataset([_here(d) for d in es["data_dirs"]], samples_per_user=int(es.get("samples_per_user") or 512),
                        sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
                        exclude_users=[_here(u) for u in es["exclude_users"]], swap_data=not bool(es.get("swap_data", False)),
                        seed=_seed_value(seed, 2), within_dataset_negatives=True, channels=ck.get("channels", "full"),
                        window_stride=es.get("window_stride"), cross_session_positives=True, encoding=es.get("encoding", "raw"))
    normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if normalizer.enabled:
        normalizer.transform(ds.sample_index)
    return ds

def window_scripts(index, table) -> list[str | None]:
    out = [None] * index.sample_count
    sess = index.window_session_ids.tolist()
    for u, user_dir in enumerate(index.user_dirs):
        files = sorted(f for f in os.listdir(user_dir) if f.endswith(".csv"))   # UserProfile's order
        name = Path(user_dir).name
        for w in index.user_sample_indices[u].tolist():
            out[w] = table.get((name, files[sess[w]][:-4]))
    return out

def embed_all(model, samples: torch.Tensor, batch: int = 512) -> torch.Tensor:
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, samples.shape[0], batch):
            outs.append(model.embed(samples[i:i + batch].to(DEVICE)))
    return torch.cat(outs, 0)

def constrained_pairs(index, scripts, rng):
    """Positives: same user, different script. Negatives: different user, same script. Balanced per user."""
    n_users = len(index.user_sample_indices)
    by_script = {}
    for w, s in enumerate(scripts):
        by_script.setdefault(s, []).append(w)
    win_user = np.empty(index.sample_count, dtype=np.int64)
    for u, idx in enumerate(index.user_sample_indices): win_user[idx.numpy()] = u
    pos, neg, skipped = [], [], {"no_cross_script_positive": 0, "no_same_script_negative": 0}
    for u in range(n_users):
        W = index.user_sample_indices[u].numpy()
        S = np.array([scripts[w] for w in W], dtype=object)
        if len(set(S.tolist())) < 2:
            skipped["no_cross_script_positive"] += 1; continue
        P, N = [], []
        tries = 0
        while len(P) < PAIRS_PER_USER and tries < 50 * PAIRS_PER_USER:
            tries += 1; i = int(rng.integers(len(W))); cand = W[S != S[i]]
            if len(cand): P.append((int(W[i]), int(rng.choice(cand))))
        tries = 0
        while len(N) < PAIRS_PER_USER and tries < 50 * PAIRS_PER_USER:
            tries += 1; i = int(rng.integers(len(W))); pool = np.array(by_script.get(S[i], []))
            pool = pool[win_user[pool] != u]
            if len(pool): N.append((int(W[i]), int(rng.choice(pool))))
        k = min(len(P), len(N))
        if k == 0: skipped["no_same_script_negative"] += 1; continue
        pos += P[:k]; neg += N[:k]
    return pos, neg, skipped

def unconstrained_pairs(index, scripts, rng):
    """The pipeline's shape on the same embeddings: cross-session positives (any script), random negatives."""
    n_users = len(index.user_sample_indices); sess = index.window_session_ids.numpy()
    win_user = np.empty(index.sample_count, dtype=np.int64)
    for u, idx in enumerate(index.user_sample_indices): win_user[idx.numpy()] = u
    pos, neg = [], []
    for u in range(n_users):
        W = index.user_sample_indices[u].numpy(); P, N = [], []
        for _ in range(PAIRS_PER_USER):
            i = int(rng.integers(len(W))); cand = W[sess[W] != sess[W[i]]]          # any other session of the same user
            if len(cand): P.append((int(W[i]), int(rng.choice(cand))))
            j = int(rng.integers(index.sample_count))
            if win_user[j] != u: N.append((int(W[i]), j))
        k = min(len(P), len(N)); pos += P[:k]; neg += N[:k]
    return pos, neg

def auc_of(model, E, pos, neg) -> float:
    pairs = pos + neg; a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
    with torch.no_grad():
        scores = model.score(E[a], E[b]).view(-1)
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    return float(roc_auc(scores, labels))

def verdict(x: float) -> str:
    return "BAND: credited (motion)" if x >= BAND else ("FALSIFIER: activity mix" if x < FALSIFIER else "BETWEEN: partly activity mix")

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", action="append", required=True); ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args(); rows = load_rows(a.shard); table = scripts_by_sequence()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for c in a.ckpts:
        ckpt = Path(c); row = match_row(ckpt, rows); t0 = time.time()
        arm = "treatment" if int(row.get("num_drop_users", 0)) == 141 else "control"
        model, ck = load_checkpoint(str(ckpt), DEVICE, 200, return_checkpoint=True)
        ds = build_eval(ck, int(ck.get("seed", row["seed"])))
        _, _, m = evaluate(model, DataLoader(ds, batch_size=256, shuffle=False), nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
        gap = abs(float(m["auc"]) - float(row["selected_test_auc"]))
        rec = {"checkpoint": "/".join(ckpt.resolve().parts[-4:]), "seed": row["seed"], "arm": arm, "run_id": row.get("run_id"),
               "recorded": row["selected_test_auc"], "gate_rescored_cpu": float(m["auc"]), "gate_gap": gap, "gate_tol": GATE_TOL,
               "gate_passed": gap <= GATE_TOL, "users": len(ds.sample_index.user_sample_indices), "windows": ds.sample_index.sample_count}
        print(f"gate seed {row['seed']} {arm}: recorded {row['selected_test_auc']:.6f} rescored {m['auc']:.6f} gap {gap:.1e} "
              f"{'PASS' if rec['gate_passed'] else 'FAIL'}  ({rec['users']} users, {rec['windows']} windows)", flush=True)
        if rec["gate_passed"]:
            scripts = window_scripts(ds.sample_index, table)
            assert all(s is not None for s in scripts), "a window has no script label"
            E = embed_all(model, ds.sample_index.samples)
            rng = np.random.default_rng(_seed_value(int(row["seed"]), 2))
            pos, neg, skipped = constrained_pairs(ds.sample_index, scripts, rng)
            upos, uneg = unconstrained_pairs(ds.sample_index, scripts, rng)
            rec.update(constrained_auc=auc_of(model, E, pos, neg), constrained_pairs=[len(pos), len(neg)], skipped_users=skipped,
                       unconstrained_auc_same_embeddings=auc_of(model, E, upos, uneg), unconstrained_pairs=[len(upos), len(uneg)],
                       scripts_per_user_min=min(len({scripts[w] for w in idx.tolist()}) for idx in ds.sample_index.user_sample_indices),
                       verdict=verdict(rec["constrained_auc"]) if arm == "treatment" else "control: report against 0.50-0.56 / >0.60")
            print(f"  constrained AUC (cross-script pos / same-script neg) {rec['constrained_auc']:.4f} on {len(pos)}+{len(neg)} pairs | "
                  f"unconstrained on the same embeddings {rec['unconstrained_auc_same_embeddings']:.4f} | {rec['verdict']}", flush=True)
        rec["seconds"] = round(time.time() - t0, 1); results[rec["checkpoint"]] = rec
        OUT.write_text(json.dumps(results, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
