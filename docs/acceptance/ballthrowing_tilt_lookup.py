"""
Broad 2 s Q3b (broad_2s_REGISTERED.md): training-free tilt lookup on ball-throwing. Per throw window, the mean
world up-vector in the device frame (x ~ roll, z ~ pitch, from the recorded quaternion, before any encoding),
standardised per session type over the 41 people; rank-1 at N=41, gallery = session mean, probe = each throw,
Euclidean distance, the harness's rank <= 1 rule; C1 (same headset, other day) and C2 (other headset).

Gate: the same code path on head height must reproduce the recorded C1_height / C2_height per user of
ballthrowing_cross_day_s1.json exactly (height is checkpoint-independent).

    python docs/acceptance/ballthrowing_tilt_lookup.py [--out docs/acceptance/ballthrowing_tilt_lookup.json]
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model")); sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
from ballthrowing_cross_day import C1_PAIRS, C2_PAIRS, CORPUS, SESSIONS, rows, height_rank1  # noqa: E402
from across_xr_alignment import quiet  # noqa: E402


def load():
    from dataset import SampleDataset, SampleIndex
    ds = quiet(SampleDataset, str(CORPUS / "users"), sample_time=2, sample_rate=20, channels="full",
               resample="nearest", window_stride=5)
    index = SampleIndex(ds, encoding="raw")
    assert len(ds.user_dirs) == 41 and index.sample_count == 2460
    q = index.samples[:, :4, :].double().numpy()                       # x, y, z, w; raw, never normalised
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    up = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1)   # R(q)^T e_y
    assert np.allclose(np.linalg.norm(up, axis=1), 1, atol=1e-4)
    tilt = up[:, [0, 2], :].mean(axis=2)                                # (windows, 2): roll-ish, pitch-ish
    heights = index.window_mean_positions[:, 1].numpy().astype(float)
    sess = index.window_session_ids.numpy(); where = {}; users = []
    for u, (d, rr) in enumerate(zip(ds.user_dirs, index.user_sample_indices)):
        users.append(pathlib.Path(d).name)
        csvs = sorted(f for f in os.listdir(d) if f.endswith(".csv"))
        for r in rr.numpy():
            s, t = csvs[sess[r]][:-4].split("_throw"); where[(u, s, int(t))] = int(r)
    assert len(where) == 2460
    return tilt, heights, users, where


def standardise(feat, where, n):
    f = feat.reshape(len(feat), -1).astype(float).copy()
    for s in SESSIONS:
        idx = np.concatenate([rows(where, u, s) for u in range(n)])
        f[idx] = (f[idx] - f[idx].mean(axis=0)) / f[idx].std(axis=0)
    return f


def rank1_nd(f, gal_rows, probe_rows):
    gal = np.stack([f[r].mean(axis=0) for r in gal_rows])
    out = np.zeros(len(gal_rows))
    for i, pr in enumerate(probe_rows):
        d = np.linalg.norm(f[pr][:, None, :] - gal[None, :, :], axis=2)
        own = d[:, i]
        better = (d < own[:, None]).sum(axis=1); ties = (d == own[:, None]).sum(axis=1) - 1
        out[i] = float(np.mean((better == 0) & (ties == 0)))
    return out


def conditions(f, where, n, fn):
    res = {}
    for cond, pairs in (("C1", C1_PAIRS), ("C2", C2_PAIRS)):
        per = []
        for a, b in pairs:
            ga = [rows(where, u, a) for u in range(n)]; gb = [rows(where, u, b) for u in range(n)]
            per.append((fn(f, ga, gb) + fn(f, gb, ga)) / 2)
        res[cond] = np.mean(per, axis=0)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "docs/acceptance/ballthrowing_tilt_lookup.json"))
    a = ap.parse_args()
    tilt, heights, users, where = load(); n = len(users)
    rec = json.loads((ROOT / "docs/acceptance/ballthrowing_cross_day_s1.json").read_text())["results"][0]
    h = standardise(heights, where, n)
    mine_nd = conditions(h, where, n, rank1_nd)                                        # this file's path
    mine_1d = conditions(h[:, 0], where, n, height_rank1)                              # the harness's own path
    gaps = {k: max(max(abs(mine_nd[k][i] - rec[f"{k}_height"][u]), abs(mine_1d[k][i] - rec[f"{k}_height"][u]))
                   for i, u in enumerate(users)) for k in ("C1", "C2")}
    gate = {"passed": all(g == 0 for g in gaps.values()), "max_abs_gap_vs_recorded_height": gaps}
    out = {"registered": "docs/acceptance/broad_2s_REGISTERED.md Q3b", "gate": gate, "n_users": n, "chance": 1 / n}
    if gate["passed"]:
        t = standardise(tilt, where, n)
        res = conditions(t, where, n, rank1_nd)
        out["tilt"] = {k: dict(zip(users, map(float, v))) for k, v in res.items()}
        out["tilt_means"] = {k: float(v.mean()) for k, v in res.items()}
        for j, nm in ((0, "roll_only"), (1, "pitch_only")):
            r1 = conditions(t[:, [j]], where, n, rank1_nd)
            out[f"{nm}_means_descriptive"] = {k: float(v.mean()) for k, v in r1.items()}
        out["raw_tilt_by_session_type_deg"] = {s: [float(np.degrees(np.arcsin(np.clip(tilt[np.concatenate([rows(where, u, s) for u in range(n)]), j].mean(), -1, 1))))
                                                   for j in (0, 1)] for s in SESSIONS}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "tilt"}, indent=1))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
