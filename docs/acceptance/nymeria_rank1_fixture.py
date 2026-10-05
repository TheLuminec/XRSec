"""
Fixtures for nymeria_rank1.py - pure numpy, no checkpoint, no corpus. Every fixture asserts its own
structure first (that it holds what it claims to hold), then the property, in both directions where a
direction exists.

  1. identity perfectly encoded           -> rank-1 1.0 under every protocol, N=all and N=17
  2. constant embedding                   -> every score tied: mean rank exactly (N+1)/2, rank-1 exactly 0
                                             (the A1 rule: a tie is never rank 1; 1/N is the expectation of
                                             an UNTIED random scorer, fixture 3, not of a constant one)
  3. random embedding                     -> rank-1 ~ 1/N at N=all and ~ 1/17 at N=17
  4. activity only, "signature activity"  -> constrained AT chance, unconstrained ABOVE chance
     (each person also records one script nobody else does, twice; a common script is recorded by all)
  5. activity only on the REAL 48-user script table (nymeria_sequence_scripts.csv x heldout48),
     random and bundle-aligned activity geometry, two noise levels, constant or variable windows per
     sequence -> the constrained CELL-BALANCED mean (registered) at chance in every realization; the
     per-user mean is NOT (its worst excess is measured and must equal nymeria_rank1.ACTIVITY_FLOOR);
     the rejected fallback leaks under bundle-aligned activity
  6. helpers: implied_rank1 reproduces the published alyx triple; auc_ranked constant 0.5, perfect 1.0;
     N=17 draws are deterministic (identical for every checkpoint, so the arms pair)
  7. read mode end to end on six stand-in runs: every registered verdict printed

    python docs/acceptance/nymeria_rank1_fixture.py
"""
from __future__ import annotations

import csv
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nymeria_rank1 import (DRAW_SEED, auc_ranked, implied_rank1, load_heldout,  # noqa: E402
                           score_protocol)

D = 64


def table_dense():
    """24 users, scripts 0..5; user u records every script except u % 6 (5 scripts, so every script has
    20 doers >= 17), plus a second sequence of script (u+1) % 6 when that is one of theirs; 8 windows
    per sequence."""
    rows = []          # (user, seq, script)
    seq = 0
    for u in range(24):
        mine = [s for s in range(6) if s != u % 6]
        dup = (u + 1) % 6
        for s in mine + ([dup] if dup in mine else []):
            rows += [(u, seq, s)] * 8
            seq += 1
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def table_signature():
    """40 users. Scripts 0 and 1 recorded once by everyone; script 2+u recorded TWICE by user u only."""
    rows, seq = [], 0
    for u in range(40):
        for s in (0, 1, 2 + u, 2 + u):
            rows += [(u, seq, s)] * 6
            seq += 1
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def table_real(rng, per_seq=None):
    """The real held-out structure: 48 users, their actual sequences and scripts. Window counts per
    sequence are not knowable without the corpus; constant 30, or drawn 15..60 when per_seq='random'."""
    held = load_heldout()
    rows = list(csv.DictReader(open(HERE / "nymeria_sequence_scripts.csv")))
    vocab = sorted({r["script"] for r in rows})
    out, seq = [], 0
    for u, name in enumerate(held):
        mine = sorted((r for r in rows if r["participant"] == name), key=lambda r: r["act"])
        for r in mine:
            k = 30 if per_seq is None else int(rng.integers(15, 61))
            out += [(u, seq, vocab.index(r["script"]))] * k
            seq += 1
    a = np.array(out)
    others = defaultdict(set)
    for r in rows:
        if r["participant"] not in held:
            others[r["participant"]].add(vocab.index(r["script"]))
    return a[:, 0], a[:, 1], a[:, 2], vocab, others


def noisy(base, rng, sd):
    return base + sd * rng.standard_normal(base.shape)


def check(name, cond, msg):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}: {msg}")
    assert cond, f"{name}: {msg}"


def main() -> int:
    rng = np.random.default_rng(0)

    # ---------------- structure of the synthetic tables (assert the fixture is what it claims)
    print("fixture structure")
    u, q, s = table_dense()
    doers = Counter(s[np.unique(q, return_index=True)[1]].tolist())
    per_user_scripts = {x: set(s[u == x].tolist()) for x in np.unique(u)}
    check("dense table", len(np.unique(u)) == 24 and all(len(v) == 5 for v in per_user_scripts.values())
          and all(len({x for x in np.unique(u) if t in per_user_scripts[x]}) == 20 for t in range(6))
          and len(u) > 0, f"24 users x 5 scripts, 20 doers per script, {len(u)} windows, {len(np.unique(q))} sequences")
    us, qs, ss = table_signature()
    sig = {x: set(ss[us == x].tolist()) for x in np.unique(us)}
    check("signature table", all(sig[x] == {0, 1, 2 + x} for x in sig)
          and all(len(np.unique(qs[(us == x) & (ss == 2 + x)])) == 2 for x in sig),
          "40 users; scripts 0,1 by all; script 2+u by user u only, two sequences")
    ur, qr, sr, vocab, others = table_real(rng)
    check("real table", len(np.unique(ur)) == 48 and len(vocab) == 20 and len(np.unique(qr)) == 216 and len(others) == 188,
          f"48 held-out users, {len(np.unique(qr))} sequences, {len(vocab)} scripts; 188 other participants for co-occurrence")

    # ---------------- 1. identity perfectly encoded
    print("1. identity perfectly encoded -> 1.0")
    base = np.eye(D)[u]
    emb = noisy(base, rng, 0.01)
    for p in ("constrained", "unconstrained", "pair", "fallback_all48"):
        r = score_protocol(emb, u, q, s, p, draws=20)
        full = r["full"]["per_user"][np.isfinite(r["full"]["per_user"])]
        check(p, len(full) == 24 and np.all(full == 1.0), f"N=all rank-1 {full.mean():.3f} on {len(full)} users")
        if p != "pair":
            n17 = r["n17"]["per_user"][np.isfinite(r["n17"]["per_user"])]
            check(p + " N=17", len(n17) == 24 and np.all(n17 == 1.0), f"rank-1 {n17.mean():.3f}")

    # ---------------- 2. constant embedding
    print("2. constant embedding -> rank (N+1)/2, rank-1 0")
    emb = np.ones((len(u), D))
    for p in ("constrained", "unconstrained"):
        r = score_protocol(emb, u, q, s, p, draws=20)
        n_cand = 20 if p == "constrained" else 24
        mr = r["full"]["mean_rank_per_user"]
        check(p, np.all(mr[np.isfinite(mr)] == (n_cand + 1) / 2) and np.nanmax(r["full"]["per_user"]) == 0.0
              and np.nanmax(r["n17"]["per_user"]) == 0.0,
              f"mean rank {np.nanmean(mr):.2f} = (N+1)/2 = {(n_cand + 1) / 2}, rank-1 N=all 0, N=17 0")

    # ---------------- 3. random embedding
    print("3. random (untied) embedding -> 1/N")
    vals = {"full": [], "n17": []}
    for k in range(10):
        r = score_protocol(rng.standard_normal((len(u), D)), u, q, s, "constrained", draws=50)
        vals["full"].append(r["full"]["mean"])
        vals["n17"].append(r["n17"]["mean"])
    check("constrained N=all", abs(np.mean(vals["full"]) - 1 / 20) < 0.012, f"{np.mean(vals['full']):.4f} vs 1/20 = 0.050")
    check("constrained N=17", abs(np.mean(vals["n17"]) - 1 / 17) < 0.012, f"{np.mean(vals['n17']):.4f} vs 1/17 = 0.0588")

    # ---------------- 4. activity only, signature activity: the property the protocol exists for
    print("4. activity-only embedding (signature activity): constrained at chance, unconstrained above")
    res = {"constrained": [], "unconstrained": []}
    for k in range(10):
        script_vec = rng.standard_normal((int(ss.max()) + 1, D))
        emb = noisy(script_vec[ss], rng, 0.3)
        for p in res:
            r = score_protocol(emb, us, qs, ss, p, draws=50)
            res[p].append((r["full"]["mean"], r["full"]["chance"], r["n17"]["mean"], r["full"]["cell_mean"]))
    c = np.mean(res["constrained"], axis=0)
    un = np.mean(res["unconstrained"], axis=0)
    check("constrained N=all at chance", abs(c[0] - c[1]) < 0.01 and abs(c[3] - c[1]) < 0.002, f"{c[0]:.4f} (cell-balanced {c[3]:.6f}) vs chance {c[1]:.4f} (N=40)")
    check("constrained N=17 at chance", abs(c[2] - 1 / 17) < 0.012, f"{c[2]:.4f} vs 1/17")
    check("unconstrained N=all ABOVE chance", un[0] > un[1] + 0.25, f"{un[0]:.4f} vs chance {un[1]:.4f}")
    check("unconstrained N=17 ABOVE chance", un[2] > 1 / 17 + 0.25, f"{un[2]:.4f} vs 1/17")
    # and the other direction of the same fixture: identity added on top survives the constraint
    # (unit-norm activity and identity components of equal size; the activity cue is the larger
    # nuisance here, since every probe's script differs from everything in its gallery)
    unit = script_vec / np.linalg.norm(script_vec, axis=1, keepdims=True)
    emb = noisy(unit[ss] + np.eye(D)[us], rng, 0.05)
    r = score_protocol(emb, us, qs, ss, "constrained", draws=50)
    check("constrained still reads identity", r["n17"]["mean"] > 0.5, f"activity + identity: N=17 {r['n17']['mean']:.3f}")

    # ---------------- 5. activity only on the REAL 48-user script structure
    print("5. activity-only embedding on the real held-out script table (48 users, 216 sequences)")
    K = np.zeros((len(vocab), len(vocab)))
    for sset in others.values():
        for a in sset:
            for b in sset:
                K[a, b] += 1
    K = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))           # cosine co-occurrence over 188 others
    geometries = {"random": lambda: rng.standard_normal((len(vocab), D)),
                  "bundle-aligned": lambda: np.pad(K, ((0, 0), (0, D - len(vocab)))) + 0.2 * rng.standard_normal((len(vocab), D)) / np.sqrt(D)}
    # Two aggregates. POOLED (probe-weighted) is the one the neutrality argument is exact for, so it is
    # asserted tightly. PER-USER (the registered unit, so users can be bootstrapped and paired) reweights
    # probes by 1/(that user's probe count); with a near-deterministic activity argmax per script that
    # moves it off chance by a structure-dependent amount. That amount is MEASURED here and reported as
    # the resolution floor of the activity denial for the registered statistic - not asserted to be 0.
    summary = {}
    for gname, make in geometries.items():
        for noise in (0.05, 0.5):
            for counts in (None, "random"):
                rr = np.random.default_rng(1)
                acc = defaultdict(list)
                for k in range(12):
                    ur, qr, sr, _, _ = table_real(rr, counts)
                    emb = noisy(make()[sr], rng, noise)
                    for p in ("constrained", "fallback_all48"):
                        r = score_protocol(emb, ur, qr, sr, p, draws=40)
                        acc[p].append([r["full"]["mean"] - r["full"]["chance"], r["full"]["pooled"] - r["full"]["pooled_chance"],
                                       r["n17"]["mean"] - 1 / 17, r["n17"]["pooled"] - 1 / 17, r["n17"]["users"],
                                       r["full"]["cell_mean"] - r["full"]["cell_chance"], r["n17"]["cell_mean"] - 1 / 17])
                for p, v in acc.items():
                    v = np.array(v)
                    m, se = v.mean(axis=0), v.std(axis=0, ddof=1) / np.sqrt(len(v))
                    key = (gname, noise, counts or "const", p)
                    summary[key] = (m, se, np.abs(v[:, 2]).max(), np.abs(v[:, 5:7]).max())
                    print(f"     {gname:14s} noise {noise:4.2f} win/seq {str(counts or 'const'):6s} {p:14s} excess over chance: "
                          f"N=all per-user {m[0]:+.4f} pooled {m[1]:+.4f} | N=17 per-user {m[2]:+.4f} (se {se[2]:.4f}, "
                          f"worst single realization {np.abs(v[:, 2]).max():.4f}) pooled {m[3]:+.4f} ({int(m[4])} users) | "
                          f"CELL-BALANCED N=all {m[5]:+.4f} N=17 {m[6]:+.4f} (worst {np.abs(v[:, 5:7]).max():.4f})")
    floor = max(abs(x[0][2]) for (g, n, c, p), x in summary.items() if p == "constrained")
    from nymeria_rank1 import ACTIVITY_FLOOR
    worst = max(x[2] for (g, n, c, p), x in summary.items() if p == "constrained")
    for (gname, noise, counts, p), (m, se, _w, worst_cell) in summary.items():
        if p != "constrained":
            continue
        tag = f"constrained, {gname}, noise {noise}, {counts}"
        # exact in expectation over probe noise; a single realization carries only probe-sampling noise
        check(tag + ": CELL-BALANCED at chance in every realization", worst_cell < 0.01,
              f"worst single-realization excess over 12 realizations, N=all and N=17: {worst_cell:.4f}")
        check(tag + ": pooled at chance", abs(m[1]) < max(0.01, 3 * se[1]) and abs(m[3]) < max(0.01, 3 * se[3]),
              f"pooled excess N=all {m[1]:+.4f}, N=17 {m[3]:+.4f}")
        check(tag + ": per-user within envelope", abs(m[0]) < 0.015 and abs(m[2]) < 0.03,
              f"per-user excess N=all {m[0]:+.4f}, N=17 {m[2]:+.4f}")
    print(f"     -> activity-only envelope on the registered statistic (constrained, per-user, N=17): "
          f"mean |excess| <= {floor:.3f}, worst single realization {worst:.3f}")
    check("ACTIVITY_FLOOR constant matches this measurement", abs(ACTIVITY_FLOOR - round(worst, 3)) <= 0.002,
          f"nymeria_rank1.ACTIVITY_FLOOR {ACTIVITY_FLOOR} vs measured {worst:.3f}")
    for noise in (0.05, 0.5):
        for counts in ("const", "random"):
            m = summary[("bundle-aligned", noise, counts, "fallback_all48")][0]
            mc = summary[("bundle-aligned", noise, counts, "constrained")][0]
            check(f"fallback leaks, bundle-aligned, noise {noise}, {counts}", m[1] > 0.015 and m[1] > abs(mc[1]) + 0.015
                  and m[5] > 0.015 and m[6] > 0.03,
                  f"fallback excess N=all pooled {m[1]:+.4f} cell {m[5]:+.4f}, N=17 cell {m[6]:+.4f}; constrained pooled {mc[1]:+.4f}")

    # unconstrained on the real structure: reported, not asserted (no direction is promised there)
    ur, qr, sr, _, _ = table_real(np.random.default_rng(1))
    for gname, make in geometries.items():
        r = score_protocol(noisy(make()[sr], rng, 0.05), ur, qr, sr, "unconstrained", draws=40)
        print(f"     {gname:15s} unconstrained (reported only): N=all {r['full']['mean']:.4f} (chance {r['full']['chance']:.4f})  N=17 {r['n17']['mean']:.4f}")

    # ---------------- 6. helpers
    print("6. helpers")
    for a, t in ((0.593, 0.103), (0.661, 0.149), (0.539, 0.075)):
        got = implied_rank1(a, 17)
        check(f"implied_rank1({a})", abs(got - t) < 0.0015, f"{got:.4f} vs published {t}")
    check("auc constant", auc_ranked(np.zeros(50), np.zeros(70)) == 0.5, "0.5")
    check("auc perfect", auc_ranked(np.ones(50), np.zeros(70)) == 1.0, "1.0")
    check("auc inverted", auc_ranked(np.zeros(50), np.ones(70)) == 0.0, "0.0")
    emb = rng.standard_normal((len(u), D))
    r1 = score_protocol(emb, u, q, s, "constrained", draws=30, draw_seed=DRAW_SEED)
    r2 = score_protocol(emb, u, q, s, "constrained", draws=30, draw_seed=DRAW_SEED)
    check("draws deterministic", np.array_equal(r1["n17"]["per_user"], r2["n17"]["per_user"], equal_nan=True),
          "same seed -> identical per-user N=17 (the arms pair)")
    # ---------------- 7. read mode end to end, on stand-in runs (plumbing only, no number is meaningful)
    print("7. read mode on stand-in runs: 3 seeds x {treatment strong identity, control none} on the real table")
    import io, json, tempfile, contextlib
    from nymeria_rank1 import read, score_all
    ur, qr, sr, _, _ = table_real(np.random.default_rng(1))
    names = load_heldout()
    tmp = pathlib.Path(tempfile.mkdtemp())
    paths = []
    for seed in (1, 2, 3):
        sv = np.random.default_rng(100 + seed).standard_normal((20, D))
        sv /= np.linalg.norm(sv, axis=1, keepdims=True)
        for arm, w in (("treatment", 0.8), ("control", 0.0)):
            idv = np.random.default_rng(200 + seed).standard_normal((48, D)) / np.sqrt(D)
            emb = noisy(sv[sr] + w * idv[ur] * np.sqrt(D) / 4, np.random.default_rng(300 + seed), 0.3)
            res = {"arm": arm, "seed": seed, "device": "cpu", "protocols": score_all(emb, ur, qr, sr, names)}
            p = tmp / f"nymeria_rank1_{arm}_s{seed}_cpu.json"
            p.write_text(json.dumps(res))
            paths.append(str(p))
    import nymeria_rank1
    nymeria_rank1.READ_OUT = tmp / "read.json"          # never write the stand-in reading into docs/
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = read(paths)
    text = buf.getvalue()
    print("\n".join("     " + l for l in text.splitlines() if l.startswith(("constrained", "  treatment_", "  delta_", "  control_"))))
    check("read runs and scores every registered quantity", rc == 0 and text.count("-> interval") == 3,
          "three registered verdicts printed from six stand-in runs")
    # ---------------- 8. checkpoint-path plumbing: the script join on a stand-in index over real-named dirs
    print("8. index_metadata on a stand-in index (48 held-out names, one CSV per act, as UserProfile orders them)")
    import torch
    from types import SimpleNamespace
    from nymeria_rank1 import index_metadata
    from nymeria_script_pair import scripts_by_sequence
    table = scripts_by_sequence()
    root = pathlib.Path(tempfile.mkdtemp()) / "users"
    uidx, sess_ids, want, w = [], [], [], 0
    for name in names:
        d = root / name
        d.mkdir(parents=True)
        acts = sorted(a for (p, a) in table if p == name)
        for a in acts:
            (d / f"{a}.csv").write_text("x\n")
        files = sorted(f"{a}.csv" for a in acts)               # the order window_scripts assumes
        rows = []
        for si, fname in enumerate(files):
            rows += [w + k for k in range(3)]
            sess_ids += [si] * 3
            want += [table[(name, fname[:-4])]] * 3
            w += 3
        uidx.append(torch.tensor(rows))
    index = SimpleNamespace(user_dirs=[str(root / n) for n in names], user_sample_indices=uidx,
                            window_session_ids=torch.tensor(sess_ids), sample_count=w)
    wu, wq, ws, vocab, got_names = index_metadata(index, table, names)
    check("script join", [vocab[x] for x in ws] == want and got_names == names and len(np.unique(wq)) == 216
          and len(np.unique(wu)) == 48, f"{w} stand-in windows, 216 sequences, every window's script equals the CSV's")
    print("ALL FIXTURES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
