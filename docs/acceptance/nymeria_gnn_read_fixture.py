"""Fixture for nymeria_gnn_read.py, built from the committed bilstm treatment rank-1 files with answers known by
construction. Run before any GNN number exists:  python docs/acceptance/nymeria_gnn_read_fixture.py"""
import copy, json, pathlib, subprocess, sys, tempfile
A = pathlib.Path(__file__).resolve().parent
tmp = pathlib.Path(tempfile.mkdtemp())
r1 = []
for s in (1, 2, 3):
    t = json.loads((A / f"nymeria_rank1_treatment_s{s}_cuda.json").read_text())
    g = copy.deepcopy(t); g["arm"] = "gnn"
    (tmp / f"t{s}.json").write_text(json.dumps(t)); (tmp / f"g{s}.json").write_text(json.dumps(g))
    r1 += [str(tmp / f"t{s}.json"), str(tmp / f"g{s}.json")]
sp = {}
for s, (row, con) in zip((1, 2, 3), ((0.70, 0.66), (0.72, 0.68), (0.71, 0.67))):
    base = {"seed": s, "device": "cuda", "gate_passed": True, "gate_gap": 0.0, "constrained_pairs": [100, 100]}
    sp[f"b{s}"] = {**base, "arm": "treatment", "recorded": row, "constrained_auc": con, "checkpoint": f"x/b{s}"}
    sp[f"g{s}"] = {**base, "arm": "treatment_paper_gnn_bilstm", "recorded": row + 0.05, "constrained_auc": con + 0.05 + 0.001 * (s - 2),
                   "checkpoint": f"x/g{s}"}
sp["c1"] = {**sp["b1"], "arm": "control", "checkpoint": "x/c1"}        # a control entry must be ignored
(tmp / "sp.json").write_text(json.dumps(sp))
out = tmp / "read.json"
p = subprocess.run([sys.executable, str(A / "nymeria_gnn_read.py"), "--script-pair", str(tmp / "sp.json"), "--rank1", *r1,
                    "--out", str(out)], capture_output=True, text=True)
print(p.stdout, p.stderr); assert p.returncode == 0
o = json.loads(out.read_text())
c = o["rows"]["constrained_auc_gnn-bilstm"]; r = o["rows"]["rank1_n17_gnn-bilstm"]
assert abs(c["mean"] - 0.05) < 1e-12 and 0.0 < c["lo"] < 0.05 < c["hi"], c
assert "GNN BETTER" in o["verdicts"]["constrained_auc_gnn-bilstm"], o["verdicts"]
assert r["mean"] == 0 and r["lo"] == 0 and r["hi"] == 0 and r["cells"] == 115 and r["users"] == 46, r
assert "BAND" in o["verdicts"]["rank1_n17_gnn-bilstm"]
assert abs(o["levels"]["treatment"]["rank1_n17"]["mean"] - 0.555) < 0.001, o["levels"]["treatment"]["rank1_n17"]
print("FIXTURE PASSES: identical rank-1 arms read 0 with a zero interval on 115 cells / 46 users; a +0.05 constrained "
      "shift reads +0.05 and BETTER; the bilstm level reproduces the recorded 0.555; a control entry is ignored")
