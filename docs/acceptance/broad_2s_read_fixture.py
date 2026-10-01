"""Fixture for broad_2s_read.py: synthetic arms built from the committed treatment_2s results, with answers
known by construction. Run before any broad 2 s result exists:  python docs/acceptance/broad_2s_read_fixture.py"""
import copy, json, pathlib, subprocess, sys, tempfile
A = pathlib.Path(__file__).resolve().parent
tmp = pathlib.Path(tempfile.mkdtemp())
treat = [json.loads((A / f"ballthrowing_cross_day_s{s}.json").read_text()) for s in (1, 2, 3)]
def write(name, fn, drop_seed=None):
    res = []
    for d in treat:
        for r in d["results"]:
            r = copy.deepcopy(r)
            if r["seed"] == drop_seed:
                res.append({"seed": r["seed"], "refused": "fixture refusal"}); continue
            fn(r); res.append(r)
    p = tmp / f"{name}.json"; p.write_text(json.dumps({"results": res})); return p
ctrl = write("control", lambda r: r["C1"].update({u: v - 0.05 for u, v in r["C1"].items()}), drop_seed=3)
raw = write("raw", lambda r: None)
br = write("br", lambda r: r["C2"].update({u: v * 1.15 for u, v in r["C2"].items()}))
al = json.loads((A / "alyx_cross_day_gpu.json").read_text())
al2 = {"results": [dict(copy.deepcopy(r), arm="treatment_2s") for r in al["results"] if r["arm"] == "treatment"]}
(tmp / "alyx2.json").write_text(json.dumps(al2))
out = tmp / "read.json"
cmd = [sys.executable, str(A / "broad_2s_read.py"), "--bt"] + [f"treatment={A}/ballthrowing_cross_day_s{s}.json" for s in (1, 2, 3)] \
      + [f"control={ctrl}", f"raw={raw}", f"br={br}", "--alyx", str(A / "alyx_cross_day_gpu.json"), str(tmp / "alyx2.json"), "--out", str(out)]
print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
o = json.loads(out.read_text()); rows = o["rows"]
close = lambda x, y, t=1e-9: abs(x - y) < t
assert rows["nym_C1"]["seeds"] == [1, 2] and close(rows["nym_C1"]["mean"], 0.05) and close(rows["nym_C1"]["lo"], 0.05, 1e-6)
assert close(rows["nym_persistence"]["mean"], 0.05)
assert rows["raw_C1"]["mean"] == 0 and rows["raw_headset"]["mean"] == 0 and rows["raw_headset"]["lo"] == 0
assert close(rows["br_rho"]["rho_br"], 1.15 * rows["br_rho"]["rho_dyn"]) and close(rows["br_rho"]["mean"], 0.15 * rows["br_rho"]["rho_dyn"])
assert close(rows["br_rho"]["rho_dyn"], 0.4580623306233062 / 0.6932249322493226)        # the recorded levels
assert close(rows["alyx_rho_2s"]["mean"], 0.4824631386321353 / 0.7430269942708708) and rows["alyx_rho_2s"]["units"] == 43
assert o["descriptive"]["alyx_cost_2s-10s_paired"]["mean"] == 0
assert ("control", 3, "fixture refusal") in [tuple(x) for x in o["refused"]]
assert "BAND" in o["verdicts"]["nym_C1"] and "BAND" in o["verdicts"]["alyx_rho_2s"]
print("FIXTURE PASSES: every row reproduces its constructed answer; refusal excluded and listed; seeds paired")
