"""Fixture for alyx_axr_overlap_read.py from a committed P3 result file (it carries per-cell, per-user A1).
Identical arms must read exactly 0; a +0.10 shift on every cell must read +0.10 and land above the band.
    python docs/acceptance/alyx_axr_overlap_read_fixture.py"""
import copy, json, pathlib, subprocess, sys, tempfile
A = pathlib.Path(__file__).resolve().parent
src = json.loads((A / "across_xr_alignment_p3_superhot_vr.json").read_text())
assert "per_cell_user" in src["seeds"][0]["arms"]["A1"]
tmp = pathlib.Path(tempfile.mkdtemp())
for arm, shift in (("zeroshot", 0.0), ("c2lo", 0.10)):
    for s in (1, 2, 3):
        (tmp / f"across_xr_alignment_{arm}_rescored_seed{s}.json").write_text(json.dumps(src))
        sh = copy.deepcopy(src)
        for c, v in sh["seeds"][0]["arms"]["A1"]["per_cell_user"].items():
            sh["seeds"][0]["arms"]["A1"]["per_cell_user"][c] = [x + shift for x in v]
        (tmp / f"across_xr_alignment_{arm}_noalyx_seed{s}.json").write_text(json.dumps(sh))
out = tmp / "read.json"
p = subprocess.run([sys.executable, str(A / "alyx_axr_overlap_read.py"), "--dir", str(tmp), "--out", str(out)], capture_output=True, text=True)
print(p.stdout, p.stderr); assert p.returncode == 0
o = json.loads(out.read_text())
z, c = o["rows"]["zeroshot_noalyx-withalyx"], o["rows"]["c2lo_noalyx-withalyx"]
assert z["mean"] == 0 and z["lo"] == 0 and z["hi"] == 0 and "BAND" in o["verdicts"]["zeroshot_noalyx-withalyx"], z
assert abs(c["mean"] - 0.10) < 1e-9 and "HELPS" in o["verdicts"]["c2lo_noalyx-withalyx"], c
print("FIXTURE PASSES: identical arms read 0 (BAND); a +0.10 shift reads +0.10 (above the band)")
