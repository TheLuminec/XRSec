"""
The gate for the step 6 seated dyn re-run, borrowed rather than rewritten.

The Coordinator's instruction was to take the shape of score_nymeria.py's gate rather than
write a fresh scorer, and that is the point: a gate I re-implement can share a bug with the
harness it is supposed to check, while the one that passed 28/29 dyn checkpoints is an
independent object. So this imports gate() and only chooses the checkpoints and the output
path - it does not touch docs/acceptance/nymeria_gate.json, which is Generalisation's record.
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
from score_nymeria import gate, GATE_DEVICE

CKPTS = {  # seed -> the 9.3 dyn checkpoint, trained on BOXRR + alyx only
    1: "sweeps/cb0a7dd722/runs/bilstm_a41190094c/best.pth",
    2: "sweeps/cb0a7dd722/runs/bilstm_0ebccac678/best.pth",
    3: "sweeps/cb0a7dd722/runs/bilstm_ab82c3b90b/best.pth",
    4: "sweeps/cb0a7dd722/runs/bilstm_796d3932d4/best.pth",
    5: "sweeps/cb0a7dd722/runs/bilstm_8d679a46cc/best.pth",
}
OUT = pathlib.Path("docs/acceptance/step6_seated_dyn_gate.json")

if __name__ == "__main__":
    print(f"gate device {GATE_DEVICE}")
    records = [dict(gate(p), seed=s) for s, p in sorted(CKPTS.items())]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
    passed = sum(r["passed"] for r in records)
    print(f"\ngate: {passed}/{len(records)} passed -> {OUT}")
    if passed != len(records):
        print("*** STOP - report the mismatch, do not compute rank-1 and do not tune toward it ***")
        sys.exit(1)
