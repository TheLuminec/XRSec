"""
Ran once, in place, on both AVALON and (to be run) DESKTOP-C's copy of
Nymeria_Dataset -- applies the device-frame fix (fix_device_frame(),
shipped in 1f92a4a) to the quaternion columns of every already-converted
sequence CSV, leaving position untouched.

Not part of the normal conversion pipeline (prepare_nymeria.py already
applies both fixes for anything converted from raw after 1f92a4a) -- this
script exists only to bring pre-fix output up to date without a re-fetch,
and is kept here rather than deleted so the exact steps that produced the
current file contents are reproducible/auditable, matching what
PROVENANCE.md's "reconverted in place on 2026-09-04" line refers to.

    python reconvert_nymeria_device_fix.py processed_datasets/Nymeria_Dataset/users

Then, to verify both copies converged to the same result:

    find processed_datasets/Nymeria_Dataset/users -name 'act*.csv' | sort | xargs sha256sum
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_nymeria as pn


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: reconvert_nymeria_device_fix.py <users_dir>")
        return 1

    users_dir = Path(sys.argv[1])
    csv_files = sorted(users_dir.glob("*/act*.csv"))
    print(f"reconverting quaternion for {len(csv_files)} files")

    for path in csv_files:
        df = pd.read_csv(path)
        q_w = df[["UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w"]].to_numpy()
        q_final = pn.fix_device_frame(q_w)
        q_final /= np.linalg.norm(q_final, axis=1, keepdims=True)
        df["UnitQuaternion.x"] = q_final[:, 0]
        df["UnitQuaternion.y"] = q_final[:, 1]
        df["UnitQuaternion.z"] = q_final[:, 2]
        df["UnitQuaternion.w"] = q_final[:, 3]
        # Column order preserved as read (pandas keeps the original CSV's
        # header order: SessionTime, UnitQuaternion.x/y/z/w, HmdPosition.x/y/z).
        # float_format not set -- pandas.to_csv's default uses the shortest
        # round-tripping decimal repr of each float64, same as who-is-alyx's
        # and BOXRR's converters use.
        df.to_csv(path, index=False)

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
