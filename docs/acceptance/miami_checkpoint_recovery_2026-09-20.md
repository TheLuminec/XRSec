# Miami checkpoint recovery, and the second crash - 2026-09-20

## The checkpoints are recovered and verified on AVALON

Miami came back at ~15:26 with its data disk intact, and New Gen pushed the durable copy at
`runs/miami-alignment/` to AVALON by rsync at **15:28:26** - **before** the second crash.

| check | result |
| --- | --- |
| run directories | **23 / 23** |
| `.pth` files | **23** |
| `.hydra/config.yaml` | **23** |
| `sha256sum -c MANIFEST.sha256` | **92 / 92 OK**, re-verified independently on AVALON |
| **count assertion before scoring** | **23 of 23 manifest paths present** - no flattening |
| size | 87,195,962 bytes |

**The count assertion is the one that mattered.** 23 paths collapse to 15 basenames, so a flat copy
would have completed silently with 8 missing and every survivor passing its own gate. New Gen
preserved the run-directory structure, so it never arose - but the assertion is what *established*
that rather than assuming it.

**Re-scoring began and 2 of 23 passed before the run was stopped** for the incident:

```
17-24-33  recorded 0.568793  rescored 0.568812  gap 2.0e-05  PASS
18-41-34  recorded 0.580708  rescored 0.580419  gap 2.9e-04  PASS
```

Both inside the 0.002 tolerance. **The remaining 21 are outstanding.**

## TWO GATE DEFECTS THE RECOVERY EXPOSED, BOTH FIXED

Neither would ever have surfaced on the machine that trained the checkpoints - **they are defects
that only appear when a checkpoint moves**, which is exactly what a replication gate is for.

**1. The gate read only THIS machine's shard.** `shard = results/runs/{machine_name()}.jsonl`, so it
assumes the checkpoint was *trained* on the machine running the gate. On AVALON it died with
`FileNotFoundError: results/runs/avalon.jsonl` - a machine that has no shard of its own - rather
than saying what it could not find. Fixed to **search every shard** and name them all in the refusal.

**2. `eval_split` stores ABSOLUTE corpus paths from the training machine.** The recovered
checkpoints carry `/run/media/feng/Data/CalebProject/XRSec/processed_datasets/...`, which resolves
on no other node **even when the corpus is byte-identical**. Fixed by remapping anything at or below
`processed_datasets/` onto the local `ROOT`, **recording each remap in the certificate** rather than
doing it silently, and refusing with a named reason if a path still does not resolve.

**The general form: an artefact can be portable in its bytes and non-portable in its references.**
Both defects passed every check this project has because both were only ever exercised at home.

## The second crash, and what the record does and does not support

**Scoped to what was verified on AVALON, plus New Gen's account of Miami, which this file does not
independently confirm.**

Independently confirmed here: the rsync mtimes are `15:28:26.516` and `15:28:26.588`, all 92 files
verify, and **the two files that are unreadable on Miami are intact on AVALON** (`MANIFEST.sha256`
92 lines, `feng-ms-7b51.results.jsonl` 24 rows). So the replication completed before the crash and
**nothing is lost.**

New Gen's account, reported and not verified from here: three reboots today; boot -2 ended
**abruptly with no shutdown sequence**; `ntfs3` reports MFT damage on exactly the **two files written
at 15:28:24** and nothing else in that tree, consistent with metadata unflushed when the box died
hard. **So the corruption is a consequence of the hard death, not a cause, and it hit only the two
files in flight.**

**What nobody can account for is the memory.** systemd's cgroup accounting shows no user process
above 2.0 GB across a three-day uptime, and **boot -2 left no accounting at all** because it died
hard. New Gen declined to guess and that is the right call. **The user reports a Python process as
the likely cause and a report is pending.**

**A diagnostic direction worth checking rather than asserting:** Miami runs `ollama`, `docker` /
`open-webui` and an agent service. Those are the processes on that box capable of allocating tens of
gigabytes on a single model load, and they are not charged to the user slice a pipeline run would
appear in. Named as somewhere to look, not as a finding.

## The coordinator's failure, recorded as such

**DESKTOP-C measured the memory hazard hours earlier** - a pre-flight index build alone driving a
machine from 13.7 GB free to 0.03 GB, and **process RSS reading 2.5 GB at that instant** because
working sets are trimmed under pressure. It went into CLAUDE.md and to DESKTOP-C. **It never reached
Miami.** No session there had it when that node came back.

**A warning that reaches two nodes and not the third has not been circulated**, and the node it
missed is the one that died. The mechanism is not attention: there was no session on Miami to send
it to, and nothing prompts a coordinator to re-send a standing hazard to a node that appears. **The
fix is that a hazard of this class belongs in `RELAUNCH_KIT.md` step 0, where a returning node reads
it without anyone remembering to send it.**
