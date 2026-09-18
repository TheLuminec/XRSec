# Relaunch kit — bringing up a GPU node and restarting Rack seed 1

Written 2026-09-17 on AVALON, after the Miami node was lost with its data. The purpose is that a
replacement node spends its first hour **running** rather than being provisioned.

**What was lost:** Rack 2023 seed 1 — ~36 h of GPU, its checkpoints and its offline run log.
**What was not lost:** everything expensive except the GPU time. The protocol, the exact-match check
against the paper's Table IV, the environment pins, and Amendments 1–7 (the pandas regression, the
infeasible shipped configuration, the `| tee` false pass, the never-armed watcher) are all in
`docs/acceptance/sota_rack2023_reproduction_REGISTERED.md`, which is on `origin`. **A re-run costs
GPU time and none of the diagnosis.**

---

## 0. Before anything else — two things that are not optional

**(a) Replicate the checkpoints.** The **23** gated programme checkpoints are on **one machine** and
not on AVALON (verified 2026-09-17: every `runs/2026-09-10/*` and `runs/2026-09-11/*` directory is
absent here). They are a few tens of MB in total — `bilstm` at `embedding_dim=128` is ~153k
parameters. Until they are copied, **no Questset arm can run anywhere but that node**, and a second
disk failure costs not a run but the ability to re-score anything for a reviewer.

**Copy against the manifest, not against a remembered count.** This kit said *ten* until 2026-09-17;
the certificates say **23**. `docs/acceptance/checkpoint_replication_manifest.json` enumerates all of
them with each one's `recorded` figure. **Acceptance is not that the files arrived** - it is that each
copy reproduces its `recorded` figure on the receiving machine within the 0.002 the certificate was
gated at. AVALON can run that check on CPU, since it holds every corpus involved, which keeps the
verification independent of the sending node. This is the same
exposure that just cost 36 hours. Copy the `checkpoints/` directory **and** each run's
`.hydra/config.yaml`, so the split, encoding and seed travel with the weights.

**(b) Decide what happens to the dead node's disks before they move.** It held **BOXRR-derived
data** (its clause-15 inventory is committed at `docs/acceptance/boxrr_inventory_feng-ms-7b51.json`).
The DUA forbids further distribution, so **handing a disk to an external recovery vendor is
plausibly a distribution event**. In-house recovery at the same institution is covered by the
existing ruling; an outside vendor, an RMA, or disposal is **the user's call to make, not a
technical detail** — and it needs deciding before the hardware moves, not after.

---

## 1. Provision the node

Pins are a **correctness** matter here, not convenience — the authors pinned nothing, so the
environment is a free variable and two of the three bugs in the amendments are dependency-version
bugs. From the registration:

| package | pin | why |
| --- | --- | --- |
| Python | **3.8.20** | their code runs unmodified on it; **3.14 is a hard blocker** for our own pipeline (Hydra 1.3.6 dies in argparse), use **3.13** for ours |
| pytorch-lightning | **1.9.5** | last 1.x — their config passes `gpus=` and `auto_scale_batch_size=`, both removed in PL 2.0 |
| pytorch-metric-learning | **1.7.3** (last 1.x) | 2.x renamed `embeddings_come_from_same_source` → `ref_includes_query` **and reordered the positional arguments**, so renaming the keyword alone silently passes reference embeddings as query labels and returns a plausible number. This crashed the 09-11 run at minute 22 |
| pandas | see Amendment 3 | 1.x consolidated on `.values` access and 2.x does not — a 525× per-item cost and a silently discarded write. Read the amendment before choosing |
| torch | 2.0.1+cu118 was used | record `torch.cuda.get_arch_list()` **and** the device capability tuple, not just the device name — the pair says whether kernels ran native or through the compatibility path |

**Two environment traps for our own pipeline, both already paid for:** `torch_geometric` is a hard
dependency of *every* run (the extractor package auto-imports, so a `bilstm`-only job dies naming a
GNN library), and a green test suite is not evidence the node can run anything — 475 tests passed on
a node where nothing could run.

## 2. Rebuild the input

The Rack reproduction needs `15_fps-63_subjects-metric_learning_movement.hdf5`, built from
who_is_alyx by their `01_aggregate.py`, which selects players with exactly 2 sessions — taking
AVALON's 76 players to **63**, matching their config filename to the digit. AVALON holds the
who_is_alyx corpus (76 users), so this is reproducible without re-downloading anything.

**Prefer copying an already-gated corpus to reconverting one.** `code_identity` covers `model/*.py`
and **not** the `prepare_*.py` converters, so two machines can hold materially different
`processed_datasets/` while every row reports the same identity.

## 3. Relaunch

```bash
git clone <origin> && cd XRSec
git checkout main            # the registration and all amendments are here
sed -n '1,130p' docs/acceptance/sota_rack2023_reproduction_REGISTERED.md   # read the protocol first
```

Apply the source change from **Amendment 2/3** (the narrow one at `window_dataset.py:23`, *not*
`base_dataset.py:221`, which would break `bin_maker.py`'s `rolling(...)`), then launch at
`seed=42`, `max_epochs=100`. Measured cost: **~21.5 min/epoch including validation, ~36 h/seed**.

**Seeds 2–3 stay parked** until seed 1 reproduces the published 99 / 89 / 25 curve. That staging is
what kept a broken port from costing three more days, and it worked.

## 4. The watcher — arm it, then prove it is armed

Amendment 7 said "watcher armed" and **the command had never been run**; `pgrep` matched nothing.
That is why seed 1's outcome was nearly missed, and it is the fourth member of a defect class this
project tracks: a guard whose existence is asserted in the artefact meant to prove it.

- Arm it on **both** the done and `.failed` markers — a runner heartbeat answers "is the runner
  alive", not "did my job succeed", and an empty queue is not success.
- **Verify by inspecting the process and record the pid.** For anything that must be *live* rather
  than merely done, the artefact is a process, not a file.
- Have it **push a notification**, not only write a file. Seed 1's outcome sat unread for two hours
  because the watcher wrote to disk and nothing told anyone.
- Never end a gate or run in `| tee` — a pipeline's exit status is the last command's, so `tee`
  masks a python crash. Use a redirect; verified both ways (redirect returns 1, pipe returns 0).

## 5. The standing rule this cost us — apply it from the first epoch

**Commit the result artefact the moment it exists, not when the analysis around it is finished.**
Seed 1's registration was committed *before* the run and nothing during or after it, so the old rule
was satisfied and the run was still lost. The curve was decoded, read, interpreted and argued across
four messages while living on exactly one disk.

Concretely, when the watcher fires: write a one-line JSON with the argmax, the window means and the
checkpoint epochs, and **push it before reading it**. It costs a minute. The reasoning is the
expensive part and it is entirely re-doable; the numbers are cheap to save and impossible to
recreate without the GPU.

## 6. What to run first, in order

1. **Validation gate** — `eval_harness --split validation --gate`, reproducing the recorded 5-min
   value at the selected checkpoint. The lost run's referent was `0.9227739722096133` at `epoch_052`;
   a fresh run has its own. Gate first, quote nothing before it passes.
2. **Test-split sweep** against the published **99 / 89 / 25** curve. This is the comparison that
   decides seeds 2–3, and **nothing was ever compared against it** — the lost run stopped one step
   short.
3. **Questset arms A1/A2/A3** — but note these need **no GPU** and should run on AVALON's CPU as
   soon as the checkpoints are replicated, rather than queueing behind Rack.
   `docs/acceptance/questset_harness.py --gate` already passes here (12/12 cells exact), so the
   plumbing is validated and waiting on weights.

## 7. Two readings from the lost run, recorded as leads and **not** as results

Both are transcribed second-hand from the dead node's messages (SALVAGE RECORD in the registration)
and neither is gated:

- The 5-min validation curve was a **plateau, not a spike** — flat at ~0.904–0.910 from epoch 40,
  with the argmax at epoch 052 and epoch 091 a tie within the band. **So the 100-epoch cap was not
  binding**; this is the durable half, because it is about shape rather than level.
- **Do not quote 0.9228 as the level.** It is the maximum over ~100 noisy points on a flat curve —
  about +0.015 of free selection inflation. The supportable figure is the plateau, **~0.905**, on
  **9 validation subjects** under the authors' hardcoded `[::150]` enrolment, which is **not** the
  paper's test figure.

If the disk is recovered, the artefacts supersede both readings entirely.
