# Dr. Feng's `paper_gnn_bilstm` on the Nymeria in-domain arm (registered 2026-10-06, before any run)

## Why

`paper_gnn_bilstm` is the published architecture this project started from. Its graph branches over the
channels are the difference from `bilstm`. The last time it was compared was the pooled-corpus, `raw`,
verification sweep, where three extractors landed within 0.002 of each other. Since then everything has
moved to `bilstm` under `dyn`, and that sweep was run before the AR-glasses result existed. The user asked
for it to be retested on a setup with good results. The comparison point is the Nymeria in-domain treatment:
- constrained (activity-matched) AUC 0.669;
- activity-matched rank-1 0.555 at N=17.

## Design: one change

| | bilstm (exists) | **gnn (new)** |
| --- | --- | --- |
| extractor | `bilstm` | **`paper_gnn_bilstm`**, its defaults |
| everything else | the Nymeria treatment: `dyn` 10 s stride 5, 3,072 training identities, 141 BOXRR dropped, the 1,071 validation users, the same 48 held-out Nymeria users, `identity_softmax` 0.35/30, embedding 128, 120 epochs, patience 15 | identical |
| seeds | 1 / 2 / 3, runs 2026-09-21 09-16-23 / 12-08-08 / 14-57-30 | 1 / 2 / 3 |

The configs come from `treatment_short_lists.py --seed s --sample-time 10 --extractor paper_gnn_bilstm`. All
three were checked line by line against `nymeria_in_domain_treatment_s{s}.yaml`: identical except
`experiment_name` and `extractor`. The digests are listed in COORDINATION (2026-10-06).

**Scoring.** Everything runs on cuda, with each checkpoint gated first, so every comparison is on the same
device. The bilstm treatment checkpoints are re-scored on cuda into the same file, so the two arms pair on
one device and one harness.
- `nymeria_script_pair.py` with `SCRIPT_PAIR_OUT=docs/acceptance/nymeria_gnn_script_pair.json`, on both arms.
  This gives the constrained AUC on identical pairs per seed.
- `nymeria_rank1.py score`, arm `gnn`. Its pairing partner is the committed bilstm files
  `nymeria_rank1_treatment_s{n}_cuda.json`.
- Reading: `nymeria_gnn_read.py`, fixture-tested (`nymeria_gnn_read_fixture.py`) and committed with this file.

## Registered quantities

Both are partitioned, with every region named, and scored by where the interval falls.

**1. Constrained AUC, gnn - bilstm, per-seed paired, t interval over 3 seeds**

| region | meaning |
| --- | --- |
| below -0.02 | GNN WORSE: the published graph branches cost the motion signature on AR glasses |
| **-0.02 to +0.02** | **BAND (predicted)**: architecture is worth ~0 here too, as on the pooled corpus |
| above +0.02 | GNN BETTER: the graph branches add to the motion signature on AR glasses |

**2. Rank-1, constrained protocol, N=17, cell-balanced, gnn - bilstm**
- Computed per cell, each arm averaged over seeds, with a user bootstrap.

| region | meaning |
| --- | --- |
| below -0.03 | GNN WORSE on identification |
| **-0.03 to +0.03** | **BAND (predicted)**: no identification difference |
| above +0.03 | GNN BETTER on identification |

**Resolution, stated before running, because it decides how a null reads.**
- The bilstm treatment's seed sd on the constrained AUC is about 0.009. If the two architectures' seed noise
  is independent, the paired sd is about 0.012, and three seeds give a t half-width of about
  4.30 × 0.012 / √3 ≈ **±0.03**. That is wider than the ±0.02 band. So an interval spanning the band and a
  neighbouring region is the most likely outcome, and it reads "not resolved", never "no difference".
- The rank-1 user bootstrap is tighter, about ±0.03, but it does not include seed variance.
- **The strong outcome is either outer region.** An interval wholly inside the band is unlikely at this n
  and would be strong in its own right.

**Reported, not registered:**
- the row AUC (each run's `selected_test_auc`);
- each arm's levels;
- convergence (`best_epoch`, `epochs_run`).

The bilstm treatment ran out its 120-epoch cap without stopping on patience. If the GNN stops on patience
and bilstm did not, that is a convergence difference, and it is reported beside the delta.

**What else would produce each outcome.**
- "GNN worse" is also produced by a budget the GNN needs more of, if it reaches the cap still improving.
  The convergence check separates the two.
- "GNN better" has no obvious second explanation inside this design. Same data, same users and same pairs,
  only the extractor differs.

## Run acceptance (Miami)

- **Every launch is under `gated_launch.sh`, wrapped in `avm run` so it shows on Avalon Monitor.** If `avm`
  or its config is absent on Miami, the session reports that and does not install or configure it. The
  config carries a host token, which only the user enters.
- **A one-epoch timing pilot runs first.** It is named `experiment_name=gnn_pilot` so its shard row is
  never mistaken for an arm. It reports s/epoch and `peak_mb`, with no AUC read from it. If it projects more
  than 12 h per seed, the session reports and waits.
- Push each result file to `origin/miami-server` as it lands. A gate refusal is reported, never re-scored on
  another device.

## Amendment 1, 2026-10-06: an instrument fact, found by the harness refusing; no measurement is involved

The script-pair harness filtered shard rows to `experiment == "nymeria_in_domain"`. The GNN arm's rows carry
`treatment_10s_gnn`, the generator's arm name, so seed 1's script-pair found no row and asserted (rc 1, no
output). This could have been known by reading the harness, so the amendment is legitimate.

The fix accepts `treatment_10s_gnn` as well. `gnn_pilot` rows stay excluded. Rows are still matched to a
checkpoint by its own path, and the gate is unchanged. The file is under `docs/acceptance`, so
`code_identity` does not move.

Nothing was read from the refused run. Rank-1 seed 1 was unaffected; it gated at 9.6e-8 on cuda.

**Run order:**
- Miami does not pull while the chain is live.
- When the chain ends, it pulls and runs the per-seed script-pairs and the final six-checkpoint pass, with
  the cross-check against the per-seed files as agreed.

## RESULT, 2026-10-06: the GNN is decisively WORSE on both registered measures. The prediction (BAND) failed

**Provenance.**
- Three seeds per arm, every gate passed on cuda:
  - rank-1 9.6e-8 / 2.0e-8 / 5.3e-8;
  - script-pair GNN 2.7e-7 / 9.3e-8 / 3.3e-7, bilstm 9.9e-9 / 0 / 3.3e-9.
- The files were propagated to main before reading.
- The read output is `nymeria_gnn_read.json`, committed (c26801d) before this section was written.

| measure | bilstm (s1 / s2 / s3) | paper_gnn_bilstm (s1 / s2 / s3) | gnn - bilstm | region |
| --- | --- | --- | --- | --- |
| **constrained AUC** (registered) | 0.661 / 0.678 / 0.665 | **0.520 / 0.528 / 0.518** | **-0.146 [-0.159, -0.133]** | **GNN WORSE**, whole interval |
| **rank-1 N=17, constrained, cell-balanced** (registered) | 0.555 [0.507, 0.600] | **0.309 [0.259, 0.360]** | **-0.246 [-0.279, -0.216]** | **GNN WORSE**, whole interval |
| row AUC (`selected_test_auc`, descriptive) | 0.708 / 0.726 / 0.718 | 0.577 / 0.586 / 0.580 | -0.137 [-0.147, -0.126] | |
| unconstrained AUC on the same embeddings (descriptive) | 0.712 / 0.721 / 0.712 | 0.576 / 0.585 / 0.575 | | |
| best_epoch / epochs_run | 118 / 120, 120 / 120, 118 / 120 | 120 / 120, 117 / 120, 115 / 120 | | |

**Reading:**
- Both registered intervals lie wholly in "GNN WORSE", far beyond the resolution stated before running
  (±0.03). The prediction that architecture is worth about 0 here (BAND) **failed**, and it is reported
  as failed.
- On the same data, users and pairs, Dr. Feng's published architecture retains little of the
  activity-matched motion signature: constrained AUC 0.52 against chance 0.50.
- It still identifies well above chance (0.309 at N=17, chance 0.059) and above the zero-shot bilstm
  control (0.172). But it sits 0.25 below `bilstm`.

**Convergence check (registered).** Neither arm stopped on patience. Both ran to the 120 cap, with best
epochs of 115-120. So there is no categorical convergence difference. Both arms are censored alike. "The
GNN needs a longer budget" is therefore **not excluded**, but nothing here points to it: two of three GNN
seeds peaked before the cap, against one of three bilstm seeds.

**Scope:**
- This holds under `dyn`, 10 s, `identity_softmax`, 3,072 identities, on Nymeria (AR glasses), in domain.
- It does **not** contradict the earlier pooled-corpus result (three extractors within 0.002). That was
  `raw`, verification, on a different corpus. The two together say the extractor is ~irrelevant there and
  matters a great deal here.
- **Why the graph branches cost so much under `dyn` is not established.**
  - One untested candidate: the fixed 10-node graph was designed around absolute channels
    (orientation / position / root aggregate nodes), and `dyn` removes exactly those static components.
  - That is a hypothesis for a registered follow-up, not a finding.

**Instrument note: the GNN is not bit-reproducible on cuda.**
- Scoring the same GNN checkpoint twice on the same device and the same pairs gave constrained AUCs
  differing by 2e-8 to 1.4e-7.
- The bilstm entries reproduced exactly.
- The likely cause is non-deterministic atomic scatter in the graph layers. That is unverified.
- At 1e-7 it is five orders below every band and below every gate tolerance. Re-scored GNN figures should
  be expected to agree to about 1e-7, not to the bit.
