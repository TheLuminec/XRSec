# The `dyn` index build costs ~4x `raw` in memory — measured on Miami, 2026-09-21

Why this file exists: seed 1 of the Nymeria in-domain arm's **treatment** was OOM-killed by the
gated launcher's cap (`rc=137 oom_kill=1 peak_mb=32768`, 56 s in, during the index build) while the
**control** completed at `peak_mb=30752` under the same 32 GB cap — 94% of it. The first explanation
offered (mine) was that the peak was mostly page cache and therefore harmless, since cache is
reclaimed before the cgroup OOM killer fires. **That was wrong, and measuring it is what settled it.**

Sampled from the job's own cgroup (`memory.stat`, 2 s interval) during a rebuild, Nymeria-only,
10 s @ 20 Hz, `window_stride=5`, 242,919 windows, shape (242919, 7, 200):

| build | max `anon` | max `file` (cache) | launcher `peak_mb` |
| --- | --- | --- | --- |
| `encoding=dyn` | **12,219 MB** | 1,302 MB | 14,665 |
| `encoding=raw` (identical otherwise) | **3,004 MB** | 0 MB | 4,076 |

**About 90% of the peak is anonymous, not cache**, and it is ~10x the 1.27 GiB tensor the build
produces. The two rows differ in one config key and produce the same window count and the same
shape, so **the ~4x is the `dyn` path**, not the corpus, the cache or the loader in general.

**Consequence for the arm.** Nymeria holds ~1,029 windows per identity against the pooled ~169, so
the treatment adds ~121k windows (+23%) over the control's 519,211. Against a control that already
peaked at 94% of a 32 GB cap, it does not fit. Fitting it would need ~38 GB of a 45.7 GB machine,
leaving ~4 GB for everything else — **the 100%-RAM condition the user forbade after the
2026-09-20 crash. The cap was not raised: shrinking a guard to fit a job is the guard's own failure
mode.** The fix belongs in the `dyn` build (AVALON), with the control re-run afterwards under the
new `code_identity` and required to reproduce `selected_test_auc` 0.5415 to same-device precision —
that reproduction, not the diff, is what proves the memory change touched no numerics.

**Method note.** The 4x was localised by changing exactly one key between two otherwise identical
builds. The `peak_mb` a marker records includes page cache and is therefore an upper bound on a
job's real demand; when the distinction matters, sample `memory.stat` rather than quoting the peak.
