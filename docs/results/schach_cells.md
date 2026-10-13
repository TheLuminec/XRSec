# Ours against Schach et al., per user (users 32-48, N=17)

Both arms on one harness per metric (schach_paired.json). The six uncovered pairs are the application pairs among Superhot, Synth Riders and Social VR, in none of our pretraining corpora: the content-compliant comparison for our ZERO-SHOT arm only (split_audit_2026-10-12.md). C2-lo trained on Across-XR users 0-22 in all five applications, and their model on all five, so on these pairs both of those have seen the content and only zero-shot has not. Descriptive: no band is registered on the six-pair rows.

## D2 (our template metric)

| cells | theirs | zero-shot (ours) | C2-lo (ours) | zero-shot - theirs | C2-lo - theirs | users won (zero-shot / C2-lo) |
| --- | --- | --- | --- | --- | --- | --- |
| all 20 pairs | 0.199 [0.145, 0.258] | 0.234 [0.181, 0.292] | 0.375 [0.321, 0.431] | 0.035 [-0.049, 0.124] | 0.176 [0.092, 0.260] | 10 / 16 of 17 |
| 6 uncovered pairs | 0.173 [0.107, 0.247] | 0.191 [0.136, 0.254] | 0.330 [0.276, 0.386] | 0.018 [-0.085, 0.118] | 0.157 [0.064, 0.248] | 10 / 13 of 17 |

## D1 (their nearest-reference metric)

| cells | theirs | zero-shot (ours) | C2-lo (ours) | zero-shot - theirs | C2-lo - theirs | users won (zero-shot / C2-lo) |
| --- | --- | --- | --- | --- | --- | --- |
| all 20 pairs | 0.180 [0.139, 0.224] | 0.206 [0.174, 0.240] | 0.299 [0.250, 0.353] | 0.025 [-0.030, 0.081] | 0.119 [0.050, 0.191] | 10 / 15 of 17 |
| 6 uncovered pairs | 0.175 [0.128, 0.229] | 0.186 [0.149, 0.226] | 0.266 [0.219, 0.317] | 0.010 [-0.053, 0.070] | 0.090 [0.017, 0.163] | 10 / 12 of 17 |

