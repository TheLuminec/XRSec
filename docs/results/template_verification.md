# Template verification (template_verification_PROTOCOL.md)

| corpus | arm | seed | condition | users | EER | FAR / FRR at cross-fitted EER threshold | balanced accuracy | TAR at FAR 1 % (realised FAR) | per-user FRR p10 / p50 / p90 | plumbing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alyx | treatment | 1 | cross_day | 14 | 0.260 | 0.265 / 0.268 | 0.734 | 0.155 (0.013) | 0.03 / 0.08 / 0.82 | 0.0 |
| alyx | treatment | 1 | same_day | 14 | 0.125 | 0.132 / 0.125 | 0.872 | 0.372 (0.013) | 0.05 / 0.13 / 0.17 | 0.0 |
| alyx | treatment | 2 | cross_day | 17 | 0.201 | 0.202 / 0.201 | 0.798 | 0.169 (0.012) | 0.05 / 0.13 / 0.45 | 0.0 |
| alyx | treatment | 2 | same_day | 17 | 0.135 | 0.136 / 0.134 | 0.865 | 0.350 (0.013) | 0.03 / 0.10 / 0.29 | 0.0 |
| alyx | treatment | 3 | cross_day | 12 | 0.182 | 0.186 / 0.184 | 0.815 | 0.196 (0.014) | 0.06 / 0.10 / 0.45 | 0.0 |
| alyx | treatment | 3 | same_day | 12 | 0.114 | 0.116 / 0.118 | 0.883 | 0.435 (0.013) | 0.04 / 0.11 / 0.23 | 0.0 |
| axr (cpu) | c2lo | 1 | cross_application | 17 | 0.270 | 0.269 / 0.269 | 0.731 | 0.114 (0.012) | 0.16 / 0.30 / 0.36 | 0.0 |
| axr (cpu) | c2lo | 1 | within_application | 17 | 0.175 | 0.176 / 0.176 | 0.824 | 0.274 (0.012) | 0.09 / 0.17 / 0.26 | 0.0 |
| axr | c2lo | 1 | REFUSED: plumbing: rank-1 differs from committed by 2.74e-03 | | | | | | | |
| axr (cpu) | c2lo | 2 | cross_application | 17 | 0.270 | 0.269 / 0.270 | 0.730 | 0.143 (0.011) | 0.16 / 0.24 / 0.44 | 0.0 |
| axr (cpu) | c2lo | 2 | within_application | 17 | 0.184 | 0.184 / 0.184 | 0.816 | 0.302 (0.012) | 0.10 / 0.19 / 0.25 | 0.0 |
| axr | c2lo | 2 | REFUSED: plumbing: rank-1 differs from committed by 3.00e-03 | | | | | | | |
| axr (cpu) | c2lo | 3 | cross_application | 17 | 0.259 | 0.258 / 0.260 | 0.741 | 0.135 (0.011) | 0.17 / 0.24 / 0.41 | 0.0 |
| axr (cpu) | c2lo | 3 | within_application | 17 | 0.178 | 0.178 / 0.178 | 0.822 | 0.279 (0.011) | 0.09 / 0.19 / 0.23 | 0.0 |
| axr | c2lo | 3 | REFUSED: plumbing: rank-1 differs from committed by 2.50e-03 | | | | | | | |
| axr (cpu) | zeroshot | 1 | cross_application | 17 | 0.406 | 0.407 / 0.408 | 0.592 | 0.058 (0.013) | 0.15 / 0.42 / 0.66 | 0.0 |
| axr (cpu) | zeroshot | 1 | within_application | 17 | 0.313 | 0.320 / 0.315 | 0.682 | 0.127 (0.013) | 0.12 / 0.29 / 0.55 | 0.0 |
| axr | zeroshot | 1 | REFUSED: plumbing: rank-1 differs from committed by 2.43e-03 | | | | | | | |
| axr (cpu) | zeroshot | 2 | cross_application | 17 | 0.405 | 0.406 / 0.406 | 0.594 | 0.057 (0.014) | 0.18 / 0.47 / 0.66 | 0.0 |
| axr (cpu) | zeroshot | 2 | within_application | 17 | 0.307 | 0.314 / 0.310 | 0.688 | 0.126 (0.015) | 0.15 / 0.29 / 0.49 | 0.0 |
| axr | zeroshot | 2 | REFUSED: plumbing: rank-1 differs from committed by 1.88e-03 | | | | | | | |
| axr (cpu) | zeroshot | 3 | cross_application | 17 | 0.402 | 0.402 / 0.404 | 0.597 | 0.056 (0.013) | 0.16 / 0.37 / 0.62 | 0.0 |
| axr (cpu) | zeroshot | 3 | within_application | 17 | 0.296 | 0.303 / 0.299 | 0.699 | 0.126 (0.013) | 0.12 / 0.26 / 0.52 | 0.0 |
| axr | zeroshot | 3 | REFUSED: plumbing: rank-1 differs from committed by 2.32e-03 | | | | | | | |
| ballthrowing | treatment_2s | 1 | C0_same_session | 41 | 0.122 | 0.126 / 0.124 | 0.875 | 0.613 (0.011) | 0.03 / 0.08 / 0.30 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 1 | C1_same_headset_other_day | 41 | 0.179 | 0.183 / 0.180 | 0.819 | 0.450 (0.011) | 0.04 / 0.16 / 0.35 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 1 | C2_other_headset | 41 | 0.262 | 0.268 / 0.263 | 0.735 | 0.213 (0.011) | 0.06 / 0.23 / 0.58 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 2 | C0_same_session | 41 | 0.093 | 0.095 / 0.094 | 0.906 | 0.724 (0.011) | 0.03 / 0.08 / 0.16 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 2 | C1_same_headset_other_day | 41 | 0.150 | 0.152 / 0.150 | 0.849 | 0.546 (0.011) | 0.02 / 0.10 / 0.38 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 2 | C2_other_headset | 41 | 0.234 | 0.237 / 0.234 | 0.765 | 0.307 (0.011) | 0.02 / 0.21 / 0.45 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 3 | C0_same_session | 41 | 0.109 | 0.112 / 0.109 | 0.889 | 0.660 (0.011) | 0.02 / 0.08 / 0.23 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 3 | C1_same_headset_other_day | 41 | 0.167 | 0.170 / 0.167 | 0.831 | 0.507 (0.011) | 0.02 / 0.13 / 0.38 | 2.220446049250313e-16 |
| ballthrowing | treatment_2s | 3 | C2_other_headset | 41 | 0.236 | 0.241 / 0.237 | 0.761 | 0.272 (0.011) | 0.05 / 0.21 / 0.46 | 2.220446049250313e-16 |
