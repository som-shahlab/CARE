# Results (round0; α=0.15, δ=0.05, 100 resplits)

Cells: mean test violation % / mean surfaced source sentences per doc / % of resplits with test violation > α

| Dataset | LTT split-order | Code-FST | LTT τ+γ (last) | LTT τ+γ (min-WL) | 1D-Imp | Product | Union | 1D-Imp HB | Product HB | Union HB | lowest among HB-guaranteed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACI-Bench | 4.1 / 31.0 / 2% | 14.0 / 9.8 / 40% | 4.9 / 22.9 / 3% | 5.6 / 15.3 / 4% | 11.9 / 25.2 / 24% | 14.1 / 11.0 / 44% | 11.3 / 11.5 / 30% | 6.3 / 29.8 / 6% | 6.0 / 15.9 / 4% | 0.9 / 33.6 / 0% | LTT τ+γ (min-WL) |
| MIMIC-BHC | 6.0 / 83.1 / 0% | 13.7 / 66.3 / 7% | 8.7 / 74.0 / 0% | 10.4 / 71.4 / 0% | 11.9 / 84.7 / 35% | 10.1 / 72.3 / 0% | 10.1 / 71.8 / 0% | 10.0 / 87.7 / 0% | 9.8 / 72.9 / 0% | 6.8 / 78.0 / 0% | LTT τ+γ (min-WL) |
| MIMIC-CXR | 7.1 / 1.5 / 0% | 9.3 / 0.9 / 0% | 0.2 / 7.5 / 0% | 2.0 / 2.1 / 0% | 7.6 / 4.9 / 0% | 5.6 / 1.5 / 1% | 8.1 / 1.0 / 0% | 7.6 / 4.9 / 0% | 5.4 / 1.5 / 0% | 5.5 / 1.2 / 0% | Union HB |
| Priv-DS | 9.7 / 230.0 / 0% | 11.7 / 219.0 / 0% | 8.8 / 261.6 / 0% | 9.1 / 235.8 / 0% | 8.8 / 261.6 / 0% | 13.0 / 228.4 / 9% | 9.2 / 234.6 / 0% | 8.8 / 261.6 / 0% | 7.4 / 250.0 / 0% | 6.8 / 255.8 / 0% | LTT split-order |
| SumPubMed | 7.8 / 124.5 / 0% | 12.5 / 112.1 / 14% | 8.3 / 138.5 / 0% | 7.4 / 122.4 / 0% | 6.9 / 137.8 / 0% | 10.1 / 121.0 / 0% | 11.9 / 113.2 / 0% | 6.9 / 137.8 / 0% | 10.1 / 121.0 / 0% | 7.1 / 127.2 / 0% | Product HB |

Split-order fallbacks to (0,0): ACI-Bench 0, MIMIC-BHC 0, MIMIC-CXR 0, Priv-DS 0, SumPubMed 0
