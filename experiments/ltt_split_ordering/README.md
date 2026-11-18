# LTT-FST with a split-learned ordering (experimental)

Ablation of the LTT-FST ordering and deploy rule. It reuses the HB p-value from
`care.calibration` and the data root from `care.config`, and is run by
`reproduce.sh` (App. B).

## Why

The paper (Sec 3.4, App B.1) describes LTT-FST with Hoeffding–Bentkus (HB)
p-values over a fixed τ+γ-ascending ordering. CARE now implements exactly that
(`care.calibration.select_omission_threshold_ltt_fst`). The submission-time code
(`select_omission_threshold_fst_fractional`, kept as a legacy comparator) used a
CRC-bound scan in descending order, which did not carry the LTT guarantee. Run
faithfully, the τ+γ ordering stops early and loses power.

LTT allows any ordering chosen independently of the data used for testing.
This experiment learns the ordering on a held-out part of the calibration set.

## Rule (fixed before running; do not tune)

For each of the 100 random 70/30 calibration/test resplits (seed 42, same as
`paper/scripts/compute_fst_resplit_cis.py`):

1. Randomly split the calibration set into an ordering part (⌊n_cal/3⌋ docs)
   and a certification part (the remaining docs). The split uses
   `np.random.default_rng(10_000 + resplit_index)`.
2. On the ordering part, compute each grid cell's empirical fractional omission
   risk R̂_ord and mean workload Ŵ_ord (surfaced source sentences per doc).
   Candidate cells: R̂_ord ≤ α.
3. Order the candidates by Ŵ_ord descending. Break ties by τ+γ ascending, then
   τ ascending, then γ ascending.
4. On the certification part, run fixed-sequence testing with the HB p-value at
   level δ. Reject H_g: R(g) > α while p_g ≤ δ; stop at the first non-rejection.
5. Deploy the certified cell with the lowest certification-part workload.
   If nothing is certified, fall back to (0, 0), which surfaces everything.

Validity: the ordering depends only on the ordering part. Fixed-sequence
testing on the independent certification part controls FWER at δ, so any
certified cell satisfies Pr(R(g*) ≤ α) ≥ 1 − δ (Angelopoulos et al., LTT,
Thm 1). Selecting among certified cells by workload is allowed.

Settings: α = 0.15, δ = 0.05, grid {0, 0.05, …, 1}² (441 cells).

**Primary result:** published-script float behaviour (`--round 0`), for direct
comparability with the paper's tables. **Robustness check:** float-grid fix
(`--round 1`), where scores and thresholds are rounded so 0.3 compares as 0.3.

## Comparators (same resplits)

- **Code-FST:** the legacy submission-time CRC-bound scan (reproduces the submitted paper's numbers).
- **LTT τ+γ (last):** the paper's procedure as written.
- **LTT τ+γ (min-WL):** the same, but deploying the min-workload certified cell.
- **1D-Imp, Product, Union:** the paper's CRC baselines (0.05 grid, as published).
- **1D-Imp HB, Product HB, Union HB:** the same baselines held to the same HB/δ
  guarantee. Union splits α and δ in half.

## Run

```bash
CARE_DATA_ROOT=/path/to/data PYTHONPATH=. python experiments/ltt_split_ordering/run.py --round 0
```

Outputs go to `experiments/ltt_split_ordering/outputs/`.
