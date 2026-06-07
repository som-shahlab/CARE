#!/usr/bin/env bash
# Reproduce all tables and figures for the CARE paper.
#
# The primary omission controller is LTT-FST (Learn-Then-Test, fixed-sequence
# testing); the 2D-Joint method is retained as a heuristic comparator. The
# factuality controller is 1D scalar CRC throughout.
#
# All model outputs (judge/oracle scores, splits) are read from CARE_DATA_ROOT
# (defaults to /share/pi/nigam/projects/conf-summ). You need read access to
# that data — see README.md. Generated tables/figures land in paper/outputs/.
#
# Steps are run via run()/opt() helpers: a failed step prints a warning and the
# script CONTINUES, then summarizes failures at the end. Steps marked "optional"
# need extra resources (a GPU for external scorers, the clinician-study data for
# the human-eval tables, or the Llama tokenizer for the cost table) and are
# expected to fail in a stock environment.

set -uo pipefail
cd "$(dirname "$0")"

FAILED=()
run() { echo "▶ $*"; if "$@" >/dev/null 2>&1; then echo "  ✓"; else echo "  ✗ FAILED (continuing)"; FAILED+=("$*"); fi; }
opt() { echo "▶ [optional] $*"; if "$@" >/dev/null 2>&1; then echo "  ✓"; else echo "  ⊘ skipped/failed (needs GPU / clinician data / tokenizer)"; fi; }

P=paper/scripts

echo "=== [1/6] Calibration thresholds ==="
run python3 $P/compute_fst_thresholds.py

echo ""
echo "=== [2/6] Resplit CIs — core (100 random 70/30 resplits; slow, ~10-20 min) ==="
run python3 $P/compute_resplit_cis.py        # 2D-Joint comparator + factuality -> resplit_cis.json
run python3 $P/compute_fst_resplit_cis.py    # LTT-FST omission              -> fst_resplit_cis.json
run python3 $P/merge_resplit_cis_fst.py      # canonical merge              -> resplit_cis_fst.json

echo ""
echo "=== [3/6] Sweeps & robustness CIs (slow) ==="
run python3 $P/compute_alpha_sweep_cis.py        # -> alpha_sweep_cis.json (figs, fact panel)
run python3 $P/compute_fst_alpha_sweep_cis.py    # -> fst_alpha_sweep_cis.json (figs, omit panel)
run python3 $P/compute_fst_sample_complexity.py  # -> fst_sample_complexity.json (fig 4, table crc-vs-devset)
run python3 $P/compute_tail_risk.py              # -> tail_risk_cis.json (table 15, tail-risk figure)
run python3 $P/compute_prompt_ablation_cis.py    # -> prompt_ablation_resplit_cis.json (table 8, fact)
run python3 $P/compute_fst_prompt_ablation.py    # -> fst_prompt_ablation.json (table 8)
run python3 $P/compute_fst_summarizer_transfer.py # -> fst_summarizer_resplit_cis.json + gemini_resplit_cis.json (table 11)
run python3 $P/compute_fst_distribution_shift.py # -> fst_distribution_shift.json (table 12; cross-dataset recompute, ~15 min)

echo ""
echo "=== [4/6] Optional / resource-gated computations ==="
opt python3 $P/compute_external_scorer_results.py     # GPU: NLI/BERTScore/AlignScore (2D comparator)
opt python3 $P/compute_fst_external_scorers.py        # GPU: NLI/BERTScore/AlignScore (FST)
opt python3 $P/compute_gemini_judge_results.py        # needs Gemini-judge phase_2 scores
opt python3 $P/compute_oracle_validation_stratified.py # needs clinician annotation CSVs
opt python3 $P/compute_clinician_summary_quality.py   # needs clinician-study data + API key
opt python3 $P/fst_rescore_clinician_study.py         # needs clinician-study data

echo ""
echo "=== [5/6] Tables ==="
run python3 $P/table_1_dataset_stats.py
run python3 $P/table_2_main_results.py
run python3 $P/table_3_baselines.py
run python3 $P/table_5_binary_vs_fractional.py
run python3 $P/table_7_thresholds.py
run python3 $P/table_8_prompt_ablation.py
run python3 $P/table_10_sent_vs_doc.py
run python3 $P/table_11_summarizer_transfer.py
run python3 $P/table_15_tail_risk.py
run python3 $P/table_crc_vs_devset.py
run python3 $P/table_paired_bootstrap.py
run python3 $P/render_table_fst_vs_2djoint.py
run python3 $P/table_12_distribution_shift.py   # renders from fst_distribution_shift.json (no PHI)
run python3 $P/table_9_judge_variance.py
run python3 $P/render_table_13.py               # renders cost table from committed CSV (no tokenizer)
# Optional tables (need GPU-derived scorer JSON or clinician data):
opt python3 $P/table_external_scorers.py
opt python3 $P/table_external_scorers_efficiency.py
opt python3 $P/table_4_oracle_validation.py
opt python3 $P/table_14_clinician_study.py
opt python3 $P/table_13_cost_analysis.py        # recompute from scratch: needs Llama tokenizer

echo ""
echo "=== [6/6] Figures ==="
run python3 $P/figure_2_feasible_set.py
run python3 $P/figure_3_alpha_sweep.py
run python3 $P/figure_4_sample_complexity.py
run python3 $P/figure_5_alpha_sweep_per_dataset.py
run python3 $P/figure_tail_risk_histogram.py

echo ""
echo "=== Done. Outputs in paper/outputs/ ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "WARNING: ${#FAILED[@]} required step(s) failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  exit 1
fi
echo "All required steps succeeded."
