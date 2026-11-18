#!/usr/bin/env bash
# Reproduce all tables and figures for the CARE paper.
#
# The primary omission controller is LTT-FST (Learn-Then-Test, fixed-sequence
# testing); the 2D-Joint method is retained as a heuristic comparator. The
# factuality controller is 1D scalar CRC throughout.
#
# All model outputs (judge/oracle scores, splits) are read from CARE_DATA_ROOT
# (defaults to /path/to/data). You need read access to
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
# GPU scorer steps; set SKIP_GPU=1 to reuse existing external-scorer results (e.g. while a long rescoring job runs).
gpu() { if [ -n "${SKIP_GPU:-}" ]; then echo "▶ [gpu, skipped: SKIP_GPU] $*"; else opt "$@"; fi; }

P=paper/scripts
R=paper/rebuttal
# Interpreter with the project dependencies (numpy, scipy, pandas, sklearn, matplotlib, torch).
PY=${PYTHON:-python3}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [1/6] Calibration thresholds ==="
run $PY $P/compute_fst_thresholds.py

echo ""
echo "=== [2/6] Resplit CIs — core (100 random 70/30 resplits; slow, ~10-20 min) ==="
run $PY $P/compute_resplit_cis.py        # 2D-Joint comparator + factuality -> resplit_cis.json
run $PY $P/compute_fst_resplit_cis.py    # LTT-FST omission              -> fst_resplit_cis.json
run $PY $P/merge_resplit_cis_fst.py      # canonical merge              -> resplit_cis_fst.json

echo ""
echo "=== [3/6] Sweeps & robustness CIs (slow) ==="
run $PY $P/compute_alpha_sweep_cis.py        # -> alpha_sweep_cis.json (figs, fact panel)
run $PY $P/compute_fst_alpha_sweep_cis.py    # -> fst_alpha_sweep_cis.json (figs, omit panel)
run $PY $P/compute_fst_sample_complexity.py  # -> fst_sample_complexity.json (fig 4, table crc-vs-devset)
run $PY $P/compute_tail_risk.py              # -> tail_risk_cis.json (table 15, tail-risk figure)
run $PY $P/compute_prompt_ablation_cis.py    # -> prompt_ablation_resplit_cis.json (table 8, fact)
run $PY $P/compute_fst_prompt_ablation.py    # -> fst_prompt_ablation.json (table 8)
run $PY $P/compute_fst_summarizer_transfer.py # -> fst_summarizer_resplit_cis.json + gemini_resplit_cis.json (table 11)
run $PY $P/compute_fst_distribution_shift.py # -> fst_distribution_shift.json (table 12; cross-dataset recompute, ~15 min)

echo ""
echo "=== [4/6] Optional / resource-gated computations ==="
gpu $PY $P/compute_external_scorer_results.py     # GPU: NLI/BERTScore/AlignScore (2D comparator)
gpu $PY $P/compute_fst_external_scorers.py        # GPU: NLI/BERTScore/AlignScore (FST)
opt $PY $P/compute_gemini_judge_results.py        # submission-time 2D-Joint judge comparison (kept for reference)
run $PY $P/compute_fst_alt_judges.py              # Gemini 2.5 Pro / Qwen3.5-4B judges under CRC + LTT-FST -> fst_alt_judge_results.json
opt $PY $P/compute_oracle_validation_stratified.py # needs clinician annotation CSVs
opt $PY $P/compute_clinician_summary_quality.py   # needs clinician-study data + API key

echo ""
echo "=== [4b/6] Camera-ready analyses (rebuttal) ==="
run $PY experiments/ltt_split_ordering/run.py --round 0    # deploy-rule / ordering ablation (App. B)
opt $PY $R/r3_q1_label_agreement.py      # oracle-clinician error-class agreement + confusion counts (App. A)
opt $PY $R/r4_human_ground_truth.py      # thresholds scored against clinician labels (App. A)
run $PY $R/r4_oracle_noise.py            # oracle-label noise stress test (App. A)
opt $PY $R/r3_q2_clinician_breakdown.py  # clinician-study endpoint decomposition (Sec. 5.6, App. L)
run $PY $R/r3_w4_doc_abstention.py       # document-level abstention baselines, 100 + 2000 resplits (App. J)
gpu $PY $R/r1_w2_default_thresholds.py   # detectors at default 0.5 threshold (App. H; needs scorer caches)
run $PY $R/r2_privds_redundancy.py       # Priv-DS copy-forward (App. D)
run $PY $R/r3_q4_grouping.py             # reference exclusion + patient overlap (App. D)

echo ""
echo "=== [5/6] Tables ==="
run $PY $P/table_1_dataset_stats.py
run $PY $P/table_2_main_results.py
run $PY $P/table_3_baselines.py
run $PY $P/table_5_binary_vs_fractional.py
run $PY $P/table_7_thresholds.py
run $PY $P/table_8_prompt_ablation.py
run $PY $P/table_10_sent_vs_doc.py
run $PY $P/table_11_summarizer_transfer.py
run $PY $P/table_15_tail_risk.py
run $PY $P/table_crc_vs_devset.py
run $PY $P/table_paired_bootstrap.py
run $PY $P/render_table_fst_vs_2djoint.py
run $PY $P/table_12_distribution_shift.py   # renders from fst_distribution_shift.json (no PHI)
run $PY $P/table_9_judge_variance.py
run $PY $P/render_table_13.py               # renders cost table from committed CSV (no tokenizer)
# Optional tables (need GPU-derived scorer JSON or clinician data):
opt $PY $P/table_external_scorers.py
opt $PY $P/table_external_scorers_efficiency.py
opt $PY $P/table_4_oracle_validation.py
opt $PY $P/table_14_clinician_study.py
opt $PY $P/table_13_cost_analysis.py        # recompute from scratch: needs Llama tokenizer

run $PY $P/camera_ready_tables.py        # camera-ready table fragments -> paper/outputs/camera_ready/
run $PY $P/example_aci.py                # worked ACI-Bench example -> paper/outputs/camera_ready/example_aci.tex
opt $PY $P/patch_external_scorer_tables.py   # write external-scorer LTT-FST cells into ML4H_2026___CARE/main.tex (Tables 11/12)
opt $PY $P/fst_rescore_clinician_study.py    # clinician-study flags recomputed under each threshold regime

echo ""
echo "=== [6/6] Figures ==="
run $PY $P/figure_2_feasible_set.py
run $PY $P/figure_3_alpha_sweep.py
run $PY $P/figure_4_sample_complexity.py
run $PY $P/figure_5_alpha_sweep_per_dataset.py
run $PY $P/figure_tail_risk_histogram.py

echo ""
echo "=== Done. Outputs in paper/outputs/ ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "WARNING: ${#FAILED[@]} required step(s) failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  exit 1
fi
echo "All required steps succeeded."
