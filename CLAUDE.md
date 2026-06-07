# CLAUDE.md

## Project Overview
This project implements **post-hoc conformal risk control (CRC)** for clinical summarization.
We do **not** retrain or regenerate summaries. The system overlays calibrated risk signals on
black-box LLM summaries to control:
1) factual error risk, and
2) omission of important content.

The core idea is:
- Controller A: factuality filtering over summary sentences
- Controller B: importance recall over source sentences

Human reference summaries are used **offline only** for supervision and calibration.

---

## Non-Negotiables
- Do NOT assume access to human references at test time.
- Do NOT conflate factuality and omission into a single controller.
- Guarantees must be **finite-sample, distribution-free**, and **clearly stated**.
- Monotonicity of losses must be explicit.

## Repo layout:
- `care/`:              Core package (config, pipeline phases, filters, prompts, LLM client)
- `paper/`:             Paper source, table/figure generation scripts, outputs
- `paper/tables/`:      Table generation scripts (one per paper table)
- `paper/figures/`:     Figure generation scripts
- `paper/outputs/`:     Generated .tex, .png files
- `data/`:              Local data cache (canonical data at CARE_DATA_ROOT)
- `docs/plan.tex`:      Formal project specification
- `reproduce.sh`:       Regenerate all tables and figures

Install: `pip install -e .`
Reproduce: `./reproduce.sh`