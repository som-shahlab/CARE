#!/usr/bin/env python3
"""
cost_analysis.py

Analyzes token usage and API costs for the CRC pipeline:
  1. Llama 3.3 summarization
  2. GPT-5o-mini judge (fact + importance + coverage, N_REPLICATES replicates)
  3. GPT-5 oracle     (fact + importance + coverage, N_REPLICATES_ORACLE replicate)

Outputs: paper/outputs/table_13_cost_analysis.csv + formatted tables to stdout
"""

import base64
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict

# Ensure repo root is on the path so care/ is importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from care.config import LLAMA_TOKENIZER_DIR, O200K_PATH, PAPER_OUTPUT_DIR, get_phase1_labels
from care.prompts import (
    get_prompt_config,
    get_summarizer_system_prompt,
    get_summarizer_user_prompt,
)
from tiktoken import Encoding
from transformers import AutoTokenizer

# CONSTANTS

# Prompt keys are dataset-specific identifiers used by care.prompts
_DATASET_PROMPT_KEYS = {
    "MIMIC_IV_BHC":  "bhc",
    "ACI_Bench":     "aci",
    "OMOP":          "omop",
    "SumPubMed":     "pubmed",
    "MIMIC_III_CXR": "cxr",
}
DATASETS = {
    ds: (key, get_phase1_labels(ds))
    for ds, key in _DATASET_PROMPT_KEYS.items()
}

# Cost constants
LLAMA_INPUT_USD_PER_1K  = 0.00071
LLAMA_OUTPUT_USD_PER_1K = 0.00071
GPT5O_MINI_INPUT_PER_1M  = 0.25
GPT5O_MINI_OUTPUT_PER_1M = 2.00
GPT5_INPUT_PER_1M  = 1.25
GPT5_OUTPUT_PER_1M = 10.00
# Gemini 2.5 Pro standard tier (Google direct, ≤200K context, 2026): $1.25/M
# input, $10/M output. Azure does not host Gemini natively, so we use Google's
# published rates as the canonical price.
GEMINI_25_PRO_INPUT_PER_1M  = 1.25
GEMINI_25_PRO_OUTPUT_PER_1M = 10.00

# Datasets where Gemini 2.5 Pro was actually run as a judge (App. K).
GEMINI_DATASETS = {"ACI_Bench", "MIMIC_IV_BHC"}

# External scorers (NLI cross-encoder, BERT embedding, AlignScore) run as local
# inference on a single GPU; no per-token API cost. We report $0.00 in the API
# cost column with a footnote explaining this.
EXTERNAL_SCORER_LABEL = "external_local"

# Batching / output parameters
BATCH_SIZE             = 15
OUTPUT_TOKENS_PER_SENT = 7
N_REPLICATES           = 5
CHAT_OVERHEAD_PER_CALL = 0

# oracle-specific
N_REPLICATES_ORACLE = 1

N_CALIB_SAMPLE = 100  # number of calibration docs to sample from the calib split

# Module-level tokenizer handles — set in main()
llama_tokenizer = None
enc = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_calib_indices(oracle_path_str, n=N_CALIB_SAMPLE):
    """
    Load a random sample of `n` calibration doc_ids from the dataset's split_indices.json.
    Derives the split path from the oracle_labels.jsonl path:
      .../phase_1/oracle_labels.jsonl  ->  .../split/split_indices.json
    Returns a set of int doc_ids, or None if file not found.
    """
    split_path = Path(oracle_path_str).parent.parent / "split" / "split_indices.json"
    if not split_path.exists():
        print(f"[WARNING] split_indices.json not found: {split_path}")
        return None
    with split_path.open() as f:
        splits = json.load(f)
    calib_all = splits["calibration_indices"]
    random.seed(42)
    calib_indices = random.sample(calib_all, min(n, len(calib_all)))
    return set(calib_indices)


# Summarization (Llama)

def build_llama_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def count_tokens(text: str) -> int:
    return len(llama_tokenizer(text, add_special_tokens=False)["input_ids"])


def compute_base_tokens(dataset_key):
    sys_p = get_summarizer_system_prompt(dataset_key)
    usr_p = get_summarizer_user_prompt(dataset_key, "")
    full = build_llama_chat_prompt(sys_p, usr_p)
    return count_tokens(full)


def compute_token_stats(path, source_field="source_text", summary_field="summary",
                        row_indices=None):
    """
    Returns per-document token counts as lists, so callers can compute
    sum/mean exactly rather than using medians or caps.

    row_indices: optional set of doc_ids to restrict to (calibration split).
    Returns:
        source_token_list  : list of int, one per document
        output_token_list  : list of int | None per document (None if field missing)
        n_rows             : int
    """
    source_tokens = []
    output_tokens = []
    path = Path(path)
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if row_indices is not None and obj["doc_id"] not in row_indices:
                continue
            src = obj.get(source_field, "")
            source_tokens.append(count_tokens(src))
            if summary_field in obj and obj[summary_field] is not None:
                output_tokens.append(count_tokens(str(obj[summary_field])))
            else:
                output_tokens.append(None)
    return source_tokens, output_tokens, len(source_tokens)


def run_summarization():
    SUMMARY_FIELD = "generated_summary"

    token_stats = {}
    for display_name, (prompt_key, path) in DATASETS.items():
        if not Path(path).exists():
            token_stats[display_name] = {"error": "file_not_found", "path": path}
            continue

        base_tokens = compute_base_tokens(prompt_key)
        calib_indices = load_calib_indices(str(path))
        src_list, out_list, n_rows = compute_token_stats(
            path, source_field="source_text", summary_field=SUMMARY_FIELD,
            row_indices=calib_indices,
        )

        mean_padded = n_rows > 0 and n_rows < N_CALIB_SAMPLE
        n_actual = n_rows
        if mean_padded:
            pad = N_CALIB_SAMPLE - n_rows
            mean_src = sum(src_list) / n_rows
            valid_out = [t for t in out_list if t is not None]
            mean_out = sum(valid_out) / len(valid_out) if valid_out else None
            src_list = src_list + [mean_src] * pad
            out_list = out_list + [mean_out] * pad
            n_rows   = N_CALIB_SAMPLE

        total_input_list  = [base_tokens + s for s in src_list]
        total_output_list = out_list

        src_sum = sum(src_list)
        inp_sum = sum(total_input_list)
        valid_out = [t for t in total_output_list if t is not None]
        out_sum   = sum(valid_out) if valid_out else None
        n_out     = len(valid_out)

        token_stats[display_name] = {
            "prompt_key": prompt_key,
            "n_rows": n_rows,
            "base_prompt_tokens": base_tokens,
            "sum_source_tokens": src_sum,
            "sum_total_input_tokens": inp_sum,
            "sum_output_tokens": out_sum,
            "n_rows_with_output": n_out,
            "mean_source_tokens": src_sum / n_rows,
            "mean_total_input_tokens": inp_sum / n_rows,
            "mean_output_tokens": (out_sum / n_out) if out_sum is not None else None,
            "mean_padded": mean_padded,
            "n_actual_calib_docs": n_actual,
            "_per_doc_total_input_tokens": total_input_list,
            "_per_doc_output_tokens": total_output_list,
        }

    # Print summary
    for ds, stats in token_stats.items():
        print(f"\n {ds} ")
        if stats.get("mean_padded"):
            print(f"  *** MEAN-PADDED: only {stats['n_actual_calib_docs']} calib docs found, padded to {N_CALIB_SAMPLE} ***")
        for k, v in stats.items():
            if k.startswith("_"):
                continue
            print(f"  {k}: {v}")

    # Cost estimation
    total_cost_stats = {}
    for ds, s in token_stats.items():
        if "sum_total_input_tokens" not in s or "n_rows" not in s:
            continue

        n_rows  = s["n_rows"]
        inp_sum = s["sum_total_input_tokens"]

        if s.get("sum_output_tokens") is not None:
            out_sum    = s["sum_output_tokens"]
            n_out      = s["n_rows_with_output"]
            out_source = f"({n_out}/{n_rows} docs)"
        else:
            print(f"[WARNING] {ds}: no output tokens found; skipping output cost.")
            out_sum    = 0
            n_out      = 0
            out_source = "unavailable"

        input_cost_dataset  = (inp_sum / 1000.0) * LLAMA_INPUT_USD_PER_1K
        output_cost_dataset = (out_sum / 1000.0) * LLAMA_OUTPUT_USD_PER_1K
        total_cost_dataset  = input_cost_dataset + output_cost_dataset
        mean_in  = inp_sum / n_rows
        mean_out = (out_sum / n_out) if n_out else 0.0

        total_cost_stats[ds] = {
            "n_rows": n_rows,
            "sum_input_tokens": inp_sum,
            "sum_output_tokens": out_sum,
            "output_token_source": out_source,
            "mean_input_tokens": mean_in,
            "mean_output_tokens": mean_out,
            "input_cost_per_example_usd":  (mean_in  / 1000.0) * LLAMA_INPUT_USD_PER_1K,
            "output_cost_per_example_usd": (mean_out / 1000.0) * LLAMA_OUTPUT_USD_PER_1K,
            "total_cost_per_example_usd":  (mean_in  / 1000.0) * LLAMA_INPUT_USD_PER_1K + (mean_out / 1000.0) * LLAMA_OUTPUT_USD_PER_1K,
            "input_cost_dataset_usd":  input_cost_dataset,
            "output_cost_dataset_usd": output_cost_dataset,
            "total_cost_dataset_usd":  total_cost_dataset,
        }

    print("\nLlama Summarization Cost Estimates (Input + Output, exact sums)")
    print(f"Prices: input=${LLAMA_INPUT_USD_PER_1K}/1K, output=${LLAMA_OUTPUT_USD_PER_1K}/1K")
    for ds, r in total_cost_stats.items():
        print(f"\n{ds}")
        print(f"  n_rows:               {r['n_rows']}")
        print(f"  sum_input_tokens:     {r['sum_input_tokens']}")
        print(f"  sum_output_tokens:    {r['sum_output_tokens']}  [{r['output_token_source']}]")
        print(f"  mean_input_tokens:    {r['mean_input_tokens']:.1f}")
        print(f"  mean_output_tokens:   {r['mean_output_tokens']:.1f}")
        print(f"  input_cost/example:   ${r['input_cost_per_example_usd']:.6f}")
        print(f"  output_cost/example:  ${r['output_cost_per_example_usd']:.6f}")
        print(f"  TOTAL_cost/example:   ${r['total_cost_per_example_usd']:.6f}")
        print(f"  input_cost_dataset:   ${r['input_cost_dataset_usd']:.4f}")
        print(f"  output_cost_dataset:  ${r['output_cost_dataset_usd']:.4f}")
        print(f"  TOTAL_cost_dataset:   ${r['total_cost_dataset_usd']:.4f}")

    return total_cost_stats


# Judge / Oracle GPT tokenizer

def tok(text: str) -> int:
    return len(enc.encode(text))


def compute_doc_judge_tokens(cfg, obj, batch_size=BATCH_SIZE,
                              output_tokens_per_sent=OUTPUT_TOKENS_PER_SENT):
    source_text         = obj.get("source_text", "")
    source_sentences    = obj.get("source_sentences", []) or []
    generated_summary   = obj.get("generated_summary", "")
    generated_sentences = obj.get("generated_sentences", []) or []

    n_gen = len(generated_sentences)
    n_src = len(source_sentences)

    fact_sys_tok = tok(cfg.factuality_system_prompt)
    imp_sys_tok  = tok(cfg.importance_judge_system_prompt)
    cov_sys_tok  = tok(cfg.coverage_system_prompt)
    src_text_tok = tok(source_text)
    gen_sum_tok  = tok(generated_summary)

    fact_ctx_per_call = fact_sys_tok + src_text_tok + CHAT_OVERHEAD_PER_CALL
    imp_ctx_per_call  = imp_sys_tok  + src_text_tok + CHAT_OVERHEAD_PER_CALL
    cov_ctx_per_call  = cov_sys_tok  + gen_sum_tok  + CHAT_OVERHEAD_PER_CALL

    gen_sent_tok = sum(tok(s) for s in generated_sentences)
    src_sent_tok = sum(tok(s) for s in source_sentences)

    n_fact_calls = math.ceil(n_gen / batch_size) if n_gen > 0 else 0
    n_imp_calls  = math.ceil(n_src / batch_size) if n_src > 0 else 0
    n_cov_calls  = math.ceil(n_src / batch_size) if n_src > 0 else 0

    fact_input = n_fact_calls * fact_ctx_per_call + gen_sent_tok
    imp_input  = n_imp_calls  * imp_ctx_per_call  + src_sent_tok
    cov_input  = n_cov_calls  * cov_ctx_per_call  + src_sent_tok

    fact_output = n_gen * output_tokens_per_sent
    imp_output  = n_src * output_tokens_per_sent
    cov_output  = n_src * output_tokens_per_sent

    return {
        "n_gen": n_gen, "n_src": n_src,
        "n_fact_calls": n_fact_calls, "n_imp_calls": n_imp_calls, "n_cov_calls": n_cov_calls,
        "fact_input": fact_input, "imp_input": imp_input, "cov_input": cov_input,
        "omit_input":   imp_input + cov_input,
        "total_input":  fact_input + imp_input + cov_input,
        "fact_output": fact_output, "imp_output": imp_output, "cov_output": cov_output,
        "omit_output":  imp_output + cov_output,
        "total_output": fact_output + imp_output + cov_output,
    }


def compute_doc_oracle_tokens(cfg, obj, batch_size=BATCH_SIZE,
                               output_tokens_per_sent=OUTPUT_TOKENS_PER_SENT):
    """
    Oracle differs from judge only in importance:
      - Judge imp context  = source_text
      - Oracle imp context = reference_summary (comparing to human reference)
    """
    source_text         = obj.get("source_text", "")
    source_sentences    = obj.get("source_sentences", []) or []
    generated_summary   = obj.get("generated_summary", "")
    generated_sentences = obj.get("generated_sentences", []) or []
    reference_summary   = obj.get("reference_text", "") or obj.get("reference_summary", "") or ""

    n_gen = len(generated_sentences)
    n_src = len(source_sentences)

    fact_sys_tok = tok(cfg.factuality_system_prompt)
    imp_sys_tok  = tok(cfg.importance_oracle_system_prompt)
    cov_sys_tok  = tok(cfg.coverage_system_prompt)
    src_text_tok = tok(source_text)
    ref_sum_tok  = tok(reference_summary)
    gen_sum_tok  = tok(generated_summary)

    fact_ctx_per_call = fact_sys_tok + src_text_tok + CHAT_OVERHEAD_PER_CALL
    imp_ctx_per_call  = imp_sys_tok  + ref_sum_tok  + CHAT_OVERHEAD_PER_CALL  # ← reference
    cov_ctx_per_call  = cov_sys_tok  + gen_sum_tok  + CHAT_OVERHEAD_PER_CALL

    gen_sent_tok = sum(tok(s) for s in generated_sentences)
    src_sent_tok = sum(tok(s) for s in source_sentences)

    n_fact_calls = math.ceil(n_gen / batch_size) if n_gen > 0 else 0
    n_imp_calls  = math.ceil(n_src / batch_size) if n_src > 0 else 0
    n_cov_calls  = math.ceil(n_src / batch_size) if n_src > 0 else 0

    fact_input = n_fact_calls * fact_ctx_per_call + gen_sent_tok
    imp_input  = n_imp_calls  * imp_ctx_per_call  + src_sent_tok
    cov_input  = n_cov_calls  * cov_ctx_per_call  + src_sent_tok

    fact_output = n_gen * output_tokens_per_sent
    imp_output  = n_src * output_tokens_per_sent
    cov_output  = n_src * output_tokens_per_sent

    return {
        "n_gen": n_gen, "n_src": n_src,
        "n_fact_calls": n_fact_calls, "n_imp_calls": n_imp_calls, "n_cov_calls": n_cov_calls,
        "fact_input": fact_input, "imp_input": imp_input, "cov_input": cov_input,
        "total_input":  fact_input + imp_input + cov_input,
        "fact_output": fact_output, "imp_output": imp_output, "cov_output": cov_output,
        "total_output": fact_output + imp_output + cov_output,
    }


def _accumulate(path_str, cfg, compute_fn):
    """Shared accumulation loop for judge and oracle."""
    sums = dict(
        n_docs=0, n_gen=0, n_src=0,
        n_fact_calls=0, n_imp_calls=0, n_cov_calls=0,
        fact_input=0, imp_input=0, cov_input=0,
        fact_output=0, imp_output=0, cov_output=0,
    )
    calib_indices = load_calib_indices(path_str)
    with Path(path_str).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if calib_indices is not None and obj["doc_id"] not in calib_indices:
                continue
            d = compute_fn(cfg, obj)
            sums["n_docs"]       += 1
            sums["n_gen"]        += d["n_gen"]
            sums["n_src"]        += d["n_src"]
            sums["n_fact_calls"] += d["n_fact_calls"]
            sums["n_imp_calls"]  += d["n_imp_calls"]
            sums["n_cov_calls"]  += d["n_cov_calls"]
            sums["fact_input"]   += d["fact_input"]
            sums["imp_input"]    += d["imp_input"]
            sums["cov_input"]    += d["cov_input"]
            sums["fact_output"]  += d["fact_output"]
            sums["imp_output"]   += d["imp_output"]
            sums["cov_output"]   += d["cov_output"]

    # Pad to N_CALIB_SAMPLE using mean if fewer docs available
    mean_padded = sums["n_docs"] > 0 and sums["n_docs"] < N_CALIB_SAMPLE
    n_actual = sums["n_docs"]
    if mean_padded:
        scale = N_CALIB_SAMPLE / sums["n_docs"]
        for k in sums:
            if k != "n_docs":
                sums[k] = sums[k] * scale
        sums["n_docs"] = N_CALIB_SAMPLE

    return sums, mean_padded, n_actual


def run_judge_or_oracle(mode="judge"):
    dataset_filter = None
    if mode == "judge":
        compute_fn   = compute_doc_judge_tokens
        n_rep        = N_REPLICATES
        inp_price_1m = GPT5O_MINI_INPUT_PER_1M
        out_price_1m = GPT5O_MINI_OUTPUT_PER_1M
        model_label  = "GPT-5o-mini"
    elif mode == "judge_gemini":
        # Token counts are identical to GPT-5-mini (same prompts, same batches,
        # same m=5 replicates). Only the price differs. We restrict to the
        # datasets where Gemini was actually run as a judge.
        compute_fn     = compute_doc_judge_tokens
        n_rep          = N_REPLICATES
        inp_price_1m   = GEMINI_25_PRO_INPUT_PER_1M
        out_price_1m   = GEMINI_25_PRO_OUTPUT_PER_1M
        model_label    = "Gemini 2.5 Pro"
        dataset_filter = GEMINI_DATASETS
    else:
        compute_fn   = compute_doc_oracle_tokens
        n_rep        = N_REPLICATES_ORACLE
        inp_price_1m = GPT5_INPUT_PER_1M
        out_price_1m = GPT5_OUTPUT_PER_1M
        model_label  = "GPT-5 Oracle"

    def cost_in(toks):  return (toks / 1_000_000) * inp_price_1m
    def cost_out(toks): return (toks / 1_000_000) * out_price_1m

    token_stats = {}
    for ds_name, (prompt_key, path_str) in DATASETS.items():
        if dataset_filter is not None and ds_name not in dataset_filter:
            continue
        if not Path(path_str).exists():
            token_stats[ds_name] = {"error": "file_not_found", "path": path_str}
            continue

        cfg = get_prompt_config(prompt_key)
        sums, mean_padded, n_actual = _accumulate(path_str, cfg, compute_fn)
        n = sums["n_docs"]

        token_stats[ds_name] = {
            "prompt_key":              prompt_key,
            "n_docs":                  n,
            "mean_padded":             mean_padded,
            "n_actual_calib_docs":     n_actual,
            "sum_n_gen":               sums["n_gen"],
            "sum_n_src":               sums["n_src"],
            "sum_fact_calls":          sums["n_fact_calls"],
            "sum_imp_calls":           sums["n_imp_calls"],
            "sum_cov_calls":           sums["n_cov_calls"],
            "sum_fact_input_tokens":   sums["fact_input"],
            "sum_imp_input_tokens":    sums["imp_input"],
            "sum_cov_input_tokens":    sums["cov_input"],
            "sum_total_input_tokens":  sums["fact_input"] + sums["imp_input"] + sums["cov_input"],
            "sum_fact_output_tokens":  sums["fact_output"],
            "sum_imp_output_tokens":   sums["imp_output"],
            "sum_cov_output_tokens":   sums["cov_output"],
            "sum_total_output_tokens": sums["fact_output"] + sums["imp_output"] + sums["cov_output"],
        }

    section_label = "Judge" if mode == "judge" else "ORACLE"
    print(f"\n{model_label} {section_label} token stats | encoding: {enc.name}")
    print(f"Batch size: {BATCH_SIZE} | output_tokens_per_sent: {OUTPUT_TOKENS_PER_SENT} | replicates: {n_rep}")
    for ds, s in token_stats.items():
        print(f"\n {ds} ")
        if "error" in s:
            print(s); continue
        if s.get("mean_padded"):
            print(f"  *** MEAN-PADDED: only {s['n_actual_calib_docs']} calib docs found, padded to {N_CALIB_SAMPLE} ***")
        print(f"  n_docs:                  {s['n_docs']}")
        print(f"  sum_n_gen:               {s['sum_n_gen']}")
        print(f"  sum_n_src:               {s['sum_n_src']}")
        print(f"  sum_fact_input_tokens:   {s['sum_fact_input_tokens']}")
        print(f"  sum_imp_input_tokens:    {s['sum_imp_input_tokens']}")
        print(f"  sum_cov_input_tokens:    {s['sum_cov_input_tokens']}")
        print(f"  sum_total_input_tokens:  {s['sum_total_input_tokens']}")
        print(f"  sum_fact_output_tokens:  {s['sum_fact_output_tokens']}")
        print(f"  sum_imp_output_tokens:   {s['sum_imp_output_tokens']}")
        print(f"  sum_cov_output_tokens:   {s['sum_cov_output_tokens']}")
        print(f"  sum_total_output_tokens: {s['sum_total_output_tokens']}")

    # Cost estimation
    cost_stats = {}
    for ds, s in token_stats.items():
        if "sum_total_input_tokens" not in s:
            continue
        n = s["n_docs"]

        inp_fact  = s["sum_fact_input_tokens"]  * n_rep
        inp_imp   = s["sum_imp_input_tokens"]   * n_rep
        inp_cov   = s["sum_cov_input_tokens"]   * n_rep
        inp_total = s["sum_total_input_tokens"] * n_rep
        out_fact  = s["sum_fact_output_tokens"] * n_rep
        out_imp   = s["sum_imp_output_tokens"]  * n_rep
        out_cov   = s["sum_cov_output_tokens"]  * n_rep
        out_total = s["sum_total_output_tokens"] * n_rep

        cost_stats[ds] = {
            "n_docs": n,
            "n_replicates": n_rep,
            "sum_input_tokens":            inp_total,
            "sum_output_tokens":           out_total,
            "fact_input_cost_dataset":     cost_in(inp_fact),
            "fact_output_cost_dataset":    cost_out(out_fact),
            "fact_total_cost_dataset":     cost_in(inp_fact)  + cost_out(out_fact),
            "imp_input_cost_dataset":      cost_in(inp_imp),
            "imp_output_cost_dataset":     cost_out(out_imp),
            "imp_total_cost_dataset":      cost_in(inp_imp)   + cost_out(out_imp),
            "cov_input_cost_dataset":      cost_in(inp_cov),
            "cov_output_cost_dataset":     cost_out(out_cov),
            "cov_total_cost_dataset":      cost_in(inp_cov)   + cost_out(out_cov),
            "combined_total_cost_dataset": cost_in(inp_total) + cost_out(out_total),
        }

    inp_label = f"${inp_price_1m}/1M"
    out_label = f"${out_price_1m}/1M"
    print(f"\n {model_label} Cost (exact sums, {n_rep} replicate(s))")
    print(f"Input:  {inp_label}  | Output: {out_label}")
    print(f"Batch size: {BATCH_SIZE} | output_tokens_per_sent: {OUTPUT_TOKENS_PER_SENT}")
    for ds, r in cost_stats.items():
        print(f"\n {ds}  (n={r['n_docs']}, {r['n_replicates']} rep)")
        print("FACT:")
        print(f"  input_cost_dataset:   ${r['fact_input_cost_dataset']:.4f}")
        print(f"  output_cost_dataset:  ${r['fact_output_cost_dataset']:.4f}")
        print(f"  total_cost_dataset:   ${r['fact_total_cost_dataset']:.4f}")
        print("IMPORTANCE:")
        print(f"  input_cost_dataset:   ${r['imp_input_cost_dataset']:.4f}")
        print(f"  output_cost_dataset:  ${r['imp_output_cost_dataset']:.4f}")
        print(f"  total_cost_dataset:   ${r['imp_total_cost_dataset']:.4f}")
        print("COVERAGE:")
        print(f"  input_cost_dataset:   ${r['cov_input_cost_dataset']:.4f}")
        print(f"  output_cost_dataset:  ${r['cov_output_cost_dataset']:.4f}")
        print(f"  total_cost_dataset:   ${r['cov_total_cost_dataset']:.4f}")
        print("COMBINED:")
        print(f"  total_cost_dataset:   ${r['combined_total_cost_dataset']:.4f}")

    return cost_stats


# CSV output

def write_csv(summ_costs, judge_costs, gemini_costs, oracle_costs, out_path: Path):
    rows = []

    for ds, r in summ_costs.items():
        rows.append({
            "dataset": ds, "section": "summarization", "model": "llama3.3",
            "n_docs": r["n_rows"], "n_replicates": 1,
            "sum_input_tokens": r["sum_input_tokens"],
            "sum_output_tokens": r["sum_output_tokens"] if r["sum_output_tokens"] is not None else "",
            "mean_input_tokens": round(r["mean_input_tokens"], 1),
            "mean_output_tokens": round(r["mean_output_tokens"], 1),
            "fact_cost_usd": "", "imp_cost_usd": "", "cov_cost_usd": "",
            "total_cost_usd": round(r["total_cost_dataset_usd"], 4),
        })

    for costs, section, model in [
        (judge_costs,   "judge",        "gpt-5o-mini"),
        (gemini_costs,  "judge_gemini", "gemini-2.5-pro"),
        (oracle_costs,  "oracle",       "gpt-5"),
    ]:
        for ds, r in costs.items():
            rows.append({
                "dataset": ds, "section": section, "model": model,
                "n_docs": r["n_docs"], "n_replicates": r["n_replicates"],
                "sum_input_tokens": r["sum_input_tokens"],
                "sum_output_tokens": r["sum_output_tokens"],
                "mean_input_tokens": "", "mean_output_tokens": "",
                "fact_cost_usd":  round(r["fact_total_cost_dataset"], 4),
                "imp_cost_usd":   round(r["imp_total_cost_dataset"], 4),
                "cov_cost_usd":   round(r["cov_total_cost_dataset"], 4),
                "total_cost_usd": round(r["combined_total_cost_dataset"], 4),
            })

    # External scorers: NLI cross-encoder, BERT embedding, AlignScore.
    # Local GPU inference, no per-token API cost. We emit one row per
    # (dataset, scorer) combination with cost = $0 to make the LLM-vs-external
    # cost gap explicit; token counts are not applicable.
    for ds in DATASETS:
        for scorer_name in ("nli_deberta", "bert_embed", "alignscore"):
            rows.append({
                "dataset": ds, "section": "judge_external", "model": scorer_name,
                "n_docs": N_CALIB_SAMPLE, "n_replicates": 1,
                "sum_input_tokens": "", "sum_output_tokens": "",
                "mean_input_tokens": "", "mean_output_tokens": "",
                "fact_cost_usd": 0.0, "imp_cost_usd": "", "cov_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            })

    cols = ["dataset", "section", "model", "n_docs", "n_replicates",
            "sum_input_tokens", "sum_output_tokens",
            "mean_input_tokens", "mean_output_tokens",
            "fact_cost_usd", "imp_cost_usd", "cov_cost_usd", "total_cost_usd"]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")


# Tokenizer loading

def load_llama_tokenizer():
    return AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_DIR, use_fast=True)


def load_gpt_tokenizer() -> Encoding:
    assert O200K_PATH.exists(), f"File not found: {O200K_PATH.resolve()}"
    mergeable_ranks: Dict[bytes, int] = {}
    with O200K_PATH.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token_b64, rank_s = line.split()
            mergeable_ranks[base64.b64decode(token_b64)] = int(rank_s)
    special_tokens = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}
    return Encoding(
        name="o200k_base_local",
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


# Main

def main():
    global llama_tokenizer, enc

    print("Loading Llama tokenizer...")
    llama_tokenizer = load_llama_tokenizer()

    print("Loading GPT tokenizer (o200k)...")
    enc = load_gpt_tokenizer()
    print("Loaded encoding:", enc.name, "| vocab size:", enc.n_vocab)

    summ_costs   = run_summarization()
    judge_costs  = run_judge_or_oracle("judge")
    gemini_costs = run_judge_or_oracle("judge_gemini")
    oracle_costs = run_judge_or_oracle("oracle")

    write_csv(summ_costs, judge_costs, gemini_costs, oracle_costs,
              PAPER_OUTPUT_DIR / "table_13_cost_analysis.csv")


if __name__ == "__main__":
    main()