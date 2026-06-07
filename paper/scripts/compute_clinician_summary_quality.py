#!/usr/bin/env python3
"""Re-label coverage on clinician-edited summaries to compare summary quality
with and without CARE assistance.

Pipeline
--------
1. Load the six clinician-study JSONs (v2 + v3, 3 clinicians).
2. For each (clinician, doc) response, retrieve source_sentences,
   importance_labels, and the original LLM summary's coverage labels from
   phase_1/oracle_labels.jsonl in the canonical data dir.
3. Call `care.oracle.label_coverage(source_sentences, clinician_summary,
   oracle)` with GPT-5 to get fresh coverage labels for the clinician's
   edited summary.
4. Compute per-doc omission rates against the original LLM summary and the
   clinician-edited summary.
5. Aggregate CARE vs. control with a stratified permutation test
   (within clinician x version), same as table_14.
6. Cache per-doc results so reruns skip docs already labeled.

Outputs
-------
    paper/outputs/clinician_summary_quality_cache.jsonl   # per-doc rows
    paper/outputs/clinician_summary_quality.json          # aggregate summary

Usage
-----
    SECUREGPT_API_KEY=... python3 paper/scripts/compute_clinician_summary_quality.py
    python3 paper/scripts/compute_clinician_summary_quality.py --analyze-only
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from care import config
from care import oracle as oracle_module
from care import prompts as prompts_module
from care.oracle import label_coverage
from care.utils import LLMClient

PAPER_DIR = Path(__file__).resolve().parent.parent
CLINICIAN_DIR = PAPER_DIR / "clinician_study"
OUTPUT_DIR = PAPER_DIR / "outputs"
CACHE_PATH = OUTPUT_DIR / "clinician_summary_quality_cache.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "clinician_summary_quality.json"

STUDY_FILES = [
    ("v2", "C1", "Chloe",   "clinician_study_v2_Chloe.json"),
    ("v2", "C2", "Jenelle", "clinician_study_v2_Jenelle.json"),
    ("v2", "C3", "Anson",   "clinician_study_v2_anson.json"),
    ("v3", "C1", "Chloe",   "clinician_study_v3_chloe.json"),
    ("v3", "C2", "Jenelle", "clinician_study_v3_jenelle.json"),
    ("v3", "C3", "Anson",   "clinician_study_v3_anson.json"),
]

PERM_ITERS = 10000
SEED = 0


def load_phase1(dataset: str) -> dict:
    path = config.CANONICAL_DATA_ROOT / dataset / "phase_1" / "oracle_labels.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def load_study_responses():
    """Yield (version, label, name, response) for every clinician-study row."""
    for version, label, name, fname in STUDY_FILES:
        data = json.load(open(CLINICIAN_DIR / fname))
        for r in data["responses"]:
            yield version, label, name, r


def load_cache() -> dict:
    cache = {}
    if not CACHE_PATH.exists():
        return cache
    with open(CACHE_PATH) as f:
        for line in f:
            row = json.loads(line)
            cache[(row["version"], row["clinician"], row["doc_id"])] = row
    return cache


def append_cache(row: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def omission_rate(importance_labels, coverage_labels):
    imp = np.asarray(importance_labels)
    cov = np.asarray(coverage_labels)
    n = min(len(imp), len(cov))
    imp, cov = imp[:n], cov[:n]
    n_imp = int((imp == 1).sum())
    if n_imp == 0:
        return float("nan")
    n_omit = int(((imp == 1) & (cov == 0)).sum())
    return n_omit / n_imp


def make_oracle() -> LLMClient:
    if not config.SECUREGPT_API_KEY:
        sys.exit("SECUREGPT_API_KEY env var is not set; cannot call GPT-5.")
    return LLMClient(
        model=config.ORACLE_MODEL,
        api_key=config.SECUREGPT_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.get_azure_endpoint(config.ORACLE_MODEL),
        max_tokens=config.ORACLE_MAX_TOKENS,
        rate_limit_delay=config.RATE_LIMIT_DELAY,
    )


def score(cache_only: bool):
    cache = load_cache()
    print(f"Cache: {len(cache)} entries at {CACHE_PATH}")

    phase1_cache = {}
    oracle = None

    todo = []
    for version, label, name, r in load_study_responses():
        key = (version, name, r["doc_id"])
        if key in cache:
            continue
        orig = (r.get("original_llm_summary") or "").strip()
        clin = (r.get("clinician_summary") or "").strip()
        if not clin or clin == orig:
            # Kept the LLM draft verbatim: summary quality == original
            append_cache(dict(
                version=version, label=label, clinician=name,
                doc_id=r["doc_id"], dataset=r["dataset"],
                condition=r["condition"], edited=False,
                note="summary unchanged from LLM draft",
            ))
            continue
        todo.append((version, label, name, r))

    print(f"To score: {len(todo)} edited summaries (cache-only={cache_only})")
    if cache_only or not todo:
        return

    oracle = make_oracle()
    print(f"Oracle model: {config.ORACLE_MODEL}  (temperature={config.ORACLE_TEMPERATURE})")

    # `prompts.get_prompt_config` expects short keys ('aci', 'bhc'); map from folder.
    PROMPT_KEY = {"ACI_Bench": "aci", "MIMIC_IV_BHC": "bhc",
                  "MIMIC_III_CXR": "cxr", "SumPubMed": "pubmed", "OMOP": "omop"}
    prompt_cfgs = {}
    for version, label, name, r in tqdm(todo):
        ds = r["dataset"]
        if ds not in prompt_cfgs:
            prompt_cfgs[ds] = prompts_module.get_prompt_config(PROMPT_KEY[ds])
        oracle_module.prompt_config = prompt_cfgs[ds]
        if ds not in phase1_cache:
            phase1_cache[ds] = load_phase1(ds)
        p1 = phase1_cache[ds].get(r["doc_id"])
        if p1 is None:
            append_cache(dict(
                version=version, label=label, clinician=name,
                doc_id=r["doc_id"], dataset=ds, condition=r["condition"],
                edited=True, note="doc missing from phase_1",
            ))
            continue
        source_sents = p1["source_sentences"]
        imp = p1["importance_labels"]
        cov_orig = p1["coverage_labels"]
        clin_summary = (r["clinician_summary"] or "").strip()

        try:
            cov_clin = label_coverage(source_sents, clin_summary, oracle)
        except Exception as e:
            append_cache(dict(
                version=version, label=label, clinician=name,
                doc_id=r["doc_id"], dataset=ds, condition=r["condition"],
                edited=True, error=str(e),
            ))
            continue

        row = dict(
            version=version, label=label, clinician=name,
            doc_id=r["doc_id"], dataset=ds, condition=r["condition"], edited=True,
            n_src=len(source_sents),
            n_imp=int(sum(imp)),
            omit_rate_orig=omission_rate(imp, cov_orig),
            omit_rate_clin=omission_rate(imp, cov_clin),
            coverage_labels_clin=cov_clin,
        )
        row["reduction"] = (
            row["omit_rate_orig"] - row["omit_rate_clin"]
            if not np.isnan(row["omit_rate_orig"]) and not np.isnan(row["omit_rate_clin"])
            else float("nan")
        )
        append_cache(row)


def permutation_test(rows, metric, direction, n_iter=PERM_ITERS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = [r for r in rows if metric in r and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
    if not rows:
        return float("nan"), float("nan")

    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r["clinician"], r["version"]), []).append(i)

    vals = np.array([r[metric] for r in rows])
    is_care = np.array([r["condition"] == "highlighted" for r in rows], dtype=bool)
    obs = vals[is_care].mean() - vals[~is_care].mean()

    perm = is_care.copy()
    null = np.empty(n_iter)
    for it in range(n_iter):
        for idxs in strata.values():
            sub = perm[idxs].copy()
            rng.shuffle(sub)
            perm[idxs] = sub
        null[it] = vals[perm].mean() - vals[~perm].mean()

    if direction == "greater":
        p = (null >= obs).mean()
    elif direction == "less":
        p = (null <= obs).mean()
    else:
        p = (np.abs(null) >= abs(obs)).mean()
    return obs, p


def analyze():
    cache = load_cache()
    rows = [r for r in cache.values() if r.get("edited") and "omit_rate_clin" in r]
    if not rows:
        print("No scored rows in cache yet.")
        return

    ctrl = [r for r in rows if r["condition"] == "control"]
    care = [r for r in rows if r["condition"] == "highlighted"]

    def m(xs, k):
        v = [x[k] for x in xs if not (isinstance(x[k], float) and np.isnan(x[k]))]
        return float(np.mean(v)) if v else float("nan")

    agg = dict(
        n_total=len(rows), n_ctrl=len(ctrl), n_care=len(care),
        orig_omit_rate_ctrl=m(ctrl, "omit_rate_orig"),
        orig_omit_rate_care=m(care, "omit_rate_orig"),
        clin_omit_rate_ctrl=m(ctrl, "omit_rate_clin"),
        clin_omit_rate_care=m(care, "omit_rate_clin"),
        reduction_ctrl=m(ctrl, "reduction"),
        reduction_care=m(care, "reduction"),
    )

    d_clin, p_clin = permutation_test(rows, "omit_rate_clin", "less")    # CARE should lower omission
    d_red,  p_red  = permutation_test(rows, "reduction", "greater")       # CARE should increase reduction
    agg.update(
        delta_clin_omit_rate=d_clin, p_clin_omit_rate=p_clin,
        delta_reduction=d_red, p_reduction=p_red,
    )

    per_clin = []
    for lbl in ["C1", "C2", "C3"]:
        sub = [r for r in rows if r["label"] == lbl]
        sc = [r for r in sub if r["condition"] == "control"]
        sh = [r for r in sub if r["condition"] == "highlighted"]
        per_clin.append(dict(
            label=lbl,
            name=sub[0]["clinician"] if sub else "",
            n_ctrl=len(sc), n_care=len(sh),
            clin_omit_ctrl=m(sc, "omit_rate_clin"),
            clin_omit_care=m(sh, "omit_rate_clin"),
            reduction_ctrl=m(sc, "reduction"),
            reduction_care=m(sh, "reduction"),
        ))
    agg["per_clinician"] = per_clin

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(agg, f, indent=2)

    print()
    print(f"Scored summaries: {agg['n_total']}  ({agg['n_ctrl']} ctrl / {agg['n_care']} CARE)")
    print()
    print("Omission rate on edited summary (lower is better):")
    print(f"  Control: {agg['clin_omit_rate_ctrl']:.3f}")
    print(f"  CARE   : {agg['clin_omit_rate_care']:.3f}")
    print(f"  Δ (CARE-ctrl) = {agg['delta_clin_omit_rate']:+.3f}   perm p (CARE<ctrl) = {agg['p_clin_omit_rate']:.3f}")
    print()
    print("Omission reduction from LLM draft (higher is better):")
    print(f"  Control: {agg['reduction_ctrl']:+.3f}")
    print(f"  CARE   : {agg['reduction_care']:+.3f}")
    print(f"  Δ (CARE-ctrl) = {agg['delta_reduction']:+.3f}   perm p (CARE>ctrl) = {agg['p_reduction']:.3f}")
    print()
    print("Baseline LLM omission rate (sanity check — should be similar across arms):")
    print(f"  Control docs: {agg['orig_omit_rate_ctrl']:.3f}")
    print(f"  CARE    docs: {agg['orig_omit_rate_care']:.3f}")
    print()
    print("Per-clinician (omit_rate on edited summary):")
    for r in per_clin:
        print(f"  {r['label']} ({r['name']:<7}): "
              f"ctrl={r['clin_omit_ctrl']:.3f} (n={r['n_ctrl']})  "
              f"CARE={r['clin_omit_care']:.3f} (n={r['n_care']})  "
              f"Δ={r['clin_omit_care']-r['clin_omit_ctrl']:+.3f}")
    print(f"\nSaved: {SUMMARY_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip GPT-5 calls, analyze whatever is in the cache.")
    args = ap.parse_args()

    if not args.analyze_only:
        score(cache_only=False)
    analyze()


if __name__ == "__main__":
    main()
