#!/usr/bin/env python3
"""
Robust harm-weighted omission analysis.

Loads source sentences surfaced as omissions by Controller B at a given alpha
(default 0.15) and classifies each by AHRQ harm level using an LLM judge.

Improvements over the original version:
- Retries empty / malformed LLM responses
- More robust parsing of label outputs
- Optional larger context budgets for source and summary
- Optional local sentence window around omitted sentences
- Fallback from batch mode to single-item mode if batch parsing fails
- Better checkpointing and diagnostics

Usage:
    python analyze_omission_harm.py \
        --input /path/to/calibrated_scores.jsonl \
        --thresholds /path/to/conformal_thresholds.json \
        --dataset aci \
        --split test \
        --model gpt-5 \
        --output /path/to/harm_analysis.jsonl

Thresholds (tau, gamma) can be specified three ways (in priority order):
  1. --tau / --gamma directly
  2. --thresholds pointing to a conformal_thresholds.json (Phase 3 output)
  3. --alpha-sweep pointing to an alpha_sweep_results.json + --alpha

AHRQ harm categories (adapted for omissions):
  A — Death
  B — Severe harm
  C — Moderate harm
  D — Mild harm
  E — No harm
  F — Unknown
"""

import argparse
import ast
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from care import config
from care.utils import LLMClient, load_jsonl, setup_logging

logger = setup_logging("HarmAnalysis")


# ============================================================================
# Prompting
# ============================================================================

def get_ahrq_system_prompt(dataset: str) -> str:
    from care.prompts import get_domain_spec
    spec = get_domain_spec(dataset)
    return f"""You are a clinical patient safety expert evaluating the harm potential of omissions in {spec['output_type']}s.

You will be given a {spec['source_type']} and one or more source sentences that may be omitted from the generated {spec['output_type']}.
For each target sentence, classify the worst plausible patient harm that could result if this information were truly left out of the {spec['output_type']}.

Categories:

A — Death:
Omitting this information could directly contribute to patient death.

B — Severe harm:
Omitting this could cause significant bodily or psychological injury that substantially interferes with function or quality of life.

C — Moderate harm:
Omitting this could adversely affect function or quality of life, but not to the level of severe harm.

D — Mild harm:
Omitting this could cause minimal symptoms, or harm limited to additional treatment, monitoring, or slightly extended length of stay.

E — No harm:
Omitting this would not cause patient harm. The information is background, redundant, already captured elsewhere, or not clinically actionable.

F — Unknown:
There is insufficient context to determine harm potential.

Return ONLY a JSON array with exactly one label per target sentence.
Valid labels: "A", "B", "C", "D", "E", "F"
Do not include explanations.
"""


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_numbered_lines(sentences: List[str], include_indices: Optional[set] = None) -> List[str]:
    lines = []
    for i, s in enumerate(sentences):
        if include_indices is None or i in include_indices:
            lines.append(f"{i+1}. {s}")
    return lines


def collect_neighbor_indices(target_indices: List[int], total_n: int, window: int) -> set:
    keep = set()
    for idx in target_indices:
        lo = max(0, idx - window)
        hi = min(total_n - 1, idx + window)
        keep.update(range(lo, hi + 1))
    return keep


def truncate_lines_by_budget(lines: List[str], char_budget: int) -> Tuple[List[str], bool]:
    out = []
    total = 0
    truncated = False
    for line in lines:
        add_len = len(line) + 1
        if total + add_len <= char_budget:
            out.append(line)
            total += add_len
        else:
            truncated = True
            break
    return out, truncated


def make_source_block(
    source_sentences: List[str],
    target_indices: List[int],
    source_char_budget: int,
    sentence_window: Optional[int],
    always_include_targets: bool = True,
) -> str:
    """
    Build source context block.

    Modes:
    - sentence_window is not None: include local neighborhoods around targets
    - sentence_window is None: allow larger global source context up to budget
    """
    if sentence_window is not None:
        keep = collect_neighbor_indices(target_indices, len(source_sentences), sentence_window)
        lines = build_numbered_lines(source_sentences, include_indices=keep)
        kept, truncated = truncate_lines_by_budget(lines, source_char_budget)

        # Ensure target sentences are always preserved
        if always_include_targets:
            present_targets = set()
            for line in kept:
                try:
                    sent_num = int(line.split(".", 1)[0]) - 1
                    present_targets.add(sent_num)
                except Exception:
                    pass

            missing_targets = [i for i in target_indices if i not in present_targets]
            if missing_targets:
                extra_lines = [f"{i+1}. {source_sentences[i]}" for i in missing_targets]
                kept.extend(extra_lines)

        text = "\n".join(kept)
        if truncated:
            text += "\n[Additional local source context truncated to fit budget.]"
        return text

    # Global mode: include as much of the full source as possible,
    # but never lose the target sentences.
    lines = []
    total = 0
    target_set = set(target_indices)
    for i, s in enumerate(source_sentences):
        line = f"{i+1}. {s}"
        needed = i in target_set
        add_len = len(line) + 1

        if needed or total + add_len <= source_char_budget:
            lines.append(line)
            if not needed:
                total += add_len
        else:
            lines.append(f"{i+1}. [truncated]")
    return "\n".join(lines)


def make_summary_block(
    generated_sentences: List[str],
    summary_char_budget: int,
) -> str:
    lines = build_numbered_lines(generated_sentences)
    kept, truncated = truncate_lines_by_budget(lines, summary_char_budget)
    text = "\n".join(kept)
    if truncated:
        text += "\n[Additional summary context truncated to fit budget.]"
    return text


def get_ahrq_user_prompt(
    items: List[Dict[str, Any]],
    dataset: str,
    source_char_budget: int = 12000,
    sentence_window: Optional[int] = None,
) -> str:
    """
    Build prompt with configurable context size.

    sentence_window:
      - None: use larger global source context up to source_char_budget
      - integer k: include +/- k neighboring source sentences around targets
    """
    from care.prompts import get_domain_spec

    spec = get_domain_spec(dataset)
    source_sentences = items[0]["source_sentences"]
    target_indices = [item["sent_idx"] for item in items]

    source_text = make_source_block(
        source_sentences=source_sentences,
        target_indices=target_indices,
        source_char_budget=source_char_budget,
        sentence_window=sentence_window,
        always_include_targets=True,
    )

    omitted_lines = []
    for i, item in enumerate(items):
        omitted_lines.append(
            f'{i+1}. "{item["sentence"]}"'
        )
    omitted_text = "\n".join(omitted_lines)

    return f"""{spec['source_type']} context (numbered sentences):
{source_text}

Target source sentences that may be omitted from the generated {spec['output_type']}:
{omitted_text}

Return a JSON array with exactly {len(items)} labels, one per target sentence.
Allowed labels only: "A", "B", "C", "D", "E", "F"
"""


# ============================================================================
# Parsing
# ============================================================================

VALID_LABELS = {"A", "B", "C", "D", "E", "F"}


def _normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if s in VALID_LABELS else None


def parse_harm_labels(response: str, n_expected: int) -> List[str]:
    """
    Robust parser that handles:
    - exact JSON arrays: ["A", "C"]
    - python-style arrays: ['A', 'C']
    - objects containing arrays
    - loose text with A/B/C labels in order
    """
    if not response or not response.strip():
        logger.warning("Could not parse harm labels: empty response")
        return ["PARSE_ERROR"] * n_expected

    text = response.strip()

    # Strip fenced code blocks
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # 1. Strict JSON list
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            labels = [_normalize_label(x) for x in parsed]
            if len(labels) == n_expected and all(x is not None for x in labels):
                return labels
        if isinstance(parsed, dict):
            # Try common keys
            for key in ["labels", "harm_labels", "output", "predictions"]:
                if key in parsed and isinstance(parsed[key], list):
                    labels = [_normalize_label(x) for x in parsed[key]]
                    if len(labels) == n_expected and all(x is not None for x in labels):
                        return labels
    except Exception:
        pass

    # 2. Python literal list
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            labels = [_normalize_label(x) for x in parsed]
            if len(labels) == n_expected and all(x is not None for x in labels):
                return labels
    except Exception:
        pass

    # 3. Extract the first [...] block and try again
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                labels = [_normalize_label(x) for x in parsed]
                if len(labels) == n_expected and all(x is not None for x in labels):
                    return labels
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, list):
                    labels = [_normalize_label(x) for x in parsed]
                    if len(labels) == n_expected and all(x is not None for x in labels):
                        return labels
            except Exception:
                pass

    # 4. Regex fallback: quoted labels
    found = re.findall(r'"([A-Fa-f])"', text)
    if len(found) >= n_expected:
        labels = [x.upper() for x in found[:n_expected]]
        if all(x in VALID_LABELS for x in labels):
            return labels

    # 5. Regex fallback: bare labels
    found = re.findall(r"\b([A-Fa-f])\b", text)
    if len(found) >= n_expected:
        labels = [x.upper() for x in found[:n_expected]]
        if all(x in VALID_LABELS for x in labels):
            return labels

    logger.warning(
        f"Could not parse harm labels from response (expected {n_expected}): {text[:500]}"
    )
    return ["PARSE_ERROR"] * n_expected


# ============================================================================
# Threshold Loading
# ============================================================================

def load_thresholds(
    thresholds_file: Optional[Path],
    alpha_sweep_file: Optional[Path],
    alpha: float,
    tau_override: Optional[float],
    gamma_override: Optional[float],
) -> Tuple[float, float]:
    if tau_override is not None and gamma_override is not None:
        logger.info(f"Using directly specified thresholds: tau={tau_override}, gamma={gamma_override}")
        return tau_override, gamma_override

    if thresholds_file is not None:
        with open(thresholds_file) as f:
            thresholds = json.load(f)
        tau = thresholds["omission"]["tau"]
        gamma = thresholds["omission"]["gamma"]
        alpha_in_file = thresholds.get("config", {}).get("alpha_omit", "?")
        logger.info(
            f"Loaded thresholds from {thresholds_file}: tau={tau}, gamma={gamma} (alpha={alpha_in_file})"
        )
        return tau, gamma

    if alpha_sweep_file is not None:
        with open(alpha_sweep_file) as f:
            sweep = json.load(f)
        best = min(sweep, key=lambda r: abs(r["alpha"] - alpha))
        tau = best["omission"]["tau"]
        gamma = best["omission"]["gamma"]
        logger.info(
            f"From alpha_sweep at alpha={best['alpha']:.2f}: tau={tau}, gamma={gamma}"
        )
        return tau, gamma

    raise ValueError(
        "Provide thresholds via --thresholds, --alpha-sweep, or --tau/--gamma directly."
    )


# ============================================================================
# Surfaced omissions
# ============================================================================

def get_surfaced_omissions(
    documents: List[Dict[str, Any]],
    tau: float,
    gamma: float,
) -> List[Dict[str, Any]]:
    """
    Surfaced set: {u : imp_prob >= tau AND non_cov_probs >= gamma}
    """
    surfaced = []
    for doc in documents:
        imp_probs = np.array(doc["importance_probs"])
        cov_probs = np.array(doc["coverage_probs"])
        imp_labels = np.array(doc["importance_labels"])
        cov_labels = np.array(doc["coverage_labels"])
        source_sents = doc["source_sentences"]
        generated_sents = doc.get("generated_sentences", [])

        # corrected calc for surfaced omissions
        non_cov_probs = 1.0 - cov_probs
        mask = (imp_probs >= tau) & (non_cov_probs >= gamma)

        for i, is_surfaced in enumerate(mask):
            if not is_surfaced:
                continue

            is_true_omission = bool(imp_labels[i] == 1 and cov_labels[i] == 0)

            surfaced.append({
                "doc_id": doc.get("doc_id"),
                "sent_idx": i,
                "sentence": source_sents[i],
                "source_sentences": source_sents,
                "generated_sentences": generated_sents,
                "imp_prob": float(imp_probs[i]),
                "cov_prob": float(cov_probs[i]),
                "imp_label": int(imp_labels[i]),
                "cov_label": int(cov_labels[i]),
                "is_true_omission": is_true_omission,
                "harm_label": None,
                "raw_response": None,
            })

    return surfaced


# ============================================================================
# LLM helpers
# ============================================================================

def call_llm_with_retries(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    n_expected: int,
    max_attempts: int = 3,
    retry_sleep_sec: float = 2.0,
) -> Tuple[List[str], str]:
    """
    Returns (labels, raw_response).
    Retries on empty response or parse failure using exponential backoff with jitter.
    retry_sleep_sec is the base delay; actual delay on attempt k is retry_sleep_sec * 2^(k-1) + jitter.
    """
    import random

    last_response = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.generate(system_prompt, user_prompt)
            response = response if response is not None else ""
            last_response = response

            if not response.strip():
                logger.warning(f"Empty response from LLM on attempt {attempt}/{max_attempts}")
            else:
                labels = parse_harm_labels(response, n_expected)
                if all(lbl != "PARSE_ERROR" for lbl in labels):
                    return labels, response

                logger.warning(
                    f"Parse failed on attempt {attempt}/{max_attempts}; response preview: {response[:300]}"
                )
        except Exception as e:
            logger.error(f"LLM call failed on attempt {attempt}/{max_attempts}: {e}")

        if attempt < max_attempts:
            delay = retry_sleep_sec * (2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})...")
            time.sleep(delay)

    return ["PARSE_ERROR"] * n_expected, last_response


def classify_one_batch(
    batch: List[Dict[str, Any]],
    llm: LLMClient,
    dataset: str,
    source_char_budget: int,
    sentence_window: Optional[int],
    max_attempts: int,
    retry_sleep_sec: float,
    summary_char_budget: int = 5000,
) -> Tuple[List[str], str]:
    system_prompt = get_ahrq_system_prompt(dataset)
    user_prompt = get_ahrq_user_prompt(
        items=batch,
        dataset=dataset,
        source_char_budget=source_char_budget,
        sentence_window=sentence_window,
    )
    logger.debug(f"Prompt chars: {len(system_prompt) + len(user_prompt)}")

    labels, raw_response = call_llm_with_retries(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        n_expected=len(batch),
        max_attempts=max_attempts,
        retry_sleep_sec=retry_sleep_sec,
    )
    return labels, raw_response


def _classify_one_doc(
    doc_id: Any,
    doc_items: List[Dict[str, Any]],
    doc_num: int,
    total_docs: int,
    llm: LLMClient,
    dataset: str,
    batch_size: int,
    source_char_budget: int,
    summary_char_budget: int,
    sentence_window: Optional[int],
    max_attempts: int,
    retry_sleep_sec: float,
    fallback_to_single: bool,
) -> List[Dict[str, Any]]:
    logger.info(
        f"Classifying doc {doc_num}/{total_docs} (doc_id={doc_id}, {len(doc_items)} omissions)..."
    )

    for batch_start in range(0, len(doc_items), batch_size):
        batch = doc_items[batch_start: batch_start + batch_size]
        logger.info(
            f"  doc {doc_id} batch {batch_start // batch_size + 1}: size={len(batch)}, "
            f"sent_idx_range={[x['sent_idx'] for x in batch]}"
        )

        labels, raw_response = classify_one_batch(
            batch=batch,
            llm=llm,
            dataset=dataset,
            source_char_budget=source_char_budget,
            summary_char_budget=summary_char_budget,
            sentence_window=sentence_window,
            max_attempts=max_attempts,
            retry_sleep_sec=retry_sleep_sec,
        )

        batch_failed = any(lbl == "PARSE_ERROR" for lbl in labels)

        if batch_failed and fallback_to_single and len(batch) > 1:
            logger.warning(
                f"Batch parse failed for doc {doc_id}; falling back to single-item classification "
                f"for {len(batch)} items."
            )
            single_labels = []
            for item in batch:
                one_labels, one_raw = classify_one_batch(
                    batch=[item],
                    llm=llm,
                    dataset=dataset,
                    source_char_budget=source_char_budget,
                    summary_char_budget=summary_char_budget,
                    sentence_window=sentence_window,
                    max_attempts=max_attempts,
                    retry_sleep_sec=retry_sleep_sec,
                )
                item["raw_response"] = one_raw
                single_labels.extend(one_labels)
            labels = single_labels
        else:
            for item in batch:
                item["raw_response"] = raw_response

        for item, label in zip(batch, labels):
            item["harm_label"] = label

    return doc_items


def classify_harm_batch(
    items: List[Dict[str, Any]],
    llm: LLMClient,
    dataset: str,
    batch_size: int = 10,
    save_path: Optional[Path] = None,
    source_char_budget: int = 12000,
    summary_char_budget: int = 5000,
    sentence_window: Optional[int] = None,
    max_attempts: int = 3,
    retry_sleep_sec: float = 2.0,
    fallback_to_single: bool = True,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    Classify AHRQ harm levels.

    Groups by document so target sentences share context.
    If a batch fails parsing, optionally falls back to single-item classification.
    """
    items_by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    for item in items:
        items_by_doc.setdefault(item["doc_id"], []).append(item)

    total_docs = len(items_by_doc)
    classified: List[Dict[str, Any]] = []
    checkpoint_lock = threading.Lock()

    def _process_doc(args):
        doc_num, doc_id, doc_items = args
        return _classify_one_doc(
            doc_id=doc_id,
            doc_items=doc_items,
            doc_num=doc_num,
            total_docs=total_docs,
            llm=llm,
            dataset=dataset,
            batch_size=batch_size,
            source_char_budget=source_char_budget,
            summary_char_budget=summary_char_budget,
            sentence_window=sentence_window,
            max_attempts=max_attempts,
            retry_sleep_sec=retry_sleep_sec,
            fallback_to_single=fallback_to_single,
        )

    doc_args = [
        (doc_num, doc_id, doc_items)
        for doc_num, (doc_id, doc_items) in enumerate(items_by_doc.items(), 1)
    ]

    effective_workers = min(max_workers, total_docs)
    logger.info(f"Classifying {total_docs} documents with {effective_workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_process_doc, args): args for args in doc_args}
        for future in as_completed(futures):
            doc_items = future.result()
            with checkpoint_lock:
                classified.extend(doc_items)
                if save_path is not None:
                    _save_checkpoint(classified, save_path)

    return classified


# ============================================================================
# Serialization / stats
# ============================================================================

def items_to_doc_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    for item in items:
        by_doc.setdefault(item["doc_id"], []).append(item)

    records = []
    for doc_id, doc_items in by_doc.items():
        doc_items_sorted = sorted(doc_items, key=lambda x: x["sent_idx"])
        records.append({
            "doc_id": doc_id,
            "sent_indices": [x["sent_idx"] for x in doc_items_sorted],
            "sentences": [x["sentence"] for x in doc_items_sorted],
            "harm_labels": [x["harm_label"] for x in doc_items_sorted],
            "imp_probs": [x["imp_prob"] for x in doc_items_sorted],
            "cov_probs": [x["cov_prob"] for x in doc_items_sorted],
            "is_true_omission": [x["is_true_omission"] for x in doc_items_sorted],
        })
    return records


def _save_checkpoint(items: List[Dict[str, Any]], path: Path):
    records = items_to_doc_records(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def compute_breakdown(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    label_counts = Counter(item["harm_label"] for item in items)

    true_omission_items = [i for i in items if i["is_true_omission"]]
    true_omission_counts = Counter(item["harm_label"] for item in true_omission_items)

    def pct(n, denom):
        return round(100 * n / denom, 1) if denom > 0 else 0.0

    breakdown = {
        "total_surfaced": total,
        "total_true_omissions": len(true_omission_items),
        "all_surfaced": {
            label: {"count": cnt, "pct": pct(cnt, total)}
            for label, cnt in sorted(label_counts.items())
        },
        "true_omissions_only": {
            label: {"count": cnt, "pct": pct(cnt, len(true_omission_items))}
            for label, cnt in sorted(true_omission_counts.items())
        },
    }
    return breakdown


AHRQ_LABEL_DESCRIPTIONS = {
    "A": "Death",
    "B": "Severe harm",
    "C": "Moderate harm",
    "D": "Mild harm",
    "E": "No harm",
    "F": "Unknown",
    "PARSE_ERROR": "Parse error",
    "ERROR": "API error",
}

AHRQ_LABEL_ORDER = ["A", "B", "C", "D", "E", "F", "PARSE_ERROR", "ERROR"]


def print_breakdown(breakdown: Dict[str, Any], alpha: float, tau: float, gamma: float):
    print()
    print("=" * 68)
    print(f"AHRQ Harm Breakdown | alpha={alpha}, tau={tau}, gamma={gamma}")
    print("=" * 68)
    print(f"Total sentences surfaced as omissions : {breakdown['total_surfaced']}")
    print(f"Of which oracle-confirmed omissions  : {breakdown['total_true_omissions']}")
    print()
    print("--- All Surfaced Omissions ---")
    print(f"  {'Label':<11} {'Description':<16} {'Count':>6} {'Pct':>6}")
    print(f"  {'-'*11} {'-'*16} {'-'*6} {'-'*6}")
    for label in AHRQ_LABEL_ORDER:
        entry = breakdown["all_surfaced"].get(label)
        if entry:
            desc = AHRQ_LABEL_DESCRIPTIONS.get(label, label)
            print(f"  {label:<11} {desc:<16} {entry['count']:>6} {entry['pct']:>5.1f}%")
    print()
    print("--- Oracle-Confirmed True Omissions Only ---")
    print(f"  {'Label':<11} {'Description':<16} {'Count':>6} {'Pct':>6}")
    print(f"  {'-'*11} {'-'*16} {'-'*6} {'-'*6}")
    for label in AHRQ_LABEL_ORDER:
        entry = breakdown["true_omissions_only"].get(label)
        if entry:
            desc = AHRQ_LABEL_DESCRIPTIONS.get(label, label)
            print(f"  {label:<11} {desc:<16} {entry['count']:>6} {entry['pct']:>5.1f}%")
    print("=" * 68)
    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify AHRQ harm level for omissions flagged at chosen omission thresholds."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only compute surfaced omissions and recall; do not run harm classification.",
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Calibrated scores JSONL (Phase 2 output).",
    )
    parser.add_argument(
        "--dataset",
        choices=["aci", "meq", "bhc", "cxr", "pubmed", "omop"],
        required=True,
        help="Dataset name for domain-specific prompting.",
    )

    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="conformal_thresholds.json from Phase 3.",
    )
    parser.add_argument(
        "--alpha-sweep",
        type=Path,
        default=None,
        dest="alpha_sweep",
        help="alpha_sweep_results.json; use with --alpha.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Direct override for importance threshold tau.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Direct override for non-coverage threshold gamma.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.15,
        help="Alpha level to extract from alpha_sweep. Default: 0.15",
    )

    parser.add_argument(
        "--split",
        choices=["test", "calibration", "all"],
        default="all",
        help="Which split to analyze.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name. Default: JUDGE_MODEL from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Sentences per LLM call. Default: 5",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retries per batch. Default: 3",
    )
    parser.add_argument(
        "--retry-sleep-sec",
        type=float,
        default=2.0,
        help="Base sleep (seconds) between retries; actual delay uses exponential backoff: base * 2^(attempt-1) + jitter. Default: 2.0",
    )
    parser.add_argument(
        "--no-fallback-to-single",
        action="store_true",
        help="Disable fallback from failed batch parsing to single-item classification.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads for document classification. Default: 1",
    )

    parser.add_argument(
        "--source-char-budget",
        type=int,
        default=1000000000,
        help="Character budget for source context. Default: 16000",
    )
    parser.add_argument(
        "--summary-char-budget",
        type=int,
        default=1000000000,
        help="Character budget for summary context. Default: 6000",
    )
    parser.add_argument(
        "--sentence-window",
        type=int,
        default=None,
        help="If set, include only +/- this many source sentences around each target sentence. "
             "If omitted, use broader global source context up to the source budget.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file. Default: harm_analysis.jsonl next to input",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from partial JSONL output.",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep checkpoint file after successful completion.",
    )

    args = parser.parse_args()

    output_path = args.output or (args.input.parent / "harm_analysis.jsonl")
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")

    config.configure_dataset(args.dataset)

    tau, gamma = load_thresholds(
        thresholds_file=args.thresholds,
        alpha_sweep_file=args.alpha_sweep,
        alpha=args.alpha,
        tau_override=args.tau,
        gamma_override=args.gamma,
    )
    logger.info(f"Using omission thresholds: tau={tau}, gamma={gamma}")

    logger.info(f"Loading documents from {args.input}...")
    all_docs = load_jsonl(args.input)
    all_docs = [d for d in all_docs if d is not None]
    logger.info(f"Loaded {len(all_docs)} total documents")

    if args.split != "all":
        try:
            from care.config import filter_documents_by_split
            docs = filter_documents_by_split(all_docs, args.split)
            logger.info(f"Filtered to {args.split} split: {len(docs)} documents")
        except FileNotFoundError:
            logger.warning("split_indices.json not found; using all documents")
            docs = all_docs
    else:
        docs = all_docs

    logger.info(f"Collecting omissions surfaced at tau={tau}, gamma={gamma}...")
    surfaced = get_surfaced_omissions(docs, tau, gamma)
    logger.info(f"Found {len(surfaced)} surfaced omissions across {len(docs)} documents")

    total_true = 0
    surfaced_true = 0

    for doc in docs:
        imp_labels = np.array(doc["importance_labels"])
        cov_labels = np.array(doc["coverage_labels"])
        imp_probs = np.array(doc["importance_probs"])
        cov_probs = np.array(doc["coverage_probs"])

        true_omissions = (imp_labels == 1) & (cov_labels == 0)
        non_cov_probs = 1.0 - cov_probs
        surfaced_mask = (imp_probs >= tau) & (non_cov_probs >= gamma)

        total_true += int(true_omissions.sum())
        surfaced_true += int((true_omissions & surfaced_mask).sum())

    recall = surfaced_true / total_true if total_true else float("nan")

    print()
    print("Omission surfaced-set check")
    print("=" * 50)
    print(f"docs                 : {len(docs)}")
    print(f"tau, gamma           : {tau}, {gamma}")
    print(f"surfaced total       : {len(surfaced)}")
    print(f"true omissions total : {total_true}")
    print(f"true surfaced        : {surfaced_true}")
    print(f"omission recall      : {recall:.4f} ({100*recall:.1f}%)")
    print("=" * 50)
    print()

    if args.check_only:
        return
        

    if len(surfaced) == 0:
        print("No omissions surfaced at these thresholds. Try lowering tau or gamma.")
        return

    already_done: Dict[Tuple[Any, int], str] = {}
    if args.resume and args.resume.exists():
        logger.info(f"Resuming from checkpoint {args.resume}...")
        with open(args.resume) as f:
            for line in f:
                rec = json.loads(line)
                for idx, label in zip(rec["sent_indices"], rec["harm_labels"]):
                    if label and label not in ("PARSE_ERROR", "ERROR"):
                        already_done[(rec["doc_id"], idx)] = label
        logger.info(f"Loaded {len(already_done)} pre-classified items")

    remaining = []
    for item in surfaced:
        key = (item["doc_id"], item["sent_idx"])
        if key in already_done:
            item["harm_label"] = already_done[key]
        else:
            remaining.append(item)

    logger.info(f"Need to classify {len(remaining)} items ({len(surfaced) - len(remaining)} cached)")

    model = args.model or config.JUDGE_MODEL
    logger.info(f"Using model: {model}")

    llm = LLMClient(
        model=model,
        api_key=config.SECUREGPT_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.get_azure_endpoint(model),
        max_retries=config.MAX_RETRIES,
        timeout=config.TIMEOUT,
        temperature=0.0,
        max_tokens=4096,
        rate_limit_delay=config.RATE_LIMIT_DELAY,
    )

    if remaining:
        classify_harm_batch(
            items=remaining,
            llm=llm,
            dataset=args.dataset,
            batch_size=args.batch_size,
            save_path=checkpoint_path,
            source_char_budget=args.source_char_budget,
            summary_char_budget=args.summary_char_budget,
            sentence_window=args.sentence_window,
            max_attempts=args.max_attempts,
            retry_sleep_sec=args.retry_sleep_sec,
            fallback_to_single=not args.no_fallback_to_single,
            max_workers=args.workers,
        )

    all_classified = surfaced

    breakdown = compute_breakdown(all_classified)
    print_breakdown(breakdown, args.alpha, tau, gamma)

    doc_records = items_to_doc_records(all_classified)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in doc_records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Results saved to {output_path} ({len(doc_records)} documents)")

    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()