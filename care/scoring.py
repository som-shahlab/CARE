#!/usr/bin/env python3
"""
Phase 2: Three-Tier Vote-Rate Scoring

This script implements Phase 2 of the conformal risk control pipeline (plan.tex):
1. Load Phase 1 oracle labels (ground truth Y_fact, Y_imp, Y_cov)
2. Query judge m times per sentence with 3-tier response
3. Aggregate with simple averaging: p̂ = (1/m) * Σ Z

Three-Tier Scoring (plan.tex Section 2):
- Query judge m=5 times per sentence
- Each replicate returns Z ∈ {1.0, 0.5, 0.0}:
  * Factuality: SUPPORTED (1.0) / PARTIAL (0.5) / UNSUPPORTED (0.0)
  * Importance: ESSENTIAL (1.0) / RELEVANT (0.5) / NOT_RELEVANT (0.0)
  * Coverage: COVERED (1.0) / PARTIAL (0.5) / OMITTED (0.0)
- Aggregate: p̂ = (1/m) * Σ Z
- With m=5 and 3 tiers, possible values: {0.0, 0.1, 0.2, ..., 1.0} (11 values)

Output per document:
{
    "doc_id": <int>,
    "generated_sentences": [<str>, ...],
    "source_sentences": [<str>, ...],
    "factuality_labels": [0/1, ...],         # Y_fact from Phase 1
    "importance_labels": [0/1, ...],         # Y_imp from Phase 1
    "coverage_labels": [0/1, ...],           # Y_cov from Phase 1
    "omission_labels": [0/1, ...],           # Omission = Y_imp AND NOT Y_cov
    "factuality_probs": [0-1, ...],          # p̂_fact via vote-rate averaging
    "importance_probs": [0-1, ...],          # p̂_imp via vote-rate averaging
    "coverage_probs": [0-1, ...],            # p̂_cov via vote-rate averaging
}
"""

import argparse
import sys
import json
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for imports

from care import config, prompts
from care.utils import (
    LLMClient,
    create_llm_client,
    load_jsonl,
    save_jsonl,
    save_json,
    setup_logging,
    LocalLLMClient
)
from openai import BadRequestError

logger = setup_logging('Phase2', config.LOG_LEVEL)
skip_log = config.PHASE2_DIR / "skipped_docs.jsonl"
prompt_config = None

# Thread-local storage for judge clients
_thread_local = threading.local()
_parallel_log_lock = threading.Lock()


def get_thread_judge():
    """One judge client per thread (LocalLLMClient shares a singleton model internally)."""
    if getattr(_thread_local, "judge", None) is None:
        _thread_local.judge = create_llm_client(
            model=config.JUDGE_MODEL,
            api_key=config.SECUREGPT_API_KEY,
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.get_azure_endpoint(config.JUDGE_MODEL),
            max_tokens=config.JUDGE_MAX_TOKENS,
            rate_limit_delay=config.RATE_LIMIT_DELAY,
        )
    return _thread_local.judge


def log_skipped_doc(output_path: Path, doc_id: int, stage: str, error: Exception):
    """Log a skipped document due to errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "doc_id": doc_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_str": str(error)[:5000],
    }
    with output_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ============================================================================
# Response Parsing (3-tier and binary)
# ============================================================================

class CountMismatchError(ValueError):
    """Raised when LLM returns different count than expected."""
    pass


# 3-tier response mappings
TRIAGE_MAPPINGS = {
    # Factuality
    "SUPPORTED": 1.0, "SUPPORT": 1.0, "YES": 1.0, "Y": 1.0,
    "PARTIAL": 0.5, "AMBIGUOUS": 0.5, "UNCLEAR": 0.5,
    "UNSUPPORTED": 0.0, "UNSUPPORT": 0.0, "NO": 0.0, "N": 0.0, "CONTRADICTION": 0.0,
    # Importance
    "ESSENTIAL": 1.0, "IMPORTANT": 1.0,
    "RELEVANT": 0.5,
    "NOT_RELEVANT": 0.0, "NOTRELEVANT": 0.0, "NOT RELEVANT": 0.0, "NOISE": 0.0, "TANGENTIAL": 0.0,
    # Coverage
    "COVERED": 1.0,
    "GISTED": 0.5,
    "OMITTED": 0.0, "MISSING": 0.0,
}

# 5-point importance scale (v2: compression-aware)
# Maps to {0.0, 0.25, 0.50, 0.75, 1.0} — with m=5 votes gives 21 distinct averages
IMPORTANCE_V2_MAPPINGS = {
    "MUST_INCLUDE": 1.0, "MUSTINCLUDE": 1.0, "MUST": 1.0,
    "SHOULD_INCLUDE": 0.75, "SHOULDINCLUDE": 0.75, "SHOULD": 0.75,
    "NICE_TO_HAVE": 0.5, "NICETOHAVE": 0.5, "NICE": 0.5,
    "SAFE_TO_OMIT": 0.25, "SAFETOOMIT": 0.25, "SAFE": 0.25,
    "NOT_RELEVANT": 0.0, "NOTRELEVANT": 0.0, "NOT RELEVANT": 0.0, "IRRELEVANT": 0.0,
    # Fallbacks if model uses v1 labels
    "ESSENTIAL": 1.0, "CRITICAL": 1.0,
    "IMPORTANT": 0.75,
    "RELEVANT": 0.5,
    "TANGENTIAL": 0.25,
    "NOISE": 0.0,
}


def parse_triage_responses(response: str, expected_count: int, mapping: dict = None) -> List[float]:
    """
    Parse JSON array of tier responses to scores.

    Args:
        response: Raw LLM response containing JSON array
        expected_count: Expected number of responses
        mapping: Label-to-score mapping (default: TRIAGE_MAPPINGS for 3-tier)

    Returns:
        List of scores

    Raises:
        ValueError: If parsing fails
        CountMismatchError: If count doesn't match expected
    """
    if mapping is None:
        mapping = TRIAGE_MAPPINGS

    response = response.strip()

    # Extract first complete JSON array using bracket-depth matching.
    # Models may append explanation text with brackets after the array,
    # so greedy regex fails. Bracket-depth finds the first balanced [...].
    json_str = response
    start = response.find('[')
    if start != -1:
        depth = 0
        for i in range(start, len(response)):
            if response[i] == '[':
                depth += 1
            elif response[i] == ']':
                depth -= 1
                if depth == 0:
                    json_str = response[start:i+1]
                    break

    try:
        responses = json.loads(json_str)

        if not isinstance(responses, list):
            raise ValueError(f"Expected list, got {type(responses)}")

        # Convert to triage scores
        scores = []
        for item in responses:
            if isinstance(item, str):
                item_clean = item.strip().upper().replace("-", "_")
                if item_clean in mapping:
                    scores.append(mapping[item_clean])
                else:
                    # Try prefix matching
                    matched = False
                    for key, val in mapping.items():
                        if item_clean.startswith(key):
                            scores.append(val)
                            matched = True
                            break
                    if not matched:
                        logger.warning(f"Unknown triage response: {item}. Defaulting to 0.5.")
                        scores.append(0.5)
            elif isinstance(item, (int, float)):
                # Handle numeric responses directly
                if item >= 0.75:
                    scores.append(1.0)
                elif item >= 0.25:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            else:
                logger.warning(f"Invalid response type: {type(item)}. Defaulting to 0.5.")
                scores.append(0.5)

        if len(scores) != expected_count:
            raise CountMismatchError(
                f"Expected {expected_count} scores, got {len(scores)}. "
                f"Response may be truncated or malformed."
            )

        return scores

    except CountMismatchError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse triage responses: {e}")


def parse_binary_responses(response: str, expected_count: int) -> List[int]:
    """
    Parse JSON array of binary responses (YES/NO) to votes (1/0).
    Kept for backward compatibility with Oracle prompts.
    """
    response = response.strip()

    json_str = response
    start = response.find('[')
    if start != -1:
        depth = 0
        for i in range(start, len(response)):
            if response[i] == '[':
                depth += 1
            elif response[i] == ']':
                depth -= 1
                if depth == 0:
                    json_str = response[start:i+1]
                    break

    try:
        responses = json.loads(json_str)

        if not isinstance(responses, list):
            raise ValueError(f"Expected list, got {type(responses)}")

        votes = []
        for item in responses:
            if isinstance(item, str):
                item_upper = item.strip().upper()
                if item_upper.startswith("YES") or item_upper == "Y":
                    votes.append(1)
                elif item_upper.startswith("NO") or item_upper == "N":
                    votes.append(0)
                else:
                    logger.warning(f"Unknown response: {item}. Defaulting to 0.")
                    votes.append(0)
            elif isinstance(item, (int, float)):
                votes.append(1 if item > 0.5 else 0)
            elif isinstance(item, bool):
                votes.append(1 if item else 0)
            else:
                logger.warning(f"Invalid response type: {type(item)}. Defaulting to 0.")
                votes.append(0)

        if len(votes) != expected_count:
            raise CountMismatchError(
                f"Expected {expected_count} votes, got {len(votes)}. "
                f"Response may be truncated or malformed."
            )

        return votes

    except CountMismatchError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse binary responses: {e}")


# ============================================================================
# Vote-Rate Scoring Functions
# ============================================================================

def simple_average(scores: List[float], m: int) -> float:
    """
    Simple vote-rate averaging for 3-tier scores.

    p̂ = (1/m) * Σ Z

    Args:
        scores: List of triage scores (0.0, 0.5, or 1.0)
        m: Number of replicates

    Returns:
        Average score in [0, 1]
    """
    return sum(scores) / m


def jeffreys_smoothing(votes: List[int], m: int) -> float:
    """
    Apply Jeffreys smoothing to vote counts (legacy, for binary scoring).

    p̂ = (sum_votes + 0.5) / (m + 1)
    """
    return (sum(votes) + 0.5) / (m + 1)


def score_factuality_vote_rate(
    summary_sentences: List[str],
    source_text: str,
    judge: LLMClient,
    m: int = 5,
    max_retries: int = 3,
    domain_specific: bool = False,
) -> Tuple[List[float], List[List[float]]]:
    """
    Score factuality using 3-tier vote-rate with simple averaging.

    Query judge m times per batch with 3-tier responses:
    - SUPPORTED (1.0) / PARTIAL (0.5) / UNSUPPORTED (0.0)
    Aggregate: p̂_fact(v) = (1/m) * Σ Z

    Args:
        summary_sentences: List of summary sentences V(X)
        source_text: Source dialogue X
        judge: LLM client for judge scoring
        m: Number of replicates (default 5)
        max_retries: Max retries on count mismatch (default 3)
        domain_specific: If True, use domain-specific factuality triage prompt

    Returns:
        Tuple of (probs, raw_scores):
        - probs: List of averaged probabilities in [0, 1]
        - raw_scores: List of per-sentence replicate scores, raw_scores[i] = [rep1, rep2, ..., repm]
    """
    if not summary_sentences:
        return [], []

    # Storage for scores: scores[sentence_idx] = list of triage scores
    all_scores = [[] for _ in summary_sentences]

    # Process in chunks to avoid token limits
    chunk_size = config.JUDGE_BATCH_SIZE

    for i in range(0, len(summary_sentences), chunk_size):
        chunk = summary_sentences[i:i + chunk_size]

        if domain_specific:
            system_prompt = prompt_config.custom_factuality_triage_system_prompt
        else:
            system_prompt = prompt_config.factuality_triage_system_prompt
        user_prompt = prompt_config.get_factuality_triage_user_prompt(chunk, source_text)

        # Batch all m replicates in one generate_batch call (GPU-efficient for local models)
        responses = judge.generate_batch(
            [(system_prompt, user_prompt)] * m,
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            top_p=config.JUDGE_TOP_P,
        )

        for rep, response in enumerate(responses):
            scores = None
            for attempt in range(max_retries):
                try:
                    scores = parse_triage_responses(response, len(chunk))
                    break
                except CountMismatchError as e:
                    logger.warning(f"Count mismatch rep {rep + 1}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        # re-generate just this replicate on count mismatch
                        response = judge.generate(
                            system_prompt, user_prompt,
                            temperature=config.JUDGE_TEMPERATURE,
                            max_tokens=config.JUDGE_MAX_TOKENS,
                            top_p=config.JUDGE_TOP_P,
                        )
                    continue
                except ValueError as e:
                    logger.error(f"Failed to parse factuality triage chunk {i//chunk_size + 1}, rep {rep + 1}: {e}")
                    break

            if scores is not None and len(scores) == len(chunk):
                for j, score in enumerate(scores):
                    all_scores[i + j].append(score)
            else:
                logger.warning(f"Skipping factuality rep {rep + 1} for chunk {i//chunk_size + 1} - incomplete response")

    # Aggregate with simple averaging using actual score count per sentence
    probs = []
    for scores in all_scores:
        actual_m = len(scores)
        if actual_m == 0:
            raise ValueError("No valid factuality scores collected for sentence - cannot compute probability")
        prob = simple_average(scores, actual_m)
        probs.append(prob)

    return probs, all_scores


def score_importance_vote_rate(
    source_sentences: List[str],
    source_text: str,
    judge: LLMClient,
    m: int = 5,
    max_retries: int = 3,
    importance_prompt_version: str = "v1",
    domain_specific: bool = False,
) -> Tuple[List[float], List[List[float]]]:
    """
    Score importance using vote-rate with simple averaging.

    v1: 3-tier (ESSENTIAL/RELEVANT/NOT_RELEVANT) → {1.0, 0.5, 0.0}
    v2: 5-tier compression-aware (MUST_INCLUDE/.../NOT_RELEVANT) → {1.0, 0.75, 0.5, 0.25, 0.0}

    Args:
        source_sentences: List of source sentences U(X)
        source_text: Full source dialogue X
        judge: LLM client for judge scoring
        m: Number of replicates (default 5)
        max_retries: Max retries on count mismatch (default 3)
        importance_prompt_version: "v1" (3-tier) or "v2" (5-tier compression-aware)
        domain_specific: If True, use domain-specific importance triage prompt

    Returns:
        Tuple of (probs, raw_scores):
        - probs: List of averaged probabilities in [0, 1]
        - raw_scores: List of per-sentence replicate scores, raw_scores[i] = [rep1, rep2, ..., repm]
    """
    if not source_sentences:
        return [], []

    # Select prompts and mapping based on version
    if domain_specific:
        get_system = lambda: prompt_config.custom_importance_triage_system_prompt
        get_user = lambda chunk: prompt_config.get_importance_triage_user_prompt(chunk, source_text)
        parse_mapping = None  # uses default TRIAGE_MAPPINGS (same 3-tier)
    elif importance_prompt_version == "v2":
        get_system = lambda: prompt_config.importance_triage_v2_system_prompt
        get_user = lambda chunk: prompt_config.get_importance_triage_v2_user_prompt(chunk, source_text)
        parse_mapping = IMPORTANCE_V2_MAPPINGS
    elif importance_prompt_version == "fewshot":
        get_system = lambda: prompt_config.importance_triage_fewshot_system_prompt
        get_user = lambda chunk: prompt_config.get_importance_triage_fewshot_user_prompt(chunk, source_text)
        parse_mapping = None  # uses default TRIAGE_MAPPINGS (same 3-tier as v1)
    else:
        get_system = lambda: prompt_config.importance_triage_system_prompt
        get_user = lambda chunk: prompt_config.get_importance_triage_user_prompt(chunk, source_text)
        parse_mapping = None  # uses default TRIAGE_MAPPINGS

    # Storage for scores: scores[sentence_idx] = list of triage scores
    all_scores = [[] for _ in source_sentences]

    chunk_size = config.JUDGE_BATCH_SIZE

    for i in range(0, len(source_sentences), chunk_size):
        chunk = source_sentences[i:i + chunk_size]

        system_prompt = get_system()
        user_prompt = get_user(chunk)

        responses = judge.generate_batch(
            [(system_prompt, user_prompt)] * m,
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            top_p=config.JUDGE_TOP_P,
        )

        for rep, response in enumerate(responses):
            scores = None
            for attempt in range(max_retries):
                try:
                    scores = parse_triage_responses(response, len(chunk), mapping=parse_mapping)
                    break
                except CountMismatchError as e:
                    logger.warning(f"Count mismatch rep {rep + 1}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        response = judge.generate(
                            system_prompt, user_prompt,
                            temperature=config.JUDGE_TEMPERATURE,
                            max_tokens=config.JUDGE_MAX_TOKENS,
                            top_p=config.JUDGE_TOP_P,
                        )
                    continue
                except ValueError as e:
                    logger.error(f"Failed to parse importance triage chunk {i//chunk_size + 1}, rep {rep + 1}: {e}")
                    break

            if scores is not None and len(scores) == len(chunk):
                for j, score in enumerate(scores):
                    all_scores[i + j].append(score)
            else:
                logger.warning(f"Skipping importance rep {rep + 1} for chunk {i//chunk_size + 1} - incomplete response")

    # Aggregate with simple averaging using actual score count per sentence
    probs = []
    for scores in all_scores:
        actual_m = len(scores)
        if actual_m == 0:
            raise ValueError("No valid importance scores collected for sentence - cannot compute probability")
        prob = simple_average(scores, actual_m)
        probs.append(prob)

    return probs, all_scores


def score_coverage_vote_rate(
    source_sentences: List[str],
    generated_summary: str,
    judge: LLMClient,
    m: int = 5,
    max_retries: int = 3,
    coverage_prompt_version: str = "v1",
) -> Tuple[List[float], List[List[float]]]:
    """
    Score coverage using 3-tier vote-rate with simple averaging.

    Query judge m times per batch with 3-tier responses:
    - COVERED (1.0) / PARTIAL (0.5) / OMITTED (0.0)
    Aggregate: p̂_cov(u) = (1/m) * Σ Z

    Args:
        source_sentences: List of source sentences U(X)
        generated_summary: Generated summary Ŝ(X)
        judge: LLM client for judge scoring
        m: Number of replicates (default 5)
        max_retries: Max retries on count mismatch (default 3)
        coverage_prompt_version: Coverage prompt variant (default "v1")

    Returns:
        Tuple of (probs, raw_scores):
        - probs: List of averaged probabilities in [0, 1]
        - raw_scores: List of per-sentence replicate scores, raw_scores[i] = [rep1, rep2, ..., repm]
    """
    if not source_sentences:
        return [], []

    # Storage for scores: scores[sentence_idx] = list of triage scores
    all_scores = [[] for _ in source_sentences]

    chunk_size = config.JUDGE_BATCH_SIZE

    # Select prompt version (extensible for future prompt variants)
    get_system = lambda: prompt_config.coverage_triage_system_prompt
    get_user = lambda chunk: prompt_config.get_coverage_triage_user_prompt(chunk, generated_summary)

    for i in range(0, len(source_sentences), chunk_size):
        chunk = source_sentences[i:i + chunk_size]

        system_prompt = get_system()
        user_prompt = get_user(chunk)

        responses = judge.generate_batch(
            [(system_prompt, user_prompt)] * m,
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            top_p=config.JUDGE_TOP_P,
        )

        for rep, response in enumerate(responses):
            scores = None
            for attempt in range(max_retries):
                try:
                    scores = parse_triage_responses(response, len(chunk))
                    break
                except CountMismatchError as e:
                    logger.warning(f"Count mismatch rep {rep + 1}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        response = judge.generate(
                            system_prompt, user_prompt,
                            temperature=config.JUDGE_TEMPERATURE,
                            max_tokens=config.JUDGE_MAX_TOKENS,
                            top_p=config.JUDGE_TOP_P,
                        )
                    continue
                except ValueError as e:
                    logger.error(f"Failed to parse coverage triage chunk {i//chunk_size + 1}, rep {rep + 1}: {e}")
                    break

            if scores is not None and len(scores) == len(chunk):
                for j, score in enumerate(scores):
                    all_scores[i + j].append(score)
            else:
                logger.warning(f"Skipping coverage rep {rep + 1} for chunk {i//chunk_size + 1} - incomplete response")

    # Aggregate with simple averaging using actual score count per sentence
    probs = []
    for scores in all_scores:
        actual_m = len(scores)
        if actual_m == 0:
            raise ValueError("No valid coverage scores collected for sentence - cannot compute probability")
        prob = simple_average(scores, actual_m)
        probs.append(prob)

    return probs, all_scores


# ============================================================================
# Token probs (logit-readout scorer for local judges)
# ============================================================================
# Single-letter A/B/C labels are used at the model interface so the next
# token after the assistant turn opener is guaranteed to be a single BPE
# piece with no prefix collisions. The output dict is remapped to the
# verbose tier names so downstream consumers see the same keys as the
# original implementation.

# Letters actually scored at the logit position
LETTER_TOKENS = ["A", "B", "C"]

# Verbose tier names exposed in the output dict (downstream key contract)
FACTUALITY_LABEL_TOKENS = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
IMPORTANCE_LABEL_TOKENS = ["ESSENTIAL", "RELEVANT", "NOT_RELEVANT"]
COVERAGE_LABEL_TOKENS = ["COVERED", "PARTIAL", "OMITTED"]


def _remap_letter_probs(letter_probs_list, tier_labels):
    """Remap [{'A': p, 'B': p, 'C': p}, ...] to [{tier_labels[0]: p, ...}, ...]."""
    return [
        {tier_labels[i]: d[letter] for i, letter in enumerate(LETTER_TOKENS)}
        for d in letter_probs_list
    ]


def enrich_with_token_probs(doc, judge, domain_specific=False):
    """Single forward-pass logit-readout token-probs per sentence (LocalLLMClient only).

    For each sentence we run one forward pass under the local judge with a
    dedicated single-label prompt that asks for exactly one of A / B / C,
    then read the next-token logits, slice down to the three letter ids,
    and softmax. The resulting dict (per sentence) is remapped to the
    verbose tier names so callers can index by 'SUPPORTED' / 'PARTIAL' /
    'UNSUPPORTED' (etc.) exactly as before.

    `domain_specific` is currently a no-op for the token-probs path: the
    single-label prompts already inject the domain via spec['role'] /
    spec['output_type']. The flag is retained to keep the call signature
    aligned with score_document.
    """
    if not isinstance(judge, LocalLLMClient):
        return doc

    source_text = doc["source_text"]
    gen_summary = doc["generated_summary"]

    # --- Factuality ---
    fact_sys = prompt_config.factuality_token_probs_system_prompt
    fact_prompts = [
        (fact_sys, prompt_config.get_factuality_token_probs_user_prompt(s, source_text))
        for s in doc["generated_sentences"]
    ]
    fact_letter_probs = judge.batch_forward_token_probs(fact_prompts, LETTER_TOKENS)
    doc["factuality_token_probs"] = _remap_letter_probs(fact_letter_probs, FACTUALITY_LABEL_TOKENS)

    # --- Importance ---
    imp_sys = prompt_config.importance_token_probs_system_prompt
    imp_prompts = [
        (imp_sys, prompt_config.get_importance_token_probs_user_prompt(s, source_text))
        for s in doc["source_sentences"]
    ]
    imp_letter_probs = judge.batch_forward_token_probs(imp_prompts, LETTER_TOKENS)
    doc["importance_token_probs"] = _remap_letter_probs(imp_letter_probs, IMPORTANCE_LABEL_TOKENS)

    # --- Coverage ---
    cov_sys = prompt_config.coverage_token_probs_system_prompt
    cov_prompts = [
        (cov_sys, prompt_config.get_coverage_token_probs_user_prompt(s, gen_summary))
        for s in doc["source_sentences"]
    ]
    cov_letter_probs = judge.batch_forward_token_probs(cov_prompts, LETTER_TOKENS)
    doc["coverage_token_probs"] = _remap_letter_probs(cov_letter_probs, COVERAGE_LABEL_TOKENS)

    return doc

# ============================================================================
# Document Scoring
# ============================================================================


def score_document(
    doc: Dict[str, Any],
    judge: LLMClient,
    m: int = 5,
    coverage_only: bool = False,
    importance_only: bool = False,
    coverage_prompt_version: str = "v1",
    importance_prompt_version: str = "v1",
    domain_specific: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Score all sentences in a document using vote-rate method.

    Args:
        doc: Document with oracle labels from Phase 1
        judge: LLM client for judge scoring
        m: Number of replicates
        coverage_only: If True, skip factuality/importance and keep existing probs
        importance_only: If True, skip factuality/coverage and keep existing probs
        coverage_prompt_version: "v1" or "anchor"
        importance_prompt_version: "v1" (3-tier) or "v2" (5-tier compression-aware)
        domain_specific: If True, use domain-specific factuality/importance prompts

    Returns:
        Document with added vote-rate probabilities, or None if skipped
    """
    doc_id = doc["doc_id"]
    source_text = doc["source_text"]
    generated_sentences = doc["generated_sentences"]
    source_sentences = doc["source_sentences"]
    generated_summary = doc["generated_summary"]

    # Determine which dimensions to score vs reuse from existing doc
    score_fact = not (coverage_only or importance_only)
    score_imp = not coverage_only
    score_cov = not importance_only

    # --- Factuality ---
    if score_fact:
        try:
            factuality_probs, factuality_raw = score_factuality_vote_rate(
                generated_sentences, source_text, judge, m, domain_specific=domain_specific
            )
        except BadRequestError as e:
            msg = str(e)
            if ("content_filter" in msg) or ("ResponsibleAIPolicyViolation" in msg):
                logger.warning(f"Skipping doc {doc_id} due to content filter (factuality).")
                log_skipped_doc(skip_log, doc_id, stage="factuality_vote_rate", error=e)
                return None
            raise
        except ValueError as e:
            logger.warning(f"Skipping doc {doc_id} - factuality scoring failed: {e}")
            log_skipped_doc(skip_log, doc_id, stage="factuality_vote_rate", error=e)
            return None
    else:
        factuality_probs = doc.get('factuality_probs', [])
        factuality_raw = doc.get('factuality_raw_scores', [])

    # --- Importance ---
    if score_imp:
        try:
            importance_probs, importance_raw = score_importance_vote_rate(
                source_sentences, source_text, judge, m,
                importance_prompt_version=importance_prompt_version,
                domain_specific=domain_specific,
            )
        except BadRequestError as e:
            msg = str(e)
            if ("content_filter" in msg) or ("ResponsibleAIPolicyViolation" in msg):
                logger.warning(f"Skipping doc {doc_id} due to content filter (importance).")
                log_skipped_doc(skip_log, doc_id, stage="importance_vote_rate", error=e)
                return None
            raise
        except ValueError as e:
            logger.warning(f"Skipping doc {doc_id} - importance scoring failed: {e}")
            log_skipped_doc(skip_log, doc_id, stage="importance_vote_rate", error=e)
            return None
    else:
        importance_probs = doc.get('importance_probs', [])
        importance_raw = doc.get('importance_raw_scores', [])

    # --- Coverage ---
    if score_cov:
        try:
            coverage_probs, coverage_raw = score_coverage_vote_rate(
                source_sentences, generated_summary, judge, m,
                coverage_prompt_version=coverage_prompt_version,
            )
        except BadRequestError as e:
            msg = str(e)
            if ("content_filter" in msg) or ("ResponsibleAIPolicyViolation" in msg):
                logger.warning(f"Skipping doc {doc_id} due to content filter (coverage).")
                log_skipped_doc(skip_log, doc_id, stage="coverage_vote_rate", error=e)
                return None
            raise
        except ValueError as e:
            logger.warning(f"Skipping doc {doc_id} - coverage scoring failed: {e}")
            log_skipped_doc(skip_log, doc_id, stage="coverage_vote_rate", error=e)
            return None
    else:
        coverage_probs = doc.get('coverage_probs', [])
        coverage_raw = doc.get('coverage_raw_scores', [])

    result = {
        **doc,
        'factuality_probs': factuality_probs,
        'importance_probs': importance_probs,
        'coverage_probs': coverage_probs,
        'factuality_raw_scores': factuality_raw,
        'importance_raw_scores': importance_raw,
        'coverage_raw_scores': coverage_raw,
    }

    result = enrich_with_token_probs(result, judge, domain_specific=domain_specific)

    return result


def _score_document_parallel(
    doc: Dict[str, Any],
    m: int,
    coverage_only: bool = False,
    importance_only: bool = False,
    coverage_prompt_version: str = "v1",
    importance_prompt_version: str = "v1",
    domain_specific: bool = False,
) -> Optional[Dict[str, Any]]:
    """Score a single document in parallel (uses thread-local judge)."""
    judge = get_thread_judge()
    with _parallel_log_lock:
        logger.info(f"  [Thread] Scoring doc {doc['doc_id']}...")
    result = score_document(doc, judge, m,
                            coverage_only=coverage_only,
                            importance_only=importance_only,
                            coverage_prompt_version=coverage_prompt_version,
                            importance_prompt_version=importance_prompt_version,
                            domain_specific=domain_specific)
    if result is not None:
        with _parallel_log_lock:
            logger.info(f"  [Thread] Completed doc {doc['doc_id']}")
    return result


# ============================================================================
# Main Pipeline
# ============================================================================

def run_phase2(
    input_file: Path = config.PHASE1_DIR / 'oracle_labels.jsonl',
    output_dir: Path = config.PHASE2_DIR,
    num_docs: int = None,
    m: int = 5,
    parallel_workers: int = 1,
    coverage_only: bool = False,
    importance_only: bool = False,
    coverage_prompt_version: str = "v1",
    importance_prompt_version: str = "v1",
    domain_specific: bool = False,
):
    """
    Run Phase 2: Vote-rate scoring.

    Args:
        input_file: Path to Phase 1 oracle labels (JSONL)
        output_dir: Directory for output files
        num_docs: Number of documents to process (None = all)
        m: Number of replicates for vote-rate (default: 5)
        parallel_workers: Number of parallel workers (default: 1)
        coverage_only: If True, only re-score coverage (keep existing fact/imp)
        importance_only: If True, only re-score importance (keep existing fact/cov)
        coverage_prompt_version: "v1" or "anchor"
        importance_prompt_version: "v1" (3-tier) or "v2" (5-tier compression-aware)
        domain_specific: If True, use domain-specific factuality/importance prompts
    """
    logger.info("="*80)
    if coverage_only:
        logger.info(f"Phase 2: Coverage-Only Re-scoring (prompt {coverage_prompt_version})")
    elif importance_only:
        logger.info(f"Phase 2: Importance-Only Re-scoring (prompt {importance_prompt_version})")
    else:
        logger.info("Phase 2: Vote-Rate Scoring")
    logger.info("="*80)
    logger.info(f"Replicates (m): {m}")
    # With 3-tier scoring (0, 0.5, 1.0) and m replicates, we get m*2+1 discrete values
    logger.info(f"Possible probability values: {[round(i / (2 * m), 3) for i in range(2 * m + 1)]}")

    # Setup
    config.setup_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'scored_docs.jsonl'

    # Initialize LLM client for judge
    logger.info(f"Initializing judge client...")
    logger.info(f"  Judge: {config.JUDGE_MODEL}")
    logger.info(f"  API Key: {config.SECUREGPT_API_KEY[:10]}...")
    if domain_specific:
        logger.info(f"[PROMPT CHECK] Factuality system prompt:\n{prompt_config.custom_factuality_triage_system_prompt}")
        logger.info(f"[PROMPT CHECK] Importance system prompt:\n{prompt_config.custom_importance_triage_system_prompt}")

    judge = create_llm_client(
        model=config.JUDGE_MODEL,
        api_key=config.SECUREGPT_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.get_azure_endpoint(config.JUDGE_MODEL),
        max_tokens=config.JUDGE_MAX_TOKENS,
        rate_limit_delay=config.RATE_LIMIT_DELAY,
    )

    # Load Phase 1 data
    logger.info(f"Loading Phase 1 oracle labels from {input_file}...")
    documents = load_jsonl(input_file)

    if num_docs:
        documents = documents[:num_docs]
        logger.info(f"Processing first {num_docs} documents")
    else:
        logger.info(f"Processing all {len(documents)} documents")

    # Resume from cache if exists
    scored_cache = output_dir / 'scored_docs.jsonl'
    scored_docs = []
    scored_ids = set()

    if scored_cache.exists():
        scored_docs = load_jsonl(scored_cache)
        scored_ids = {d["doc_id"] for d in scored_docs}
        logger.info(f"Resuming from cache: {len(scored_docs)} docs already scored")

    # Score documents
    docs_to_score = [doc for doc in documents if doc['doc_id'] not in scored_ids]
    logger.info(f"Documents to score: {len(docs_to_score)}")

    if parallel_workers > 1 and docs_to_score:
        # Parallel execution
        logger.info(f"Parallel scoring with {parallel_workers} workers...")
        _cache_write_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_score_document_parallel, doc, m,
                                coverage_only, importance_only,
                                coverage_prompt_version, importance_prompt_version,
                                domain_specific): doc
                for doc in docs_to_score
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
                doc = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        scored_docs.append(result)
                        with _cache_write_lock:
                            with scored_cache.open("a") as f:
                                f.write(json.dumps(result) + "\n")
                except Exception as e:
                    logger.error(f"Failed to score doc {doc['doc_id']}: {e}")
                    log_skipped_doc(skip_log, doc['doc_id'], stage="parallel_vote_rate", error=e)
    else:
        # Sequential execution
        for doc in tqdm(docs_to_score, desc="Scoring"):
            doc_id = doc['doc_id']
            logger.info(f"Scoring doc {doc_id}...")

            result = score_document(doc, judge, m,
                                    coverage_only=coverage_only,
                                    importance_only=importance_only,
                                    coverage_prompt_version=coverage_prompt_version,
                                    importance_prompt_version=importance_prompt_version,
                                    domain_specific=domain_specific)
            if result is not None:
                scored_docs.append(result)
                with scored_cache.open("a") as f:
                    f.write(json.dumps(result) + "\n")

    # Save final results (also write to calibrated_scores.jsonl for backward compatibility)
    calibrated_file = output_dir / 'calibrated_scores.jsonl'
    logger.info(f"Saving results to {calibrated_file}...")
    save_jsonl(scored_docs, calibrated_file)

    # Compute and save statistics
    stats = compute_statistics(scored_docs, m)
    stats_file = output_dir / 'phase2_statistics.json'
    save_json(stats, stats_file)
    logger.info(f"Saved statistics to {stats_file}")

    logger.info("="*80)
    logger.info("Phase 2 Complete!")
    logger.info("="*80)
    logger.info(f"Processed {len(scored_docs)} documents")
    logger.info(f"Output: {calibrated_file}")
    logger.info(f"Statistics: {stats_file}")


def compute_statistics(documents: List[Dict[str, Any]], m: int = 5) -> Dict[str, Any]:
    """Compute summary statistics for Phase 2 results."""
    n_docs = len(documents)

    # Collect all probabilities
    all_fact_probs = []
    all_imp_probs = []
    all_cov_probs = []

    for doc in documents:
        if doc is None:
            continue
        all_fact_probs.extend(doc.get('factuality_probs', []))
        all_imp_probs.extend(doc.get('importance_probs', []))
        all_cov_probs.extend(doc.get('coverage_probs', []))

    # Expected discrete values with 3-tier scoring (0, 0.5, 1.0) and m replicates
    # Gives m*2+1 values: {0/2m, 1/2m, 2/2m, ..., 2m/2m}
    expected_values = [i / (2 * m) for i in range(2 * m + 1)]

    stats = {
        'n_documents': n_docs,
        'vote_rate_replicates': m,
        'expected_probability_values': expected_values,
        'factuality': {
            'n_sentences': len(all_fact_probs),
            'avg_prob': float(np.mean(all_fact_probs)) if all_fact_probs else 0,
            'std_prob': float(np.std(all_fact_probs)) if all_fact_probs else 0,
            'min_prob': float(np.min(all_fact_probs)) if all_fact_probs else 0,
            'max_prob': float(np.max(all_fact_probs)) if all_fact_probs else 0,
        },
        'importance': {
            'n_sentences': len(all_imp_probs),
            'avg_prob': float(np.mean(all_imp_probs)) if all_imp_probs else 0,
            'std_prob': float(np.std(all_imp_probs)) if all_imp_probs else 0,
            'min_prob': float(np.min(all_imp_probs)) if all_imp_probs else 0,
            'max_prob': float(np.max(all_imp_probs)) if all_imp_probs else 0,
        },
        'coverage': {
            'n_sentences': len(all_cov_probs),
            'avg_prob': float(np.mean(all_cov_probs)) if all_cov_probs else 0,
            'std_prob': float(np.std(all_cov_probs)) if all_cov_probs else 0,
            'min_prob': float(np.min(all_cov_probs)) if all_cov_probs else 0,
            'max_prob': float(np.max(all_cov_probs)) if all_cov_probs else 0,
        },
    }

    return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    global prompt_config

    parser = argparse.ArgumentParser(description='Phase 2: Three-Tier Vote-Rate Scoring')
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Input Phase 1 oracle labels file (JSONL)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--num-docs',
        type=int,
        default=None,
        help='Number of documents to process (default: all)'
    )
    parser.add_argument(
        '--dataset',
        choices=["aci", "meq", "bhc", "cxr", "pubmed", "omop"],
        required=True,
        help='Define the dataset name'
    )
    parser.add_argument(
        '--m',
        '--replicates',
        type=int,
        default=config.VOTE_RATE_REPLICATES,
        help=f'Number of replicates for vote-rate scoring (default: {config.VOTE_RATE_REPLICATES})'
    )
    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Sentences per judge batch (default: {config.JUDGE_BATCH_SIZE}). Reduce for weaker models.'
    )
    parser.add_argument(
        '--coverage-only',
        action='store_true',
        help='Only re-score coverage (keep existing factuality/importance probs). '
             'Input should be existing Phase 2 scored_docs or calibrated_scores.'
    )
    parser.add_argument(
        '--importance-only',
        action='store_true',
        help='Only re-score importance (keep existing factuality/coverage probs). '
             'Input should be existing Phase 2 scored_docs or calibrated_scores.'
    )
    parser.add_argument(
        '--coverage-prompt',
        default='v1',
        help='Coverage prompt version (default: v1)'
    )
    parser.add_argument(
        '--importance-prompt',
        default='v1',
        choices=['v1', 'v2', 'fewshot'],
        help='Importance prompt version: v1 (3-tier), v2 (5-tier compression-aware), fewshot (3-tier with exemplars). Default: v1'
    )
    parser.add_argument(
        '--domain-specific',
        action='store_true',
        help='Use domain-specific factuality and importance triage prompts'
    )

    args = parser.parse_args()

    prompt_config = prompts.get_prompt_config(args.dataset)
    config.configure_dataset(args.dataset)

    # Override batch size if specified
    if args.batch_size is not None:
        config.JUDGE_BATCH_SIZE = args.batch_size

    # Set defaults after configure_dataset
    if args.coverage_only or args.importance_only:
        # Default input = existing Phase 2 calibrated scores
        input_file = args.input if args.input else config.PHASE2_DIR / 'calibrated_scores.jsonl'
    else:
        input_file = args.input if args.input else config.PHASE1_DIR / 'oracle_labels.jsonl'
    output_dir = args.output_dir if args.output_dir else config.PHASE2_DIR

    run_phase2(
        input_file=input_file,
        output_dir=output_dir,
        num_docs=args.num_docs,
        m=args.m,
        parallel_workers=args.parallel_workers,
        coverage_only=args.coverage_only,
        importance_only=args.importance_only,
        coverage_prompt_version=args.coverage_prompt,
        importance_prompt_version=args.importance_prompt,
        domain_specific=args.domain_specific,
    )


if __name__ == '__main__':
    main()
