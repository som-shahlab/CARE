#!/usr/bin/env python3
"""
Phase 1: Oracle Labeling for Factuality and Importance

This script implements Phase 1 of the conformal risk control pipeline:
1. Generate abstractive summaries: $\hat{S}(X) = f(X)$ for each document
2. Split summaries into sentences: $\mathcal{V}(X) = \text{Sent}(\hat{S}(X))$
3. Oracle factuality labels: $Y_{\text{fact}}(v; X)$ for each summary sentence
4. Oracle importance labels: $Y_{\text{imp}}(u; X, S)$ for each source sentence

Output per document:
{
    "doc_id": <int>,
    "generated_summary": <str>,
    "generated_sentences": [<str>, ...],  # V(X)
    "source_sentences": [<str>, ...],     # U(X)
    "factuality_labels": [0/1, ...],      # Y_fact(v; X) for each v in V(X)
    "importance_labels": [0/1, ...],      # Y_imp(u; X, S) for each u in U(X)
}
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import json
import re
import time
import random

from care.filters import is_valid_summary_sentence, is_valid_source_sentence
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports

from care import config, prompts
from care.utils import (
    LLMClient,
    load_jsonl,
    save_jsonl,
    save_json,
    setup_logging,
    split_summary_into_sentences,
    parse_yes_no_response,
)


logger = setup_logging('Phase1', config.LOG_LEVEL)
prompt_config = None

# Thread-local storage for LLM clients (for parallel processing)
_thread_local = threading.local()
_parallel_log_lock = threading.Lock()


def get_thread_clients():
    """Get or create thread-local LLM clients (summarizer and oracle only)."""
    if getattr(_thread_local, "initialized", False):
        return _thread_local.summarizer, _thread_local.oracle

    _thread_local.summarizer = LLMClient(
        model=config.SUMMARIZER_MODEL,
        api_key=config.SECUREGPT_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.get_azure_endpoint(config.SUMMARIZER_MODEL),
        max_tokens=config.SUMMARIZER_MAX_TOKENS,
        rate_limit_delay=config.RATE_LIMIT_DELAY,
    )
    _thread_local.oracle = LLMClient(
        model=config.ORACLE_MODEL,
        api_key=config.SECUREGPT_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.get_azure_endpoint(config.ORACLE_MODEL),
        max_tokens=config.ORACLE_MAX_TOKENS,
        rate_limit_delay=config.RATE_LIMIT_DELAY,
    )
    _thread_local.initialized = True
    return _thread_local.summarizer, _thread_local.oracle


def _process_document_parallel(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single document in parallel (uses thread-local clients)."""
    summarizer, oracle = get_thread_clients()
    with _parallel_log_lock:
        logger.info(f"  [Thread] Processing doc {doc['doc_id']}...")
    try:
        result = process_document(doc, summarizer, oracle)
        with _parallel_log_lock:
            logger.info(f"  [Thread] Completed doc {doc['doc_id']}")
        return result
    except Exception as e:
        with _parallel_log_lock:
            logger.error(f"  [Thread] Error processing doc {doc['doc_id']}: {e}")
        return None


# Exponential backoff and rate limit workarounds 
def is_rate_limit_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "429" in s
        or "rate limit" in s
        or "token limit is exceeded" in s
    )

def suggested_wait_seconds(e: Exception) -> Optional[float]:
    m = re.search(r"try again in (\d+)\s*seconds", str(e), re.IGNORECASE)
    return float(m.group(1)) if m else None

def generate_with_exponential_backoff(
    client,
    system_prompt,
    user_prompt,
    temperature,
    max_tokens,
    max_retries: int = 6,
    base: float = 2.0,
    cap: float = 60.0,
):
    """
    Exponential backoff + jitter for 429s.
    """
    for attempt in range(max_retries):
        try:
            return client.generate(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        except Exception as e:
            if not is_rate_limit_error(e):
                raise

            wait_hint = suggested_wait_seconds(e)
            if wait_hint is not None:
                sleep_s = min(cap, wait_hint + random.uniform(0, 1))
            else:
                sleep_s = min(cap, (base ** attempt) + random.uniform(0, 1))

            logger.warning(
                f"429 rate limit. attempt={attempt+1}/{max_retries}. "
                f"sleeping {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

    raise RuntimeError("Exceeded retries due to repeated rate limits.")


# Summary Generation and Caching

def get_summary_cache_path(doc_id: int) -> Path:
    """Get cache file path for a document's summary."""
    return config.SUMMARIES_DIR / f"doc_{doc_id}_summary.json"


def load_cached_summary(doc_id: int) -> str:
    """
    Load cached summary if available.

    Args:
        doc_id: Document ID

    Returns:
        Cached summary text, or None if not cached
    """
    import json
    cache_path = get_summary_cache_path(doc_id)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return data['summary']
    return None


def save_summary_to_cache(doc_id: int, summary: str):
    """
    Save summary to cache.

    Args:
        doc_id: Document ID
        summary: Generated summary text
    """
    cache_path = get_summary_cache_path(doc_id)
    import json
    with open(cache_path, 'w') as f:
        json.dump({'doc_id': doc_id, 'summary': summary}, f, indent=2)


def generate_summary(
    source_text: str,
    summarizer: LLMClient,
) -> str:
    """
    Generate abstractive summary.

    Args:
        source_text: Full source dialogue
        summarizer: LLM client for summary generation

    Returns:
        Generated summary text
    """
    system_prompt = prompt_config.summarizer_system_prompt
    user_prompt = prompt_config.get_summarizer_user_prompt(source_text)

    summary = generate_with_exponential_backoff(
        summarizer,
        system_prompt,
        user_prompt,
        temperature=config.SUMMARIZER_TEMPERATURE,
        max_tokens=config.SUMMARIZER_MAX_TOKENS,
    )

    return summary


# Oracle Labeling: Factuality

class CountMismatchError(ValueError):
    """Raised when LLM returns different count than expected."""
    pass


def verify_factuality_no_answer(
    sentence: str,
    source_text: str,
    oracle: LLMClient,
    max_retries: int = 5,
) -> int:
    """
    Two-pass verification for sentences initially marked as NOT SUPPORTED.

    Args:
        sentence: The summary sentence initially marked NO
        source_text: Source dialogue X
        oracle: LLM client for verification
        max_retries: Max retries on failure

    Returns:
        1 if verified as supported, 0 if confirmed unsupported
    """
    system_prompt = prompt_config.factuality_verification_system_prompt
    user_prompt = prompt_config.get_factuality_verification_user_prompt(sentence, source_text)

    # Debug: log prompt lengths on first call
    logger.info(f"    Verification prompts - system: {len(system_prompt)} chars, user: {len(user_prompt)} chars")

    for attempt in range(max_retries):
        try:
            # Use oracle.generate directly instead of generate_with_exponential_backoff
            response = oracle.generate(
                system_prompt,
                user_prompt,
                temperature=config.ORACLE_TEMPERATURE,
                max_tokens=config.ORACLE_MAX_TOKENS,  # Use same as batch labeling (8000)
            )

            response_clean = response.strip().upper()

            # Log the raw response for debugging
            if not response_clean or len(response_clean) < 2:
                logger.warning(f"Empty/short response (attempt {attempt + 1}): repr='{repr(response)}'. Retrying after delay...")
                time.sleep(2.0)  # Longer delay for API reliability issues
                continue

            # More robust parsing - look for YES/NO anywhere in response
            if "YES" in response_clean:
                logger.info(f"    Verified as YES: {sentence[:50]}...")
                return 1  # Changed to supported
            elif "NO" in response_clean:
                logger.debug(f"    Verified as NO: {sentence[:50]}...")
                return 0  # Confirmed unsupported
            else:
                logger.warning(f"Unexpected verification response (attempt {attempt + 1}): '{response[:100]}'. Retrying...")
                time.sleep(0.5)
                continue

        except Exception as e:
            logger.warning(f"Verification attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue

    # If all retries fail, keep original NO label (conservative)
    logger.warning(f"Verification failed after {max_retries} attempts for: {sentence[:60]}... Keeping NO.")
    return 0


def parse_json_labels(response: str, expected_count: int) -> List[int]:
    """
    Parse JSON array of yes/no responses into binary labels.

    Args:
        response: Raw LLM response containing JSON array
        expected_count: Expected number of labels

    Returns:
        List of binary labels (1 = yes, 0 = no)

    Raises:
        ValueError: If parsing fails
        CountMismatchError: If count doesn't match expected
    """

    # Extract JSON array from response (handle cases where there's extra text)
    response = response.strip()

    # Try to find JSON array in response
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        json_str = match.group()
    else:
        json_str = response

    try:
        labels_raw = json.loads(json_str)

        if not isinstance(labels_raw, list):
            raise ValueError(f"Expected list, got {type(labels_raw)}")

        # Convert yes/no to 1/0
        labels = []
        for item in labels_raw:
            item_lower = str(item).lower().strip()
            if 'yes' in item_lower:
                labels.append(1)
            elif 'no' in item_lower:
                labels.append(0)
            else:
                logger.warning(f"Unexpected label: {item}. Defaulting to 0.")
                labels.append(0)

        # Handle count mismatch - always raise error, no padding
        if len(labels) != expected_count:
            raise CountMismatchError(
                f"Expected {expected_count} labels, got {len(labels)}. "
                f"Response may be truncated or malformed."
            )

        return labels

    except CountMismatchError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse JSON labels: {e}")


def label_factuality(
    summary_sentences: List[str],
    source_text: str,
    oracle: LLMClient,
    max_retries: int = 3,
    verify_no_answers: bool = None,
) -> List[int]:
    """
    Get oracle factuality labels for all summary sentences in batched calls.

    For each summary sentence v in V(X), check if it is entailed by source X.
    Processes sentences in chunks to avoid hitting token limits.

    Two-pass verification (if enabled):
    - Pass 1: Batch labeling as usual
    - Pass 2: For sentences marked NO, verify individually with focused prompt

    Args:
        summary_sentences: List of summary sentences V(X)
        source_text: Source dialogue X
        oracle: LLM client for oracle labeling
        max_retries: Max retries on count mismatch (default 3)
        verify_no_answers: Whether to do two-pass verification (default from config)

    Returns:
        List of binary labels (1 = entailed, 0 = not entailed)
    """
    if not summary_sentences:
        return []

    if verify_no_answers is None:
        verify_no_answers = getattr(config, 'ORACLE_VERIFY_NO_ANSWERS', False)

    # Pass 1: Batch labeling
    all_labels = []
    chunk_size = config.ORACLE_BATCH_SIZE

    for i in range(0, len(summary_sentences), chunk_size):
        chunk = summary_sentences[i:i + chunk_size]

        system_prompt = prompt_config.factuality_oracle_system_prompt
        user_prompt = prompt_config.get_factuality_oracle_batch_prompt(chunk, source_text)

        labels = None
        for attempt in range(max_retries):
            response = generate_with_exponential_backoff(
                oracle,
                system_prompt,
                user_prompt,
                temperature=config.ORACLE_TEMPERATURE,
                max_tokens=config.ORACLE_MAX_TOKENS,
            )

            try:
                labels = parse_json_labels(response, len(chunk))
                break  # Success
            except CountMismatchError as e:
                logger.warning(f"Count mismatch on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                continue
            except ValueError as e:
                logger.error(f"Failed to parse factuality labels for chunk {i//chunk_size + 1}: {e}")
                logger.error(f"Response was: {response[:500]}...")
                break  # Don't retry on parse errors

        # If all retries failed, raise an error
        if labels is None:
            raise ValueError(
                f"CRITICAL: Could not get factuality labels for chunk {i//chunk_size + 1} after {max_retries} retries. "
                f"Last response: {response[:200] if response else 'None'}..."
            )

        all_labels.extend(labels)

    # Pass 2: Verify NO answers (if enabled)
    if verify_no_answers:
        no_indices = [i for i, label in enumerate(all_labels) if label == 0]
        if no_indices:
            logger.info(f"  Two-pass verification: checking {len(no_indices)} sentences marked NO...")
            changed_count = 0
            for idx in no_indices:
                sentence = summary_sentences[idx]
                verified_label = verify_factuality_no_answer(sentence, source_text, oracle)
                if verified_label == 1:
                    all_labels[idx] = 1
                    changed_count += 1
                    logger.debug(f"  Verified as YES: {sentence[:60]}...")
            if changed_count > 0:
                logger.info(f"  Two-pass verification: changed {changed_count}/{len(no_indices)} from NO to YES")

    return all_labels


# Oracle Labeling: Importance

def label_importance(
    source_sentences: List[str],
    reference_text: str,
    oracle: LLMClient,
    max_retries: int = 3,
) -> List[int]:
    """
    Get oracle importance labels for all source sentences in batched calls.

    For each source sentence u in U(X), check if it is covered by reference S.
    This serves as a proxy for importance.
    Processes sentences in chunks to avoid hitting token limits.

    Args:
        source_sentences: List of source sentences U(X)
        reference_text: Reference summary text S
        oracle: LLM client for oracle labeling
        max_retries: Max retries on count mismatch (default 3)

    Returns:
        List of binary labels (1 = covered/important, 0 = not covered)
    """
    if not source_sentences:
        return []

    # Process in chunks to avoid token limits
    all_labels = []
    chunk_size = config.ORACLE_BATCH_SIZE

    for i in range(0, len(source_sentences), chunk_size):
        chunk = source_sentences[i:i + chunk_size]

        system_prompt = prompt_config.importance_oracle_system_prompt
        user_prompt = prompt_config.get_importance_oracle_batch_prompt(chunk, reference_text)

        labels = None
        for attempt in range(max_retries):
            response = generate_with_exponential_backoff(
                oracle,
                system_prompt,
                user_prompt,
                temperature=config.ORACLE_TEMPERATURE,
                max_tokens=config.ORACLE_MAX_TOKENS,
            )

            try:
                labels = parse_json_labels(response, len(chunk))
                break  # Success
            except CountMismatchError as e:
                logger.warning(f"Count mismatch on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
            except ValueError as e:
                logger.error(f"Failed to parse importance labels for chunk {i//chunk_size + 1}: {e}")
                logger.error(f"Response was: {response[:500]}...")
                break

        # If all retries failed, raise an error
        if labels is None:
            raise ValueError(
                f"CRITICAL: Could not get importance labels for chunk {i//chunk_size + 1} after {max_retries} retries. "
                f"Last response: {response[:200] if response else 'None'}..."
            )

        all_labels.extend(labels)

    return all_labels


# Oracle Labeling: Coverage

def label_coverage(
    source_sentences: List[str],
    generated_summary: str,
    oracle: LLMClient,
    max_retries: int = 3,
) -> List[int]:
    """
    Get oracle coverage labels Y_cov for all source sentences in batched calls.

    For each source sentence u in U(X), check if it is covered by generated summary Ŝ(X).
    This identifies which source content is already in the summary.
    Uses Oracle for ground truth labels (per plan.tex Section 1).

    Processes sentences in chunks to avoid hitting token limits.

    Args:
        source_sentences: List of source sentences U(X)
        generated_summary: Generated summary text Ŝ(X)
        oracle: LLM client for oracle labeling
        max_retries: Max retries on count mismatch (default 3)

    Returns:
        List of binary labels Y_cov (1 = covered by summary, 0 = not covered)
    """
    if not source_sentences:
        return []

    # Process in chunks to avoid token limits
    all_labels = []
    chunk_size = config.ORACLE_BATCH_SIZE

    for i in range(0, len(source_sentences), chunk_size):
        chunk = source_sentences[i:i + chunk_size]

        system_prompt = prompt_config.coverage_oracle_system_prompt
        user_prompt = prompt_config.get_coverage_oracle_batch_prompt(chunk, generated_summary)

        labels = None
        for attempt in range(max_retries):
            response = generate_with_exponential_backoff(
                oracle,
                system_prompt,
                user_prompt,
                temperature=config.ORACLE_TEMPERATURE,
                max_tokens=config.ORACLE_MAX_TOKENS,
            )

            try:
                labels = parse_json_labels(response, len(chunk))
                break  # Success
            except CountMismatchError as e:
                logger.warning(f"Count mismatch on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
            except ValueError as e:
                logger.error(f"Failed to parse coverage labels for chunk {i//chunk_size + 1}: {e}")
                logger.error(f"Response was: {response[:500]}...")
                break

        # If all retries failed, raise an error
        if labels is None:
            raise ValueError(
                f"CRITICAL: Could not get coverage labels for chunk {i//chunk_size + 1} after {max_retries} retries. "
                f"Last response: {response[:200] if response else 'None'}..."
            )

        all_labels.extend(labels)

    return all_labels


# Document Processing

def process_document(
    doc: Dict[str, Any],
    summarizer: LLMClient,
    oracle: LLMClient,
    no_cache: bool = False,
) -> Dict[str, Any]:
    """
    Process a single document through Phase 1 pipeline.

    All ground truth labels (Y_fact, Y_imp, Y_cov) are produced by the Oracle
    per plan.tex Section 1.

    Args:
        doc: Document dict with source and reference
        summarizer: LLM for summary generation
        oracle: LLM for oracle labeling (all ground truth labels)

    Returns:
        Processed document with labels
    """
    doc_id = doc['doc_id']
    source_text = doc['source_text']
    source_sentences = doc['source_sentences']  # U(X)
    reference_text = doc['reference_text']

    logger.info(f"Processing doc {doc_id}...")

    logger.info("[prompts] SUMMARIZER_SYSTEM_PROMPT head: %s", prompt_config.summarizer_system_prompt[:80].replace("\n"," "))
    logger.info("[prompts] summarizer_user_prompt head: %s", prompt_config.get_summarizer_user_prompt(source_text)[:80].replace("\n"," "))


    #Generate summary (with caching)
    cached_summary = None if no_cache else load_cached_summary(doc_id)
    if cached_summary is not None:
        logger.info(f"  Using cached summary ({len(cached_summary)} chars)")
        generated_summary = cached_summary
    else:
        logger.info(f"  Generating summary...")
        generated_summary = generate_summary(source_text, summarizer)
        save_summary_to_cache(doc_id, generated_summary)
        logger.info(f"  Generated and cached summary ({len(generated_summary)} chars)")

    # Split summary into sentences
    generated_sentences_raw = split_summary_into_sentences(generated_summary)  # V(X) before filtering
    logger.info(f"  Split into {len(generated_sentences_raw)} summary sentences")

    # Apply deterministic unit filters (defines V'(X) and U'(X))
    # These are label-agnostic and applied identically to cal/test.
    # Full source_text/reference_text/generated_summary are still passed as context.
    summary_removed = [s for s in generated_sentences_raw if not is_valid_summary_sentence(s)]
    source_removed = [s for s in source_sentences if not is_valid_source_sentence(s)]
    generated_sentences = [s for s in generated_sentences_raw if is_valid_summary_sentence(s)]
    source_sentences = [s for s in source_sentences if is_valid_source_sentence(s)]
    logger.info(f"  Unit filter: kept {len(generated_sentences)}/{len(generated_sentences_raw)} summary, "
                f"{len(source_sentences)}/{len(source_sentences) + len(source_removed)} source")
    if summary_removed:
        logger.info(f"  Removed summary: {summary_removed}")
    if source_removed:
        logger.info(f"  Removed source: {source_removed[:5]}{'...' if len(source_removed) > 5 else ''}")

    # Label factuality (Y_fact)
    n_fact_chunks = (len(generated_sentences) + config.ORACLE_BATCH_SIZE - 1) // config.ORACLE_BATCH_SIZE
    logger.info(f"  Batch labeling factuality for {len(generated_sentences)} summary sentences ({n_fact_chunks} chunks)...")
    factuality_labels = label_factuality(generated_sentences, source_text, oracle)

    # Label importance (Y_imp)
    n_imp_chunks = (len(source_sentences) + config.ORACLE_BATCH_SIZE - 1) // config.ORACLE_BATCH_SIZE
    logger.info(f"  Batch labeling importance for {len(source_sentences)} source sentences ({n_imp_chunks} chunks)...")
    importance_labels = label_importance(source_sentences, reference_text, oracle)

    # Label coverage (check which source sentences are covered by generated summary)
    # Uses Oracle for ground truth Y_cov labels (per plan.tex Section 1)
    n_cov_chunks = (len(source_sentences) + config.ORACLE_BATCH_SIZE - 1) // config.ORACLE_BATCH_SIZE
    logger.info(f"  Batch labeling coverage for {len(source_sentences)} source sentences ({n_cov_chunks} chunks)...")
    coverage_labels = label_coverage(source_sentences, generated_summary, oracle)

    # Compute omission labels: important AND NOT covered by summary
    # omission_labels[i] = 1 if sentence is important but NOT in the generated summary
    omission_labels = [
        int(imp == 1 and cov == 0)
        for imp, cov in zip(importance_labels, coverage_labels)
    ]

    # Compute statistics
    n_factual = sum(factuality_labels)
    n_important = sum(importance_labels)
    n_covered = sum(coverage_labels)
    n_omitted = sum(omission_labels)
    logger.info(f"  Results: {n_factual}/{len(factuality_labels)} factual, "
                f"{n_important}/{len(importance_labels)} important, "
                f"{n_covered}/{len(coverage_labels)} covered, "
                f"{n_omitted}/{len(omission_labels)} omitted")

    return {
        'doc_id': doc_id,
        'source_text': source_text,
        'source_sentences': source_sentences,
        'reference_text': reference_text,
        'generated_summary': generated_summary,
        'generated_sentences': generated_sentences,
        'factuality_labels': factuality_labels,
        'importance_labels': importance_labels,
        'coverage_labels': coverage_labels,
        'omission_labels': omission_labels,
        'filter_metadata': {
            'summary_removed': summary_removed,
            'source_removed': source_removed,
            'summary_kept': len(generated_sentences),
            'summary_total': len(generated_sentences_raw),
            'source_kept': len(source_sentences),
            'source_total': len(source_sentences) + len(source_removed),
        },
    }


def run_phase1(
    input_file: Path = config.CALIBRATION_FILE,
    output_dir: Path = config.PHASE1_DIR,
    num_docs: int = None,
    resume: bool = False,
    parallel_workers: int = 1,
    no_cache: bool = False,
):
    """
    Run Phase 1: Oracle labeling for all calibration documents.

    Args:
        input_file: Path to calibration data (JSONL)
        output_dir: Directory for output files
        num_docs: Number of documents to process (None = all)
        resume: Whether to resume from existing checkpoint
        parallel_workers: Number of parallel workers (default: 1)
    """
    logger.info("="*80)
    logger.info("Phase 1: Oracle Labeling")
    logger.info("="*80)

    # Setup directories
    config.setup_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'oracle_labels.jsonl'
    logger.info(f"Output: {output_file}")

    # Initialize LLM clients (Oracle provides all ground truth labels per plan.tex)
    logger.info(f"Initializing LLM clients...")
    logger.info(f"  Summarizer: {config.SUMMARIZER_MODEL}")
    logger.info(f"  Oracle (all ground truth labels): {config.ORACLE_MODEL}")
    logger.info(f"  API Key: {config.SECUREGPT_API_KEY[:10]}...")

    summ_api_key = config.BEDROCK_API_KEY if config.SUMMARIZER_MODEL.lower().startswith('bedrock/') else config.SECUREGPT_API_KEY
    summarizer = LLMClient(
        model=config.SUMMARIZER_MODEL,
        api_key=summ_api_key,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.get_azure_endpoint(config.SUMMARIZER_MODEL),
        max_tokens=config.SUMMARIZER_MAX_TOKENS,
        rate_limit_delay=config.RATE_LIMIT_DELAY,
    )

    oracle = LLMClient(
        model=config.ORACLE_MODEL,
        api_key=config.SECUREGPT_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.get_azure_endpoint(config.ORACLE_MODEL),
        max_tokens=config.ORACLE_MAX_TOKENS,
        rate_limit_delay=config.RATE_LIMIT_DELAY,
    )

    # Load data
    logger.info(f"Loading data from {input_file}...")
    documents = load_jsonl(input_file)

    if num_docs:
        documents = documents[:num_docs]
        logger.info(f"Processing first {num_docs} documents")
    else:
        logger.info(f"Processing all {len(documents)} documents")

    # Resume handling
    processed_docs = []
    start_idx = 0
    labeled_ids = set()

    if resume:
        if output_file.exists():
            processed_docs = load_jsonl(output_file)
            labeled_ids = {d["doc_id"] for d in processed_docs}
            logger.info(f"Already labeled {len(labeled_ids)} docs")
        before = len(documents)
        documents = [d for d in documents if d["doc_id"] not in labeled_ids]
        logger.info(f"Remaining docs to label: {len(documents)}/{before}")
    else:
        # Not resuming - clear output file if it exists (for parallel mode append)
        if output_file.exists():
            output_file.unlink()

    # Process documents
    logger.info(f"Processing {len(documents)} documents...")

    if parallel_workers > 1 and documents:
        # Parallel execution
        logger.info(f"Using {parallel_workers} parallel workers...")
        _cache_write_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_process_document_parallel, doc): doc
                for doc in documents
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Documents"):
                doc = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        processed_docs.append(result)
                        # Append to file incrementally for crash recovery
                        with _cache_write_lock:
                            with output_file.open("a") as f:
                                f.write(json.dumps(result) + "\n")
                except Exception as e:
                    logger.error(f"Failed to process doc {doc['doc_id']}: {e}")
    else:
        # Sequential execution
        for i, doc in enumerate(tqdm(documents, desc="Documents")):
            try:
                processed_doc = process_document(doc, summarizer, oracle, no_cache=no_cache)
                processed_docs.append(processed_doc)

                # Save checkpoint periodically
                if (i + 1) % config.SAVE_FREQUENCY == 0:
                    logger.info(f"Saving checkpoint at {i + 1} documents...")
                    save_jsonl(processed_docs, output_file)

            except Exception as e:
                logger.error(f"Error processing doc {doc['doc_id']}: {e}")
                logger.error(f"Skipping document and continuing...")
                continue

    # Final save
    logger.info(f"Saving final results to {output_file}...")
    save_jsonl(processed_docs, output_file)

    # Save summary statistics
    stats = compute_statistics(processed_docs)
    stats_file = output_dir / 'phase1_statistics.json'
    save_json(stats, stats_file)
    logger.info(f"Saved statistics to {stats_file}")

    logger.info("="*80)
    logger.info("Phase 1 Complete!")
    logger.info("="*80)
    logger.info(f"Processed {len(processed_docs)} documents")
    logger.info(f"Output: {output_file}")
    logger.info(f"Statistics: {stats_file}")


def compute_statistics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for Phase 1 results."""
    n_docs = len(documents)

    # Count sentences
    n_generated = [len(d['generated_sentences']) for d in documents]
    n_source = [len(d['source_sentences']) for d in documents]

    # Count labels
    n_factual = [sum(d['factuality_labels']) for d in documents]
    n_important = [sum(d['importance_labels']) for d in documents]
    n_covered = [sum(d['coverage_labels']) for d in documents]
    n_omitted = [sum(d['omission_labels']) for d in documents]

    # Proportions
    prop_factual = [
        sum(d['factuality_labels']) / max(len(d['factuality_labels']), 1)
        for d in documents
    ]
    prop_important = [
        sum(d['importance_labels']) / max(len(d['importance_labels']), 1)
        for d in documents
    ]
    prop_covered = [
        sum(d['coverage_labels']) / max(len(d['coverage_labels']), 1)
        for d in documents
    ]
    prop_omitted = [
        sum(d['omission_labels']) / max(len(d['omission_labels']), 1)
        for d in documents
    ]

    return {
        'n_documents': n_docs,
        'avg_generated_sentences': sum(n_generated) / n_docs,
        'avg_source_sentences': sum(n_source) / n_docs,
        'avg_factual_sentences': sum(n_factual) / n_docs,
        'avg_important_sentences': sum(n_important) / n_docs,
        'avg_covered_sentences': sum(n_covered) / n_docs,
        'avg_omitted_sentences': sum(n_omitted) / n_docs,
        'avg_factuality_proportion': sum(prop_factual) / n_docs,
        'avg_importance_proportion': sum(prop_important) / n_docs,
        'avg_coverage_proportion': sum(prop_covered) / n_docs,
        'avg_omission_proportion': sum(prop_omitted) / n_docs,
    }

def main():
    global prompt_config
    
    parser = argparse.ArgumentParser(description='Phase 1: Oracle Labeling')
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Input calibration file (JSONL). Default: dataset-specific calibration file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory. Default: dataset-specific phase1 directory'
    )
    parser.add_argument(
        '--num-docs',
        type=int,
        default=None,
        help='Number of documents to process (default: all)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint'
    )
    parser.add_argument(
        '--dataset',
        choices=["aci", "meq", 'bhc', 'cxr', 'pubmed', 'omop'],
        required=True,
        help='Define the dataset name')
    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1). Use 4-8 for speedup.'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable summary cache (force regeneration with current summarizer)'
    )

    args = parser.parse_args()

    prompt_config = prompts.get_prompt_config(args.dataset)
    config.configure_dataset(args.dataset)

    # Set defaults after configure_dataset (so paths are dataset-specific)
    # Use ALL_DATA_FILE (calibration + test) so Phase 3 and 4 both have labels
    input_file = args.input if args.input else config.ALL_DATA_FILE
    output_dir = args.output_dir if args.output_dir else config.PHASE1_DIR

    run_phase1(
        input_file=input_file,
        output_dir=output_dir,
        num_docs=args.num_docs,
        resume=args.resume,
        parallel_workers=args.parallel_workers,
        no_cache=args.no_cache,
    )


if __name__ == '__main__':
    main()
