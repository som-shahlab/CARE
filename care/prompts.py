"""
Prompt templates for conformal risk control pipeline.

Phase 1 (Oracle): Ground truth labels Y ∈ {0, 1} (binary)
Phase 2 (Judge): 3-tier triage scoring with vote-rate averaging

Scoring tiers (Judge):
- Tier 1 (1.0): SUPPORTED / ESSENTIAL / COVERED
- Tier 2 (0.5): PARTIAL / RELEVANT / PARTIAL
- Tier 3 (0.0): UNSUPPORTED / NOT_RELEVANT / OMITTED

Oracle uses binary YES/NO for ground truth labels.
Judge uses 3-tier for calibrated scores.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any


# ============================================================================
# 1. DOMAIN SPECIFICATIONS
# ============================================================================

DOMAIN_SPECS: Dict[str, Dict[str, Any]] = {
    "aci": {
        "role": "clinician",
        "source_type": "patient-doctor dialogue",
        "output_type": "clinical summary",
        "source_label": "Patient-doctor dialogue",
        "summarizer_instruction": "Generate an ASSESSMENT AND PLAN clinical note.",
        "key_info_types": "diagnoses, medications, test results, findings, or clinical decisions",
        "info_domain": "clinical",
        "supported_examples": "Medications, symptoms, or findings mentioned in the source, even if wording differs",
        "inference_type": "Reasonable clinical inference from stated facts",
        "templating_example": 'Standard clinical templating (e.g., "follow up as needed", "patient education was provided")',
        "fabrication_examples": "fabricating lab values, diagnoses, or medications that don't appear",
    },
    "bhc": {
        "role": "clinician",
        "source_type": "discharge note",
        "output_type": "Brief Hospital Course",
        "source_label": "Discharge note content",
        "summarizer_instruction": "Write a Brief Hospital Course (BHC) narrative.",
        "key_info_types": "diagnoses, procedures, medications, test results, or clinical decisions",
        "info_domain": "clinical",
        "supported_examples": "Medications, symptoms, or findings mentioned in the source, even if wording differs",
        "inference_type": "Reasonable clinical inference from stated facts",
        "templating_example": 'Standard clinical templating (e.g., "follow up as needed", "patient education was provided")',
        "fabrication_examples": "fabricating lab values, diagnoses, or medications that don't appear",
    },
    "cxr": {
        "role": "radiologist",
        "source_type": "chest X-ray FINDINGS",
        "output_type": "radiology IMPRESSION",
        "source_label": "FINDINGS",
        "summarizer_instruction": "Generate a radiology IMPRESSION from FINDINGS.",
        "key_info_types": "radiographic findings, impressions, or clinical recommendations",
        "info_domain": "clinical",
        "supported_examples": "Radiographic findings or impressions mentioned in the source, even if wording differs",
        "inference_type": "Reasonable clinical inference from stated findings",
        "templating_example": 'Standard radiology templating (e.g., "no acute findings", "clinical correlation recommended")',
        "fabrication_examples": "fabricating findings, measurements, or diagnoses that don't appear",
    },
    "meq": {
        "role": "medical question summarization specialist",
        "source_type": "consumer health question",
        "output_type": "summarized medical question",
        "source_label": "Consumer health question",
        "summarizer_instruction": "Distill the verbose consumer health question into a concise summary.",
        "key_info_types": "symptoms, conditions, medications, or specific medical concerns",
        "info_domain": "clinical",
        "supported_examples": "Symptoms, conditions, or medications mentioned in the source, even if wording differs",
        "inference_type": "Reasonable clinical inference from stated facts",
        "templating_example": 'Standard medical summarization phrasing',
        "fabrication_examples": "fabricating symptoms, conditions, or medications that don't appear",
    },
    "pubmed": {
        "role": "scientific writer",
        "source_type": "scientific article text",
        "output_type": "structured abstract",
        "source_label": "Article text",
        "summarizer_instruction": "Write a structured abstract with BACKGROUND, RESULTS, and CONCLUSIONS sections.",
        "key_info_types": "study objectives, methods, results, or conclusions",
        "info_domain": "scientific",
        "supported_examples": "Methods, results, or findings mentioned in the source, even if wording differs",
        "inference_type": "Reasonable scientific inference from stated results",
        "templating_example": 'Standard scientific writing conventions (e.g., "further research is needed", "taken together, these results suggest")',
        "fabrication_examples": "fabricating experimental results, statistical values, or conclusions that don't appear",
    },
    "omop": {
        "role": "clinician",
        "source_type": "clinical notes",
        "output_type": "discharge summary",
        "source_label": "Clinical notes",
        "summarizer_instruction": "Write a discharge summary from the clinical notes.",
        "key_info_types": "diagnoses, procedures, medications, lab results, clinical events, or discharge plans",
        "info_domain": "clinical",
        "supported_examples": "Diagnoses, medications, procedures, or findings mentioned in the source, even if wording differs",
        "inference_type": "Reasonable clinical inference from stated facts",
        "templating_example": 'Standard clinical templating (e.g., "patient tolerated procedure well", "discharge to home with follow-up")',
        "fabrication_examples": "fabricating diagnoses, lab values, medications, procedures, or discharge instructions that don't appear",
    },
}


def get_domain_spec(dataset: str) -> Dict[str, Any]:
    """Get domain specification."""
    key = dataset.lower().strip()
    if key not in DOMAIN_SPECS:
        raise ValueError(f"Unknown dataset: {dataset}. Expected one of: {list(DOMAIN_SPECS.keys())}")
    return DOMAIN_SPECS[key]


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def format_sentences(sentences: list, truncate_at: int = 500) -> str:
    """Format sentences as numbered list, truncating very long ones."""
    return "\n".join(
        f"{i+1}. {s[:truncate_at]}..." if len(s) > truncate_at else f"{i+1}. {s}"
        for i, s in enumerate(sentences)
    )


# ============================================================================
# 3. SUMMARIZER PROMPTS
# ============================================================================

def get_summarizer_system_prompt(dataset: str) -> str:
    spec = get_domain_spec(dataset)
    return f"You are a {spec['role']}. {spec['summarizer_instruction']}"


def get_summarizer_user_prompt(dataset: str, source_text: str) -> str:
    spec = get_domain_spec(dataset)
    return f"{spec['source_label']}:\n\n{source_text}"


# ============================================================================
# 4. FACTUALITY PROMPTS (shared between Oracle and Judge)
# ============================================================================

def get_factuality_verification_system_prompt(dataset: str) -> str:
    """System prompt for two-pass verification of sentences initially marked as unsupported."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} reviewing whether a summary sentence is supported by the source.

Answer YES if the sentence is:
- A paraphrase of source content (e.g., "i have you on bumex" = "taking Bumex")
- Standard clinical templating (e.g., "follow up as needed")
- A reasonable inference from stated facts

Answer NO only if the sentence contradicts or fabricates specific details.

Respond with exactly one word: YES or NO."""


def get_factuality_verification_user_prompt(dataset: str, sentence: str, source_text: str) -> str:
    """User prompt for two-pass verification."""
    spec = get_domain_spec(dataset)
    return f"""Original {spec['source_type']}:
{source_text}

Sentence initially marked as NOT SUPPORTED:
"{sentence}"

After careful review, is this sentence actually supported by the source?
Consider: paraphrases, clinical templating, reasonable inferences.

Your answer must be exactly YES or NO:"""


def get_factuality_system_prompt(dataset: str) -> str:
    """System prompt for factuality checking."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} verifying factual accuracy.

For each {spec['output_type']} sentence, determine if it is supported by the source {spec['source_type']}.

**Supported (YES):**
- Explicitly stated in the source
- Paraphrased or reformulated from the source
- {spec['supported_examples']}
- {spec['inference_type']}
- {spec['templating_example']}
- Accurately summarizing what was discussed

**Unsupported (NO):**
- Directly contradicts the source
- Invents specific details NOT mentioned anywhere in the source (e.g., {spec['fabrication_examples']})

Respond with a JSON array of "YES" or "NO" for each sentence.
Return ONLY the JSON array, no other text."""


def get_factuality_user_prompt(dataset: str, summary_sentences: list, source_text: str) -> str:
    """User prompt for factuality checking."""
    spec = get_domain_spec(dataset)
    sentences_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summary_sentences))
    return f"""{spec['source_type'].capitalize()}:
{source_text}

{spec['output_type'].capitalize()} sentences to verify:
{sentences_text}

JSON array:"""


# ============================================================================
# 5. IMPORTANCE PROMPTS (Oracle vs Judge differ here)
# ============================================================================

def get_importance_oracle_system_prompt(dataset: str) -> str:
    """Oracle: Compare source sentences to human reference summary."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} checking content alignment.

For each {spec['source_type']} sentence, determine if it **states specific {spec['info_domain']} information** (such as {spec['key_info_types']}) that is also represented in the reference {spec['output_type']} (possibly paraphrased).

**Included (YES):** The sentence asserts a {spec['info_domain']} fact that appears in the reference.
**Not included (NO):** The sentence does not assert {spec['info_domain']} information found in the reference.

Respond with a JSON array of "YES" or "NO" for each sentence.
Return ONLY the JSON array, no other text."""


def get_importance_oracle_user_prompt(dataset: str, source_sentences: list, reference_summary: str) -> str:
    """Oracle: User prompt comparing to reference."""
    spec = get_domain_spec(dataset)
    sentences_text = format_sentences(source_sentences)
    return f"""Reference {spec['output_type']}:
{reference_summary}

{spec['source_type'].capitalize()} sentences to check:
{sentences_text}

JSON array:"""


def get_importance_judge_system_prompt(dataset: str) -> str:
    """Judge: Predict importance without seeing reference."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} deciding what to include in a {spec['output_type']}.

For each {spec['source_type']} sentence, determine if it contains information that belongs in a {spec['output_type']}.

**Important (YES):** Contains information that a {spec['role']} would include when writing a {spec['output_type']}.
**Not important (NO):** Conversational, procedural, redundant, or provides only supporting context that would not typically appear in a {spec['output_type']}.

Respond with a JSON array of "YES" (important) or "NO" (not important) for each sentence.
Return ONLY the JSON array, no other text."""


def get_importance_judge_user_prompt(dataset: str, source_sentences: list, source_text: str) -> str:
    """Judge: User prompt for importance prediction."""
    spec = get_domain_spec(dataset)
    sentences_text = format_sentences(source_sentences)
    return f"""Full {spec['source_type']} (for context):
{source_text}

Sentences to evaluate for importance:
{sentences_text}

JSON array:"""


# ============================================================================
# 6. COVERAGE PROMPTS (shared between Oracle and Judge)
# ============================================================================

def get_coverage_system_prompt(dataset: str) -> str:
    """System prompt for coverage checking."""
    spec = get_domain_spec(dataset)
    return f"""You are an auditor checking for information loss.

For each {spec['source_type']} sentence, determine if its content is represented in the {spec['output_type']}.

**Covered:** The sentence's key information appears in the summary (possibly paraphrased).
**Omitted:** The information is missing, too vague, or its meaning has been lost.

Respond with a JSON array of "YES" (covered) or "NO" (omitted) for each sentence.
Return ONLY the JSON array, no other text."""


def get_coverage_user_prompt(dataset: str, source_sentences: list, generated_summary: str) -> str:
    """User prompt for coverage checking."""
    spec = get_domain_spec(dataset)
    sentences_text = format_sentences(source_sentences)
    return f"""Generated {spec['output_type']}:
{generated_summary}

{spec['source_type'].capitalize()} sentences to check:
{sentences_text}

JSON array:"""


# ============================================================================
# 7. THREE-TIER JUDGE PROMPTS (Phase 2 scoring)
# ============================================================================

def get_factuality_triage_system_prompt(dataset: str) -> str:
    """3-tier factuality scoring for Judge."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} verifying factual accuracy.

For each sentence, respond with one of:
- SUPPORTED: Clearly supported by the source
- PARTIAL: Ambiguous, partially supported, or reasonable inference
- UNSUPPORTED: Contradicts or fabricates details not in source

Respond with a JSON array. Example: ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]"""


def get_factuality_triage_user_prompt(dataset: str, summary_sentences: list, source_text: str) -> str:
    """User prompt for 3-tier factuality."""
    spec = get_domain_spec(dataset)
    sentences_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summary_sentences))
    return f"""Source {spec['source_type']}:
{source_text}

Sentences to verify:
{sentences_text}

JSON array (SUPPORTED/PARTIAL/UNSUPPORTED for each):"""


def get_importance_triage_system_prompt(dataset: str) -> str:
    """3-tier importance scoring for Judge."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} deciding what to include in a {spec['output_type']}.

For each {spec['source_type']} sentence, respond with one of:
- ESSENTIAL: Contains information that belongs in a {spec['output_type']}
- RELEVANT: Provides supporting context but would not typically appear in a {spec['output_type']}
- NOT_RELEVANT: Conversational, procedural, or not relevant to a {spec['output_type']}

Respond with a JSON array. Example: ["ESSENTIAL", "RELEVANT", "NOT_RELEVANT"]"""


def get_importance_triage_user_prompt(dataset: str, source_sentences: list, source_text: str) -> str:
    """User prompt for 3-tier importance."""
    spec = get_domain_spec(dataset)
    sentences_text = format_sentences(source_sentences)
    return f"""Full {spec['source_type']}:
{source_text}

Sentences to evaluate:
{sentences_text}

JSON array (ESSENTIAL/RELEVANT/NOT_RELEVANT for each):"""


def get_importance_triage_fewshot_system_prompt(dataset: str) -> str:
    """3-tier importance scoring with few-shot exemplars from calibration references.

    Key change vs v1: includes concrete examples of ESSENTIAL/RELEVANT/NOT_RELEVANT
    sentences selected using oracle labels (derived from clinician reference summaries).
    The examples teach the judge what clinicians actually consider important.
    """
    spec = get_domain_spec(dataset)

    # Few-shot examples are dataset-specific
    if dataset.lower().strip() == "bhc":
        examples_block = """
Here are examples from clinician-reviewed discharge summaries:

ESSENTIAL examples (critical information that MUST appear in the Brief Hospital Course):
- "She was started on a heparin gtt, she had a cardiac catheterization which showed stenosis of her Lcx and underwent 2 DES placed to the mid Lcx." → ESSENTIAL (key intervention: cardiac catheterization and stent placement)
- "Patient was started on digoxin for rate control and increased metoprolol, and diuresed until euvolemic." → ESSENTIAL (medication changes and therapeutic goals)

RELEVANT examples (important clinical context, already appropriately captured):
- "Selective coronary angiography of this right dominant system revealed a large superdominant RCA with a 95% mid-vessel lesion." → RELEVANT (significant diagnostic finding providing context)
- "The patient was recently admitted with anemia and melena due to his duodenal tumor." → RELEVANT (key presenting history)

NOT_RELEVANT examples (routine content clinicians exclude from summaries):
- "Vitals at the time of transfer: 97.5, 65, 133/79, 16, 100%RA." → NOT_RELEVANT (routine single-timepoint vitals)
- "It is very important to take the medications as prescribed and bring your list of medications to all of your appointments." → NOT_RELEVANT (generic boilerplate discharge instructions)
"""
    else:
        # Fallback: no examples for other datasets (use standard v1 behavior)
        examples_block = ""

    return f"""You are a {spec['role']} deciding what to include in a {spec['output_type']}.

For each {spec['source_type']} sentence, respond with one of:
- ESSENTIAL: Contains information that belongs in a {spec['output_type']}
- RELEVANT: Provides supporting context but would not typically appear in a {spec['output_type']}
- NOT_RELEVANT: Conversational, procedural, or not relevant to a {spec['output_type']}
{examples_block}
Respond with a JSON array. Example: ["ESSENTIAL", "RELEVANT", "NOT_RELEVANT"]"""


def get_importance_triage_fewshot_user_prompt(dataset: str, source_sentences: list, source_text: str) -> str:
    """User prompt for few-shot 3-tier importance."""
    spec = get_domain_spec(dataset)
    sentences_text = format_sentences(source_sentences)
    return f"""Full {spec['source_type']}:
{source_text}

Sentences to evaluate:
{sentences_text}

JSON array (ESSENTIAL/RELEVANT/NOT_RELEVANT for each):"""


def get_importance_triage_v2_system_prompt(dataset: str) -> str:
    """Compression-aware importance scoring with 5-point scale (v2).

    Key changes vs v1:
    1. Acknowledges compression explicitly (most content will be omitted)
    2. 5-point scale instead of 3-tier (finer discrimination, reduces pile-up)
    3. Base-rate hints (~15-20% MUST_INCLUDE) to calibrate model expectations
    4. Safety-grounded criteria instead of generic relevance
    """
    spec = get_domain_spec(dataset)

    if spec['info_domain'] == 'clinical':
        safety_criterion = "patient safety, accurate diagnosis, or treatment decisions"
    else:
        safety_criterion = "the accuracy of key findings or conclusions"

    return f"""You are a {spec['role']} reviewing a {spec['source_type']} that will be compressed into a much shorter {spec['output_type']}. Only about 15-20% of source content can be retained — most will necessarily be omitted.

For each sentence, judge how critical it is to RETAIN, given that most content must be excluded:

- MUST_INCLUDE: Omitting this could compromise {safety_criterion}. (Expect ~10-15% of sentences)
- SHOULD_INCLUDE: Significant {spec['info_domain']} value; retain if space allows. (Expect ~10-15%)
- NICE_TO_HAVE: Useful context but safely omissible. (Expect ~20-30%)
- SAFE_TO_OMIT: Supporting detail routinely excluded from compressed summaries. (Expect ~20-30%)
- NOT_RELEVANT: No value for this {spec['output_type']}. (Expect ~15-25%)

Respond with a JSON array. Example: ["MUST_INCLUDE", "SAFE_TO_OMIT", "NICE_TO_HAVE", "NOT_RELEVANT", "SHOULD_INCLUDE"]"""


def get_importance_triage_v2_user_prompt(dataset: str, source_sentences: list, source_text: str) -> str:
    """User prompt for 5-tier importance (v2)."""
    spec = get_domain_spec(dataset)
    sentences_text = format_sentences(source_sentences)
    return f"""Full {spec['source_type']}:
{source_text}

Sentences to evaluate:
{sentences_text}

JSON array (MUST_INCLUDE/SHOULD_INCLUDE/NICE_TO_HAVE/SAFE_TO_OMIT/NOT_RELEVANT for each):"""


def get_custom_factuality_triage_system_prompt(dataset: str) -> str:
    """Domain-specific 3-tier factuality scoring for Judge."""
    spec = get_domain_spec(dataset)
    key = dataset.lower().strip()

    prompt = f"""You are a {spec['role']} verifying factual accuracy."""

    if key == "cxr":
        fact_tiers = """
- SUPPORTED:
  The sentence is consistent with the FINDINGS in meaning, without adding new clinical implications.
  A radiologist reading the sentence would not learn anything beyond what is already in the FINDINGS.
  Paraphrasing, synthesis, restating uncertainty with the same strength, and
  standard radiology boilerplate language are acceptable.

- PARTIAL:
  The sentence is clearly anchored to a real finding in the FINDINGS but adds
  a clinically meaningful interpretation or modification that is not explicitly stated.

- UNSUPPORTED:
  The sentence contains a claim that is not grounded in the FINDINGS,
  introduces new diagnoses, etiologies, or recommendations,
  contradicts a stated finding, or substitutes language that
  changes clinical meaning (e.g., "unchanged" for "intact").
"""
    elif key == "aci":
        fact_tiers = """
- SUPPORTED:
  The sentence is consistent with the dialogue in overall meaning.
  Paraphrasing, synthesis, clinical interpretation, and standard documentation
  conventions are all acceptable.

- PARTIAL:
  The sentence refers to something discussed but represents it
  imprecisely beyond what standard clinical inference supports.

- UNSUPPORTED:
  Clear invention or contradiction relative to the dialogue, especially:
  new meds/doses, new diagnoses, new tests/results, new numeric values, invented timelines,
  incorrect negations, or plan items not discussed.
"""
    elif key == "pubmed":
        fact_tiers = """
- SUPPORTED:
  Consistent with the article in meaning and certainty. Paraphrasing and compression are OK if it does not add new claims.

- PARTIAL:
  Grounded but meaningfully imprecise — scope or certainty is softened/broadened without introducing wrong facts.

- UNSUPPORTED:
  Adds claims not in the article or contradicts it (including invented numbers/statistics, methods, or conclusions).
"""
    elif key == "bhc":
        fact_tiers = """
- SUPPORTED:
  Faithful to the patient record in clinical meaning. Paraphrasing and compression are acceptable
  as long as no clinical facts are introduced or distorted.

- PARTIAL:
  Core claim is grounded but a clinically relevant detail is
  imprecise or incomplete — without distorting clinical meaning.

- UNSUPPORTED:
  Introduces, contradicts, or misrepresents the record, including:
  fabricated facts, wrong laterality/entity/test, inverted
  negations, trend errors (saying a value rose when it fell),
  omitting a key inflection point that changes clinical meaning,
  or numbers that contradict the sentence's own narrative.
"""
    elif key == "omop":
        fact_tiers = """
- SUPPORTED:
  Faithful to the patient record in clinical meaning and timing.
  Paraphrasing and synthesis across notes are allowed if no new facts or stronger certainty are introduced.

- PARTIAL:
  Anchored to the record but meaningfully imprecise
  (e.g., blurred timing, softened/strengthened certainty,
  slightly wrong specificity) without clear fabrication.
  Use sparingly.

- UNSUPPORTED:
  Not grounded in the record or contradicts it.
  Includes invented diagnoses, procedures, meds, results,
  wrong negations, wrong timing, or wrong clinical trends.
"""
    else:
        # Fallback: use generic tiers
        fact_tiers = """
- SUPPORTED: Clearly supported by the source
- PARTIAL: Ambiguous, partially supported, or reasonable inference
- UNSUPPORTED: Contradicts or fabricates details not in source
"""

    json_fmt = '\nRespond ONLY with a JSON array using exactly: "SUPPORTED", "PARTIAL", or "UNSUPPORTED". Example: ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]'

    return prompt + fact_tiers + json_fmt


def get_custom_importance_triage_system_prompt(dataset: str) -> str:
    """Domain-specific 3-tier importance scoring for Judge."""
    spec = get_domain_spec(dataset)
    key = dataset.lower().strip()

    prompt = f"""You are a {spec['role']} deciding what is important to include in a {spec['output_type']}. For each {spec['source_type']} sentence, respond with one of:"""

    if key == "cxr":
        importance_tiers = """
- ESSENTIAL:
  Drives the overall impression or changes the clinical interpretation
  (e.g., acute abnormality, actionable finding, device position that matters).

- RELEVANT:
  True and clinically meaningful, but does not change the impression
  (e.g., stable or chronic findings, detailed negatives, supporting context).

- NOT_RELEVANT:
  Acquisition details, boilerplate, or purely descriptive text.

Important:
- Normal or negative findings are ESSENTIAL only if they are the main conclusion
  (e.g., "no acute cardiopulmonary process").
- Stable or chronic findings are usually RELEVANT, not ESSENTIAL.
"""
    elif key == "aci":
        importance_tiers = """
- ESSENTIAL:
  Contains a diagnosis, key symptom/history, test result, medication,
  or a concrete plan/action that directly affects the assessment or plan.

- RELEVANT:
  Provides supporting clinical context but could be safely omitted without changing the resulting assessment and plan.

- NOT_RELEVANT:
  Conversational filler, repetition, logistics, administrative content,
  OR clinical information that does not affect the assessment or plan.

Important:
- Do not mark headings or administrative fields (name/date/signature) as ESSENTIAL.
"""
    elif key == "bhc":
        importance_tiers = """
- ESSENTIAL: Core clinical information, including admission reason, key findings that drove care, final diagnosis,
  major treatments/med changes, and discharge plan/disposition.

- RELEVANT: Clinically meaningful context that supports the narrative but could
  be safely omitted without changing the overall story.

- NOT_RELEVANT: Does not belong in a brief narrative: boilerplate, administrative text,
  repeated information, or content with no patient-specific meaning.
"""
    elif key == "pubmed":
        importance_tiers = """
- ESSENTIAL:
  Core structured-abstract content: the study aim/question, what was done,
  the main result(s), and the main conclusion.

- RELEVANT:
  Helpful supporting context or secondary detail that can be shortened or omitted
  without changing the main story.

- NOT_RELEVANT:
  Background context, extended mechanistic detail,
  granular numbers, or discussion-level speculation.
"""
    elif key == "omop":
        importance_tiers = """
- ESSENTIAL:
  Information required to understand the hospitalization:
  why the patient was admitted, what was found, what was done,
  the final diagnoses, major treatment decisions, and the discharge plan.
  If removing the sentence would make the hospitalization story incomplete or misleading, it is ESSENTIAL.

- RELEVANT:
  Clinically meaningful context that supports the story but could be omitted without changing the main narrative.

- NOT_RELEVANT:
  Does not belong in a discharge summary narrative,
  including boilerplate, repetition, routine data, or information
  that did not affect diagnosis, management, or disposition.
"""
    else:
        # Fallback: use generic tiers
        importance_tiers = """
- ESSENTIAL: Contains information that belongs in a summary
- RELEVANT: Provides supporting context but would not typically appear in the summary
- NOT_RELEVANT: Conversational, procedural, or not relevant to the summary
"""

    json_fmt = '\nRespond with a JSON array. Example: ["ESSENTIAL", "RELEVANT", "NOT_RELEVANT"]'

    return prompt + importance_tiers + json_fmt


def get_coverage_triage_system_prompt(dataset: str) -> str:
    """3-tier coverage scoring for Judge."""
    spec = get_domain_spec(dataset)
    return f"""You are checking if source content appears in the summary.

For each source sentence, respond with one of:
- COVERED: Information is clearly present in the summary
- PARTIAL: Only partially represented or paraphrased loosely
- OMITTED: Information is missing from the summary

Respond with a JSON array. Example: ["COVERED", "PARTIAL", "OMITTED"]"""


def get_coverage_triage_user_prompt(dataset: str, source_sentences: list, generated_summary: str) -> str:
    """User prompt for 3-tier coverage."""
    spec = get_domain_spec(dataset)
    sentences_text = format_sentences(source_sentences)
    return f"""Summary:
{generated_summary}

Source sentences to check:
{sentences_text}

JSON array (COVERED/PARTIAL/OMITTED for each):"""


# ============================================================================
# 7b. SINGLE-LABEL TOKEN-PROBABILITY PROMPTS (logit-readout scorer)
# ============================================================================
# These prompts ask the model to emit EXACTLY one of three single letters
# (A / B / C) so the next-token logits at the assistant turn opener can be
# read off directly. Single-letter labels guarantee single-token tokenization
# under Llama/Qwen BPE tokenizers and avoid prefix collisions.
#
# Letter -> tier mapping is fixed and identical across factuality/importance/
# coverage to keep the consumer side simple:
#   A = highest tier  (SUPPORTED / ESSENTIAL  / COVERED)
#   B = middle tier   (PARTIAL   / RELEVANT   / PARTIAL)
#   C = lowest tier   (UNSUPPORTED / NOT_RELEVANT / OMITTED)

def get_factuality_token_probs_system_prompt(dataset: str) -> str:
    """Single-label factuality prompt for logit-readout scorer."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} verifying factual accuracy.

You will be given a source and one summary sentence. Reply with EXACTLY ONE LETTER and nothing else:
- A: the sentence is clearly supported by the source (SUPPORTED)
- B: the sentence is ambiguous, partially supported, or a reasonable inference (PARTIAL)
- C: the sentence contradicts the source or fabricates details (UNSUPPORTED)

Output a single character (A, B, or C). No explanation, no punctuation."""


def get_factuality_token_probs_user_prompt(dataset: str, summary_sentence: str, source_text: str) -> str:
    """User prompt for single-label factuality."""
    spec = get_domain_spec(dataset)
    return f"""Source {spec['source_type']}:
{source_text}

Summary sentence:
{summary_sentence}

Answer (A, B, or C):"""


def get_importance_token_probs_system_prompt(dataset: str) -> str:
    """Single-label importance prompt for logit-readout scorer."""
    spec = get_domain_spec(dataset)
    return f"""You are a {spec['role']} deciding what to include in a {spec['output_type']}.

You will be given a source and one source sentence drawn from it. Reply with EXACTLY ONE LETTER and nothing else:
- A: the sentence contains information that belongs in a {spec['output_type']} (ESSENTIAL)
- B: the sentence provides supporting context but would not typically appear in a {spec['output_type']} (RELEVANT)
- C: the sentence is conversational, procedural, or not relevant to a {spec['output_type']} (NOT_RELEVANT)

Output a single character (A, B, or C). No explanation, no punctuation."""


def get_importance_token_probs_user_prompt(dataset: str, source_sentence: str, source_text: str) -> str:
    """User prompt for single-label importance."""
    spec = get_domain_spec(dataset)
    return f"""Full {spec['source_type']}:
{source_text}

Source sentence to evaluate:
{source_sentence}

Answer (A, B, or C):"""


def get_coverage_token_probs_system_prompt(dataset: str) -> str:
    """Single-label coverage prompt for logit-readout scorer."""
    return f"""You are checking whether source content appears in a summary.

You will be given a generated summary and one source sentence. Reply with EXACTLY ONE LETTER and nothing else:
- A: the information is clearly present in the summary (COVERED)
- B: the information is only partially represented or paraphrased loosely (PARTIAL)
- C: the information is missing from the summary (OMITTED)

Output a single character (A, B, or C). No explanation, no punctuation."""


def get_coverage_token_probs_user_prompt(dataset: str, source_sentence: str, generated_summary: str) -> str:
    """User prompt for single-label coverage."""
    return f"""Summary:
{generated_summary}

Source sentence to check:
{source_sentence}

Answer (A, B, or C):"""


# ============================================================================
# 8. PROMPT CONFIG (unified interface)
# ============================================================================

@dataclass
class PromptConfig:
    """Unified prompt configuration for a dataset."""
    # Summarizer
    summarizer_system_prompt: str
    get_summarizer_user_prompt: Callable

    # Factuality (shared)
    factuality_system_prompt: str
    get_factuality_user_prompt: Callable

    # Factuality verification (two-pass for NO answers)
    factuality_verification_system_prompt: str
    get_factuality_verification_user_prompt: Callable

    # Importance - Oracle (compares to reference)
    importance_oracle_system_prompt: str
    get_importance_oracle_user_prompt: Callable

    # Importance - Judge (predicts without reference)
    importance_judge_system_prompt: str
    get_importance_judge_user_prompt: Callable

    # Coverage (shared)
    coverage_system_prompt: str
    get_coverage_user_prompt: Callable

    # 3-tier Judge prompts (Phase 2 scoring)
    factuality_triage_system_prompt: str = ""
    get_factuality_triage_user_prompt: Callable = None
    importance_triage_system_prompt: str = ""
    get_importance_triage_user_prompt: Callable = None
    importance_triage_v2_system_prompt: str = ""
    get_importance_triage_v2_user_prompt: Callable = None
    importance_triage_fewshot_system_prompt: str = ""
    get_importance_triage_fewshot_user_prompt: Callable = None
    coverage_triage_system_prompt: str = ""
    get_coverage_triage_user_prompt: Callable = None

    # Domain-specific 3-tier Judge prompts (Phase 2, --domain-specific flag)
    custom_factuality_triage_system_prompt: str = ""
    custom_importance_triage_system_prompt: str = ""

    # Single-label token-probability prompts (logit-readout scorer)
    factuality_token_probs_system_prompt: str = ""
    get_factuality_token_probs_user_prompt: Callable = None
    importance_token_probs_system_prompt: str = ""
    get_importance_token_probs_user_prompt: Callable = None
    coverage_token_probs_system_prompt: str = ""
    get_coverage_token_probs_user_prompt: Callable = None

    # Aliases for backward compatibility
    @property
    def factuality_oracle_system_prompt(self):
        return self.factuality_system_prompt

    @property
    def judge_factuality_binary_system_prompt(self):
        return self.factuality_system_prompt

    @property
    def coverage_oracle_system_prompt(self):
        return self.coverage_system_prompt

    @property
    def judge_coverage_binary_system_prompt(self):
        return self.coverage_system_prompt

    @property
    def judge_importance_binary_system_prompt(self):
        return self.importance_judge_system_prompt

    # ===== Backward compatibility: user prompt method aliases =====
    def get_factuality_oracle_batch_prompt(self, sents, src):
        return self.get_factuality_user_prompt(sents, src)

    def get_judge_factuality_binary_batch_prompt(self, sents, src):
        return self.get_factuality_user_prompt(sents, src)

    def get_importance_oracle_batch_prompt(self, sents, ref):
        return self.get_importance_oracle_user_prompt(sents, ref)

    def get_judge_importance_binary_batch_prompt(self, sents, src):
        return self.get_importance_judge_user_prompt(sents, src)

    def get_coverage_oracle_batch_prompt(self, sents, gen):
        return self.get_coverage_user_prompt(sents, gen)

    def get_judge_coverage_binary_batch_prompt(self, sents, gen):
        return self.get_coverage_user_prompt(sents, gen)


def get_prompt_config(dataset: str) -> PromptConfig:
    """Return prompt config for the given dataset."""
    key = dataset.lower().strip()
    if key not in DOMAIN_SPECS:
        raise ValueError(f"Unknown dataset: {dataset}. Expected one of: {list(DOMAIN_SPECS.keys())}")

    return PromptConfig(
        # Summarizer
        summarizer_system_prompt=get_summarizer_system_prompt(key),
        get_summarizer_user_prompt=lambda src, k=key: get_summarizer_user_prompt(k, src),

        # Factuality (shared)
        factuality_system_prompt=get_factuality_system_prompt(key),
        get_factuality_user_prompt=lambda sents, src, k=key: get_factuality_user_prompt(k, sents, src),

        # Factuality verification (two-pass)
        factuality_verification_system_prompt=get_factuality_verification_system_prompt(key),
        get_factuality_verification_user_prompt=lambda sent, src, k=key: get_factuality_verification_user_prompt(k, sent, src),

        # Importance - Oracle
        importance_oracle_system_prompt=get_importance_oracle_system_prompt(key),
        get_importance_oracle_user_prompt=lambda sents, ref, k=key: get_importance_oracle_user_prompt(k, sents, ref),

        # Importance - Judge
        importance_judge_system_prompt=get_importance_judge_system_prompt(key),
        get_importance_judge_user_prompt=lambda sents, src, k=key: get_importance_judge_user_prompt(k, sents, src),

        # Coverage (shared)
        coverage_system_prompt=get_coverage_system_prompt(key),
        get_coverage_user_prompt=lambda sents, gen, k=key: get_coverage_user_prompt(k, sents, gen),

        # Domain-specific 3-tier Judge prompts
        custom_factuality_triage_system_prompt=get_custom_factuality_triage_system_prompt(key),
        custom_importance_triage_system_prompt=get_custom_importance_triage_system_prompt(key),

        # 3-tier Judge prompts
        factuality_triage_system_prompt=get_factuality_triage_system_prompt(key),
        get_factuality_triage_user_prompt=lambda sents, src, k=key: get_factuality_triage_user_prompt(k, sents, src),
        importance_triage_system_prompt=get_importance_triage_system_prompt(key),
        get_importance_triage_user_prompt=lambda sents, src, k=key: get_importance_triage_user_prompt(k, sents, src),
        importance_triage_v2_system_prompt=get_importance_triage_v2_system_prompt(key),
        get_importance_triage_v2_user_prompt=lambda sents, src, k=key: get_importance_triage_v2_user_prompt(k, sents, src),
        importance_triage_fewshot_system_prompt=get_importance_triage_fewshot_system_prompt(key),
        get_importance_triage_fewshot_user_prompt=lambda sents, src, k=key: get_importance_triage_fewshot_user_prompt(k, sents, src),
        coverage_triage_system_prompt=get_coverage_triage_system_prompt(key),
        get_coverage_triage_user_prompt=lambda sents, gen, k=key: get_coverage_triage_user_prompt(k, sents, gen),

        # Single-label token-probability prompts
        factuality_token_probs_system_prompt=get_factuality_token_probs_system_prompt(key),
        get_factuality_token_probs_user_prompt=lambda sent, src, k=key: get_factuality_token_probs_user_prompt(k, sent, src),
        importance_token_probs_system_prompt=get_importance_token_probs_system_prompt(key),
        get_importance_token_probs_user_prompt=lambda sent, src, k=key: get_importance_token_probs_user_prompt(k, sent, src),
        coverage_token_probs_system_prompt=get_coverage_token_probs_system_prompt(key),
        get_coverage_token_probs_user_prompt=lambda sent, gen, k=key: get_coverage_token_probs_user_prompt(k, sent, gen),
    )


# ============================================================================
# 8. TESTING
# ============================================================================

if __name__ == '__main__':
    for dataset in ["aci", "meq"]:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*60}")

        config = get_prompt_config(dataset)

        print("\n--- Factuality (shared) ---")
        print(config.factuality_system_prompt)

        print("\n--- Importance Oracle (sees reference) ---")
        print(config.importance_oracle_system_prompt)

        print("\n--- Importance Judge (predicts) ---")
        print(config.importance_judge_system_prompt)

        print("\n--- Coverage (shared) ---")
        print(config.coverage_system_prompt)
