"""
Deterministic, label-agnostic unit filters for CRC pipeline.

These filters define what counts as a valid "unit" for conformal risk control.
Mathematically:
    U(X) = {u in Sent(X) : is_valid_source_sentence(u) = True}
    V(X) = {v in Sent(S_hat(X)) : is_valid_summary_sentence(v) = True}

All filters are:
- Deterministic (no randomness)
- Label-agnostic (don't look at Y_fact, Y_imp, Y_cov, or any scores)
- Applied identically to calibration and test

Design principle: only remove sentences that are PROVABLY non-clinical by
construction. Every rule must be airtight — no heuristics, no edge cases.
"""

import re
from typing import List


# ============================================================================
# Source: backchannel noise (phonetic acknowledgments, never clinical)
# ============================================================================

# These are verbal fillers with no semantic content in any context.
# Excludes "yes", "no", "correct", "sure", "right", "absolutely", "definitely"
# which can confirm/deny clinical information in dialogue.
BACKCHANNEL_EXACT = {
    "uh-huh", "uh huh", "uhuh",
    "mm-hmm", "mm hmm", "mmhmm", "mhm", "mmhm",
    "um", "uh", "um-hm", "umhm",
    "hmm", "hm",
}


# ============================================================================
# Source: BHC structural tags (MIMIC discharge note formatting)
# ============================================================================

# Angle-bracket section headers with no clinical content after the tag.
# BHC notes use <TAG> format for section breaks. Only filter when the content
# after the tag is empty or just deidentification placeholders (___).
# Does NOT filter tags with clinical content like <ALLERGIES> Codeine,
# <CHIEF COMPLAINT> confusion, <DISCHARGE DIAGNOSIS> ...
_BHC_EMPTY_TAG = re.compile(
    r'^<[A-Z][A-Z\s/&\-()]+>\s*(_+\.?\s*)?$'
)

# Specific tags that NEVER carry clinical content regardless of what follows.
# <SEX> M/F — demographic, not clinical
# <ATTENDING> ___ — provider name
# <SERVICE> MEDICINE/SURGERY — hospital service
_BHC_ADMIN_TAG = re.compile(
    r'^<(?:SEX|ATTENDING|SERVICE)>\s*\S*\.?\s*$'
)


# ============================================================================
# Summary: structural artifacts (formatting, not content)
# ============================================================================

# Full-line markdown headers — line must be ENTIRELY a header, no clinical
# content mixed in. Both patterns are end-anchored ($).
HEADER_PATTERNS = [
    r"^\*\*[A-Z][A-Za-z\s/&\-:()]+\*\*[\.:]*$",   # **ASSESSMENT AND PLAN**  **Brief Hospital Course (BHC) Narrative**.
    r"^\*\*[A-Za-z\s/]+:\*\*\.?$",                  # **Tests/Procedures:**
    r"^[A-Z][A-Z\s/&\-]+:\s*$",                     # IMPRESSION:  FINDINGS:  (plain-text radiology headers)
]

_header_regexes = [re.compile(p) for p in HEADER_PATTERNS]

# Placeholder patterns
PLACEHOLDER_PATTERNS = [
    r"\[Current Date\]",
    r"\[Redacted\]",
    r"\[Your Name\]",
    r"\[Doctor'?s?\s*Name\]",
    r"\[Patient'?s?\s*Name\]",
    r"\[DATE\]",
    r"\[NAME\]",
]

_placeholder_regexes = [re.compile(p, re.IGNORECASE) for p in PLACEHOLDER_PATTERNS]

# Signature/disclaimer patterns
SIGNATURE_PATTERNS = [
    r"^\*\*Signature",
    r"^\*\*Disclaimer",
    r"^Signature:",
    r"^Disclaimer:",
]

_signature_regexes = [re.compile(p, re.IGNORECASE) for p in SIGNATURE_PATTERNS]


# ============================================================================
# Core filter functions
# ============================================================================

def _strip_speaker_tag(text: str) -> str:
    """Remove speaker tags like [doctor], [patient], [nurse] from start of text."""
    return re.sub(r'^\[(?:doctor|patient|nurse|provider|clinician)\]\s*', '', text, flags=re.IGNORECASE)


def _normalize_source(text: str) -> str:
    """Lowercase, strip speaker tags, whitespace, and trailing punctuation."""
    text = text.strip()
    text = _strip_speaker_tag(text)
    text = text.lower()
    text = re.sub(r'[.!?,;:]+$', '', text).strip()
    return text


def is_valid_summary_sentence(sentence: str) -> bool:
    """
    Filter for generated summary sentences V(X).

    Removes (provably non-clinical):
    - Empty/whitespace-only strings
    - Full-line markdown headers (end-anchored: **ASSESSMENT AND PLAN**)
    - Placeholder-only sentences ([Current Date], [Redacted])
    - Signature/disclaimer blocks
    Does NOT remove:
    - Short diagnoses ("Hypertension.", "GERD.", "Type 2 diabetes.")
    - Numbered items with clinical content ("**1. Hypertension: controlled")
    - Any sentence with any clinical content regardless of length
    """
    text = sentence.strip()

    if not text:
        return False

    # Full-line markdown headers (both patterns are end-anchored)
    for regex in _header_regexes:
        if regex.match(text):
            return False

    # Placeholder-only: remove all placeholders, check if anything remains
    stripped = text
    for regex in _placeholder_regexes:
        stripped = regex.sub('', stripped)
    remaining = re.sub(r'[\*\s:.\-_]+', '', stripped)
    if not remaining:
        return False

    # Signature/disclaimer blocks
    for regex in _signature_regexes:
        if regex.match(text):
            return False

    return True


def is_valid_source_sentence(sentence: str, dataset: str = "aci") -> bool:
    """
    Filter for source sentences U(X).

    Removes (provably non-clinical):
    - Empty/whitespace-only strings
    - Backchannel noises ("uh-huh", "mm-hmm", "mhm")

    Does NOT remove:
    - "yes", "no", "okay", "sure", "correct" (can confirm/deny clinical content)
    - Greetings, farewells, pleasantries (tiny gain, not worth justifying)
    - Closing patterns (prefix-matching is error-prone)
    - Any sentence with any clinical content
    """
    text = sentence.strip()

    if not text:
        return False

    normalized = _normalize_source(text)

    if not normalized:
        return False

    # Backchannel noise only
    if normalized in BACKCHANNEL_EXACT:
        return False

    # Bare list numbers ("1.", "2.", "23.") — formatting artifacts from numbered lists
    if re.match(r'^\d+\.\s*$', text):
        return False

    # Roman numeral list markers ("II.", "III.", "IV.") — same as above
    if re.match(r'^[IVX]+\.\s*$', text):
        return False

    # Deidentification placeholders ("___", "___.")
    if re.match(r'^_+\.?\s*$', text):
        return False

    # PubMed citation placeholders ("<cit> .", "<cit>.")
    if re.match(r'^<cit>\s*\.?\s*$', text):
        return False

    # PubMed bare anonymized numbers ("<dig> .", "<dig>.")
    if re.match(r'^<dig>\s*\.?\s*$', text):
        return False

    # BHC angle-bracket section tags with no clinical content
    # Matches: <SOCIAL HISTORY> ___, <FOLLOWUP INSTRUCTIONS> ___, <ATTENDING> ___
    # Does NOT match: <ALLERGIES> Codeine, <CHIEF COMPLAINT> confusion
    if _BHC_EMPTY_TAG.match(text):
        return False

    # BHC administrative tags (never clinical regardless of content)
    # <SEX> M/F, <ATTENDING> ___, <SERVICE> MEDICINE
    if _BHC_ADMIN_TAG.match(text):
        return False
    
    # omop specific filters
    if re.match(r'^NOTE_DATETIME:\s*.+\|\s*NOTE_TITLE:\s*.+$', text):
        return False

    # Attending physician attestation boilerplate
    if "ATTENDING PHYSICIAN ATTESTATION" in text.upper():
        return False

    # Request ID lines: "Request ID: [0000...]"
    if re.match(r'^Request ID:\s*\[?\d+\]?\s*$', text):
        return False

    return True


def apply_filters(doc: dict, dataset: str = "aci") -> dict:
    """
    Compute filter masks for a document. Does NOT modify the document.

    Returns dict with:
        summary_mask: List[bool] - True = valid, for each generated sentence
        source_mask: List[bool] - True = valid, for each source sentence
        summary_removed: List[str] - removed summary sentences
        source_removed: List[str] - removed source sentences
    """
    summary_mask = [
        is_valid_summary_sentence(s) for s in doc['generated_sentences']
    ]
    source_mask = [
        is_valid_source_sentence(s, dataset) for s in doc['source_sentences']
    ]

    summary_removed = [
        s for s, valid in zip(doc['generated_sentences'], summary_mask) if not valid
    ]
    source_removed = [
        s for s, valid in zip(doc['source_sentences'], source_mask) if not valid
    ]

    return {
        'summary_mask': summary_mask,
        'source_mask': source_mask,
        'summary_removed': summary_removed,
        'source_removed': source_removed,
    }


def filter_arrays(arrays: dict, mask: List[bool]) -> dict:
    """
    Apply a boolean mask to multiple parallel arrays.

    Args:
        arrays: dict of {name: list} where all lists have same length as mask
        mask: boolean list, True = keep

    Returns:
        dict with same keys, filtered arrays
    """
    return {
        name: [v for v, m in zip(arr, mask) if m]
        for name, arr in arrays.items()
    }
