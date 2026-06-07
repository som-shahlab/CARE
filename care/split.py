#!/usr/bin/env python3
"""
Data splitting and sentence extraction.

This script:
1. Loads train.jsonl and test.jsonl from the specified data directory
2. Combines them and performs a random split (default: 70/30 split)
3. Performs sentence splitting:
   - Source (X): Split dialogue by [doctor] and [patient] markers (ACI-Bench) or by sentence boundaries (MeQSum)
   - Reference summary (S): Split by sentence boundaries (periods, newlines)
4. Saves split data to data/split/

Output format per document:
{
    "doc_id": <int>,
    "source_text": <str>,  # Full dialogue
    "source_sentences": [<str>, ...],  # U(X) = Sent(X)
    "reference_text": <str>,  # Full reference summary
    "reference_sentences": [<str>, ...],  # Sent(S)
}
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Deal with common abbreviations
ABBREV_PATTERNS = [
        r'\b(Dr|Mr|Mrs|Ms|Prof|vs|etc|Jr|Sr|St|Corp|Inc|Ltd|pt|dx|tx)\.',
        r'\b(e\.g|i\.e|a\.m|p\.m)\.',
        r'\b[A-Z]\.',  # Single capital letter (genus names, initials)
    ]

ABBREV_PLACEHOLDER = "<<ABBREV>>"

# Deal with headers and bullets
HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 /&()\-]{2,}:\s*$")
TAG_HEADER_RE = re.compile(r"^<[^<>]{2,60}>$")
BULLET_RE = re.compile(r"^\s*([-*•]|\d+\.)\s+")

def _newline_before_tags(text: str) -> str:
    """
    Insert newlines around <TAG> headers so they behave like section headers.
    Example: "foo <BAR> baz" -> "foo\n<BAR>\nbaz"
    """
    return re.sub(r"\s*(<[^<>]{2,60}>)\s*", r"\n\1\n", text)

def is_header_line(line: str) -> bool:
    if TAG_HEADER_RE.match(line) or HEADER_RE.match(line):
        return True
    # fallback heuristics (keep conservative)
    if (line.isupper() and len(line.split()) <= 6 and len(line) >= 4):
        return True
    if (line.endswith(":") and len(line.split()) <= 10 and len(line) >= 4):
        return True
    return False

def protect_abbreviations(text: str) -> tuple[str, list[str]]:
    """Replace abbreviations with placeholder, return protected text and originals."""
    protected_items = []

    def _repl(m):
        protected_items.append(m.group(0))
        return f"<<ABBREV{len(protected_items)-1}>>"

    protected = text
    for pattern in ABBREV_PATTERNS:
        protected = re.sub(pattern, _repl, protected, flags=re.IGNORECASE)

    return protected, protected_items

def restore_abbreviations(text: str, protected_items: list[str]) -> str:
    """Restore abbreviations from placeholder."""
    for i, orig in enumerate(protected_items):
        text = text.replace(f"<<ABBREV{i}>>", orig)
    return text

def normalize_punct_spacing(text: str) -> str:
    """Normalize spacing around punctuation."""
    if not text:
        return text
    # collapse whitespace first
    text = re.sub(r"\s+", " ", text)

    # remove spaces before punctuation: "word ." -> "word."
    text = re.sub(r"\s+([.!?])", r"\1", text)

    # ensure a space AFTER punctuation when followed by a letter or bracket: ".well" -> ". well"
    text = re.sub(r"([.!?])(?=[A-Za-z\[])", r"\1 ", text)

    # clean up extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def split_dialogue_into_sentences(dialogue: str) -> List[str]:
    """
    Split patient-doctor dialogue into sentences.

    Each turn by [doctor] or [patient] is treated as a sentence.
    Removes the speaker tag and strips whitespace.

    Args:
        dialogue: Raw dialogue string with [doctor] and [patient] tags

    Returns:
        List of dialogue sentences (turns)
    """
    # Split by speaker tags
    pattern = r'\[(doctor|patient)\]\s*'

    if not re.search(pattern, dialogue or "", flags=re.IGNORECASE):
        # Fallback: treat as plain text
        return split_source_into_sentences(dialogue)
    
    parts = re.split(pattern, dialogue)

    sentences = []
    i = 1  # Start at 1 to skip initial empty string
    while i < len(parts):
        if i + 1 < len(parts):
            speaker = parts[i]  # 'doctor' or 'patient'
            utterance = parts[i + 1].strip()
            if utterance:  # Only add non-empty utterances
                # Split utterance into sentences, then tag each
                utterance = normalize_punct_spacing(utterance)
                protected, protected_items = protect_abbreviations(utterance)
                sub_parts = re.split(r"(?<!\d)(?<=[.!?])(?!\d)\s+", protected)
                for sub in sub_parts:
                    sub = restore_abbreviations(sub, protected_items)
                    sub = sub.strip()
                    # Include speaker tag for context
                    if sub:
                        sentences.append(f"[{speaker}] {sub}")
            i += 2
        else:
            break

    return sentences


def split_source_into_sentences(source: str) -> List[str]:
    """
    Split MeQSum-style input into sentence-like units.

    Steps:
    1. Remove SUBJECT: ... MESSAGE: headers (keep only message body)
    2. Normalize whitespace
    3. Split on newlines / semicolons
    4. Split on sentence boundaries (.?!), but NOT decimals like 78.00
    """

    if not source:
        return []

    text = source.strip()

    # 1) Strip SUBJECT / MESSAGE headers
    #    Example:
    #    "SUBJECT: foo bar MESSAGE: actual message here"
    # Keep everything AFTER "MESSAGE:" if it exists
    message_match = re.search(r"\bMESSAGE\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if message_match:
        text = message_match.group(1)
    else:
        # If MESSAGE: is missing, drop SUBJECT: prefix if present
        text = re.sub(r"^\s*SUBJECT\s*:\s*", "", text, flags=re.IGNORECASE)

    # 2) Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_punct_spacing(text)

    if not text:
        return []

    units: List[str] = []

    # 3) Split first on semicolons / newlines
    for chunk in re.split(r"[;\n]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk = normalize_punct_spacing(chunk)

        # Protect abbreviations before splitting
        protected, protected_items = protect_abbreviations(chunk)

        # 4) Sentence split on .?! BUT avoid decimals (e.g., 78.00)
        #
        # Explanation:
        # - (?<!\d) ensures the punctuation is NOT preceded by a digit
        # - (?!\d) ensures it is NOT followed by a digit
        # This prevents splitting 78.00 or 3.14
        parts = re.split(r"(?<!\d)(?<=[.!?])(?!\d)\s+", protected)

        for part in parts:
            # Restore protected abbreviations
            part = restore_abbreviations(part, protected_items)
            part = part.strip()
            if part:
                units.append(part)

    return units


def split_reference_into_sentences(reference: str) -> List[str]:
    """
    Split reference summary into sentences.

    Uses period followed by space or newline as sentence boundary.
    Also splits on newlines to handle multi-paragraph structure.

    Args:
        reference: Reference summary text

    Returns:
        List of reference sentences
    """
    # First, normalize whitespace and split by periods
    # Handle "PLAN", "ASSESSMENT AND PLAN", section headers
    reference = re.sub(r"\n\s*\n+", "\n\n", reference)
    reference = reference.replace("\n\n", "<<PARA>>")
    reference = re.sub(r"\s*\n\s*", " ", reference)
    reference = reference.replace("<<PARA>>", "\n\n")

    # Split by sentence boundaries (period + space/newline)
    # But preserve section headers
    sentences = []

    # Split by newlines first to preserve paragraph structure

    paras = [p.strip() for p in reference.split("\n\n") if p.strip()]

    merged = paras

    for para in merged:
        para = para.strip()
        if not para:
            continue
        
        para = normalize_punct_spacing(para)

        # Check if it's a header (all caps or ends with colon)
        if para.isupper() or para.endswith(':'):
            sentences.append(para)
        else:
            # Protect abbreviations before splitting
            protected, protected_items = protect_abbreviations(para)
    
            # Split by periods, but be careful with abbreviations
            # Simple split for now - can refine later
            parts = re.split(r"(?<!\d)(?<=[.!?])(?!\d)\s+", protected)

            stitched = []
            j = 0
            while j < len(parts):
                cur = parts[j].strip()
                if re.match(r"^\d+\.$", cur) and j + 1 < len(parts):
                    nxt = parts[j + 1].strip()
                    stitched.append(cur + " " + nxt)
                    j += 2
                else:
                    stitched.append(parts[j])
                    j += 1

            parts = stitched

            for part in parts:
                # Restore protected abbreviations
                part = restore_abbreviations(part, protected_items)
                part = part.strip()
                if part:
                    if not re.search(r"[.!?:]$", part):
                        part = part + "."
                    sentences.append(part)


    return sentences


def split_note_into_sentences(note: str) -> List[str]:
    """
    Split a clinical note into sentence units. 
    Note: 
        - header is NOT its own unit
        - each unit is prefixed with the most recent header (if any)
        - headers do NOT stack (only most recent header is used)

     Args:
        note: Source note text

    Returns:
        List of source note sentences
    """
    if not note:
        return []

    units: List[str] = []
    current_header: str | None = None
    header_pending: bool = False

    note = _newline_before_tags(note)

    # helper to prefix with the current header (single header only)
    def _with_header(s: str) -> str:
        nonlocal header_pending
        s = s.strip()
        if not s:
            return ""
        if current_header and header_pending:
            header_pending = False
            return f"{current_header} {s}"
        return s

    for raw in note.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Header line: update state, but DO NOT emit a unit
        if is_header_line(line):
            current_header = line
            header_pending = True
            continue

        # Bullet / list item
        if BULLET_RE.match(line):
            item = normalize_punct_spacing(line)
            out = _with_header(item)
            if out:
                units.append(out)
            continue

        # Prose line: sentence split
        line = normalize_punct_spacing(line)
        protected, protected_items = protect_abbreviations(line)
        parts = re.split(r"(?<!\d)(?<=[.!?])(?!\d)\s+", protected)

        for p in parts:
            p = restore_abbreviations(p, protected_items).strip()
            out = _with_header(p)
            if out:
                units.append(out)

    return units


def process_document(doc: Dict[str, Any], doc_id: int, dataset: str) -> Dict[str, Any]:
    """
    Process a single document: extract source and reference sentences.

    Args:
        doc: Raw document dict with 'inputs' and 'target' keys
        doc_id: Unique document identifier

    Returns:
        Processed document with sentence splits
    """
    source_text = doc['inputs']
    reference_text = doc['target']

    if dataset == "aci": # dialogue-type
        source_sentences = split_dialogue_into_sentences(source_text)

    elif dataset == "bhc": # note-type 
        source_sentences = split_note_into_sentences(source_text)
    
    else: # other (meq, cxr)
        source_sentences = split_source_into_sentences(source_text)
    
    reference_sentences = split_reference_into_sentences(reference_text)
        
    return {
        'doc_id': doc_id,
        'source_text': source_text,
        'source_sentences': source_sentences,
        'reference_text': reference_text,
        'reference_sentences': reference_sentences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ACI-Bench",
                        help="Directory containing train.jsonl and test.jsonl")
    
    parser.add_argument("--output_dir", type=str, default="data/split/ACI-Bench",
                        help="Where to write calibration.jsonl and test.jsonl")
    
    # Optional single jsonl file to process
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Optional single JSONL file to split into calibration/test"
    )
    
    # Dataset flag to dictate sentence splitting 
    parser.add_argument(
        "--dataset",
        choices=["aci", "meq", "bhc", "cxr"],
        default="meq",
        help="Dataset type for sentence splitting"
    )
    
    # Keep the original train/test split provived 
    parser.add_argument("--keep_train_test", action="store_true",
                        help="If set: train -> calibration and test -> test (no reshuffle/re-split)")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--calib_frac", type=float, default=0.7,
                        help="Only used if --keep_train_test is NOT set")

    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir} ...")

    if args.input_jsonl is not None:
        # Single-file mode
        all_data = load_jsonl(Path(args.input_jsonl))
        print(f"Loaded {len(all_data)} documents from single JSONL")

        random.shuffle(all_data)
        n_total = len(all_data)
        n_calib = int(args.calib_frac * n_total)

        calib_data = all_data[:n_calib]
        test_split = all_data[n_calib:]

    else:
        # Existing train/test mode (unchanged)
        train_data = load_jsonl(data_dir / "train.jsonl")
        test_data = load_jsonl(data_dir / "test.jsonl")

        if args.keep_train_test:
            calib_data = train_data
            test_split = test_data
            print(f"Keeping original split: calib={len(calib_data)} (train), test={len(test_split)} (test)")
        else:
            all_data = train_data + test_data
            random.shuffle(all_data)
            n_total = len(all_data)
            n_calib = int(args.calib_frac * n_total)
            calib_data = all_data[:n_calib]
            test_split = all_data[n_calib:]
            print(f"Resplit: calib={len(calib_data)} ({args.calib_frac:.2f}), test={len(test_split)}")


    print("\nProcessing calibration set...")
    calib_processed = []
    for i, doc in enumerate(calib_data):
        processed = process_document(doc, doc_id=i, dataset=args.dataset)
        calib_processed.append(processed)
        if i == 0:
            print("\nExample document (calib_0):")
            print(f"  - Source sentences: {len(processed['source_sentences'])}")
            print(f"  - Reference sentences: {len(processed['reference_sentences'])}")
            if processed["source_sentences"]:
                print(f"  - First source sentence: {processed['source_sentences'][0][:80]}...")

    print("\nProcessing test set...")
    test_processed = []
    base_id = len(calib_processed)
    for i, doc in enumerate(test_split):
        processed = process_document(doc, doc_id=base_id + i, dataset=args.dataset)
        test_processed.append(processed)

    calib_path = output_dir / "calibration.jsonl"
    test_path = output_dir / "test.jsonl"

    print(f"\nSaving to disk...")
    with open(calib_path, "w") as f:
        for doc in calib_processed:
            f.write(json.dumps(doc) + "\n")
    print(f"  - {calib_path}")

    with open(test_path, "w") as f:
        for doc in test_processed:
            f.write(json.dumps(doc) + "\n")
    print(f"  - {test_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()