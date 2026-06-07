"""
Configuration for the CARE pipeline.

This module defines all hyperparameters, model assignments, and file paths.
"""

import os
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# Paths
# ============================================================================

# Project root (repo root, one level up from care/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local data directory
DATA_DIR = PROJECT_ROOT / 'data'

# Canonical data root — override with CARE_DATA_ROOT env var
CANONICAL_DATA_ROOT = Path(
    os.environ.get("CARE_DATA_ROOT", "/share/pi/nigam/projects/conf-summ")
)

# Tokenizer paths (co-located with canonical data root)
LLAMA_TOKENIZER_DIR = str(CANONICAL_DATA_ROOT / "tokenizers" / "tokenizers" / "llama33_tokenizer")
O200K_PATH = CANONICAL_DATA_ROOT / "tokenizers" / "tokenizers" / "o200k_base.tiktoken"

# Paper outputs directory
PAPER_OUTPUT_DIR = PROJECT_ROOT / "paper" / "outputs"
PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default paths (overridden by configure_dataset)
SPLIT_DIR = CANONICAL_DATA_ROOT / 'split'
OUTPUT_DIR = DATA_DIR / 'experiment_v2_outputs'

# Phase-specific output directories (calibration)
SUMMARIES_DIR = OUTPUT_DIR / 'generated_summaries'  # Cache for generated summaries
PHASE1_DIR = CANONICAL_DATA_ROOT / 'phase_1'
PHASE2_DIR = CANONICAL_DATA_ROOT / 'phase_2'
PHASE3_DIR = OUTPUT_DIR / 'phase3_thresholds'
PHASE4_DIR = OUTPUT_DIR / 'phase4_test_results'

# Phase-specific output directories (test)
TEST_PHASE1_DIR = OUTPUT_DIR / 'test_phase1_oracle_labels'
TEST_PHASE2_DIR = OUTPUT_DIR / 'test_phase2_calibration'

# Input files
CALIBRATION_FILE = SPLIT_DIR / 'calibration.jsonl'
TEST_FILE = SPLIT_DIR / 'test.jsonl'
ALL_DATA_FILE = SPLIT_DIR / 'all_data.jsonl'
SPLIT_INDICES_FILE = SPLIT_DIR / 'split_indices.json'

# ============================================================================
# Dataset Registry (used by paper table/figure scripts)
# ============================================================================

DATASETS = {
    "ACI_Bench":     {"label": "ACI-Bench",  "short": "ACI"},
    "MIMIC_IV_BHC":  {"label": "MIMIC-BHC",  "short": "BHC"},
    "MIMIC_III_CXR": {"label": "MIMIC-CXR",  "short": "CXR"},
    "OMOP":          {"label": "Priv-DS",     "short": "Priv-DS"},
    "SumPubMed":     {"label": "SumPubMed",   "short": "PubMed"},
}

def get_phase4_results(dataset: str) -> Path:
    return CANONICAL_DATA_ROOT / dataset / "phase_4" / "test_results.json"

def get_phase4_alpha_sweep(dataset: str) -> Path:
    return CANONICAL_DATA_ROOT / dataset / "phase_4" / "alpha_sweep_results.json"

def get_phase3_thresholds(dataset: str) -> Path:
    return CANONICAL_DATA_ROOT / dataset / "phase_3" / "conformal_thresholds.json"

def get_phase1_labels(dataset: str) -> Path:
    return CANONICAL_DATA_ROOT / dataset / "phase_1" / "oracle_labels.jsonl"

def get_split_file(dataset: str, split: str) -> Path:
    return CANONICAL_DATA_ROOT / dataset / "split" / f"{split}.jsonl"

def get_phase2_scores(dataset: str, judge: str = None) -> Path:
    suffix = f"_{judge}" if judge else ""
    return CANONICAL_DATA_ROOT / dataset / f"phase_2{suffix}" / "calibrated_scores.jsonl"

# ============================================================================
# Model Configuration
# ============================================================================

# Model assignments following plan.tex recommendations.
# All three are overridable via env vars. Defaults match the paper's primary
# configuration. To use a self-hosted judge, set CRC_JUDGE_MODEL='local:/path/to/model';
# for Bedrock/Gemini, e.g. 'bedrock/us.anthropic.claude-opus-4-20250514-v1:0' or
# 'gemini-2.5-pro-preview-05-06' (see get_azure_endpoint).
SUMMARIZER_MODEL = os.environ.get('CRC_SUMMARIZER_MODEL', 'llama-3.3-70b-instruct')  # Black-box abstractive summarizer f(X)
ORACLE_MODEL = os.environ.get('CRC_ORACLE_MODEL', 'gpt-5')        # Strong oracle for Y_fact and Y_imp
JUDGE_MODEL = os.environ.get('CRC_JUDGE_MODEL', 'gpt-5-mini')     # Vote-rate scorer (m=5 replicates)

# Azure API configuration (Stanford Healthcare APIM)
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(PROJECT_ROOT / '.env')

SECUREGPT_API_KEY = os.getenv('SECUREGPT_API_KEY')
AZURE_API_VERSION = '2024-12-01-preview'

BEDROCK_API_KEY = os.getenv('BEDROCK_API_KEY', '')


def is_local_model(model: str) -> bool:
    """Return True if the model name refers to a local path (local: prefix)."""
    return model.startswith("local:")


def get_local_model_path(model: str) -> str:
    """Extract the filesystem path from a 'local:<path>' model name."""
    return model[len("local:"):]


def get_azure_endpoint(model: str) -> str:
    """Get API endpoint for a given model."""
    if is_local_model(model):
        return ""
    if 'llama' in model.lower():
        return "<LLAMA_MODEL_URL>" # replace with actual URL for llama-3.3-70b-instruct
    elif model.lower().startswith('bedrock/'):
        return "<BEDROCK_MODEL_URL>" # replace with actual URL for Bedrock model
    elif 'claude' in model.lower():
        return "<CLAUDE_MODEL_URL>" # replace with actual URL for Claude model
    elif 'gemini' in model.lower():
        return "<GEMINI_MODEL_URL>" # replace with actual URL for Gemini model
    else:
        return "<GPT_MODEL_URL>" # replace with actual URL for GPT-5 and GPT-5-mini

# Request settings
MAX_RETRIES = 3
TIMEOUT = 300  # 5 minutes for long generations
TEMPERATURE = 0.0  # Deterministic by default
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls

# ============================================================================
# Phase 1: Oracle Labeling
# ============================================================================

# Summary generation settings
SUMMARIZER_MAX_TOKENS = 8000  # High limit for generation
SUMMARIZER_TEMPERATURE = 0.0  # Deterministic for reproducibility

# Oracle labeling settings
ORACLE_MAX_TOKENS = 8000  # Batch labeling needs space for JSON arrays
ORACLE_TEMPERATURE = 0.0  # Deterministic
ORACLE_BATCH_SIZE = 15  # Sentences per batch (verification catches errors, so larger batch OK)
ORACLE_VERIFY_NO_ANSWERS = True  # Two-pass verification for sentences marked NO

# Batch processing
BATCH_SIZE = 10  # Process N documents at a time
SAVE_FREQUENCY = 10  # Save checkpoint every N documents

# ============================================================================
# Phase 2: Calibration
# ============================================================================

# Judge scoring settings
JUDGE_MAX_TOKENS = 8000  # Batch JSON responses
JUDGE_TEMPERATURE = 0.5
JUDGE_TOP_P = 0.9
JUDGE_BATCH_SIZE = 10  # Sentences per batch (smaller batches improve accuracy)

# Three-tier vote-rate scoring (plan.tex Section 2)
# With m=5 and 3 tiers (0, 0.5, 1.0), we get 11 discrete probability values
VOTE_RATE_REPLICATES = 5  # Number of replicates (m)

# Logistic calibration settings (legacy, kept for compatibility)
CALIBRATION_MAX_ITER = 1000
CALIBRATION_SOLVER = 'lbfgs'

# ============================================================================
# Phase 3: Conformal Risk Control
# ============================================================================

# Threshold grid search
LAMBDA_GRID_SIZE = 1000  # Number of lambda values to search
LAMBDA_MIN = 0.0
LAMBDA_MAX = 1.0

# Risk levels to evaluate (rate-based: 0.10 = 10% missed error rate)
ALPHA_FACT_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]  # Factuality risk levels
ALPHA_OMIT_LEVELS = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50]  # Omission risk levels (relaxed)

# ============================================================================
# Phase 4: Testing
# ============================================================================

# Test evaluation settings
COMPUTE_BASELINES = True
BASELINES = [
    'naive',              # No filtering, just raw summary
    'topk_factuality',    # Top-k factuality filtering
    'factuality_only',    # Only factuality CRC
    'importance_only',    # Only importance CRC
]

# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = os.environ.get('CRC_LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# Helper Functions
# ============================================================================

def get_config_dict() -> Dict[str, Any]:
    """Return configuration as dictionary for serialization."""
    return {
        'summarizer_model': SUMMARIZER_MODEL,
        'oracle_model': ORACLE_MODEL,
        'judge_model': JUDGE_MODEL,
        'vote_rate_replicates': VOTE_RATE_REPLICATES,
        'lambda_grid_size': LAMBDA_GRID_SIZE,
        'alpha_fact_levels': ALPHA_FACT_LEVELS,
        'alpha_omit_levels': ALPHA_OMIT_LEVELS,
        'temperature': TEMPERATURE,
        'batch_size': BATCH_SIZE,
    }

def configure_dataset(dataset: str):
    """
    Configure dataset-specific input/output paths.

    Input data (split, Phase 1-4) lives at CANONICAL_DATA_ROOT/{dataset}/.
    Experiment outputs (summaries, etc.) go to DATA_DIR/{dataset}/.
    """
    name = dataset.lower().strip()

    if name in {"aci", "aci-bench", "acibench"}:
        folder = "ACI_Bench"
    elif name in {"meq", "meqsum"}:
        folder = "MeQSum"
    elif name in {"bhc", "mimic-iv", "mimic-bhc"}:
        folder = "MIMIC_IV_BHC"
    elif name in {"cxr", "mimic-iii", "mimic-cxr"}:
        folder = "MIMIC_III_CXR"
    elif name in {"pubmed"}:
        folder = "SumPubMed"
    elif name in {"omop", "priv-ds", "privds"}:
        folder = "OMOP"
    else:
        raise ValueError(
            f"Unknown dataset={dataset}. Expected one of: aci, meq, bhc, cxr, pubmed, priv-ds"
        )

    global SPLIT_DIR, OUTPUT_DIR
    global SUMMARIES_DIR, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR
    global TEST_PHASE1_DIR, TEST_PHASE2_DIR
    global CALIBRATION_FILE, TEST_FILE, ALL_DATA_FILE, SPLIT_INDICES_FILE

    # Canonical data at Carina path (inputs: split, Phase 1, Phase 2)
    CANONICAL_DIR = CANONICAL_DATA_ROOT / folder
    SPLIT_DIR = CANONICAL_DIR / "split"
    PHASE1_DIR = CANONICAL_DIR / "phase_1"
    PHASE2_DIR = CANONICAL_DIR / "phase_2"

    # Experiment outputs stay at old path (Phase 3, Phase 4, summaries)
    OUTPUT_DIR = DATA_DIR / "experiment_v2_outputs" / folder
    SUMMARIES_DIR = OUTPUT_DIR / "generated_summaries"
    PHASE3_DIR = OUTPUT_DIR / "phase3_thresholds"
    PHASE4_DIR = OUTPUT_DIR / "phase4_test_results"
    TEST_PHASE1_DIR = OUTPUT_DIR / "test_phase1_oracle_labels"
    TEST_PHASE2_DIR = OUTPUT_DIR / "test_phase2_calibration"

    # Input files (from canonical split dir)
    CALIBRATION_FILE = SPLIT_DIR / "calibration.jsonl"
    TEST_FILE = SPLIT_DIR / "test.jsonl"
    ALL_DATA_FILE = SPLIT_DIR / "all_data.jsonl"
    SPLIT_INDICES_FILE = SPLIT_DIR / "split_indices.json"



def setup_directories():
    """Create output directories if they don't exist.

    Only creates Phase 3/4 output dirs — canonical data dirs (split, Phase 1,
    Phase 2) are read-only and pre-populated by reorganize_data.py.
    """
    for dir_path in [OUTPUT_DIR, SUMMARIES_DIR, PHASE3_DIR, PHASE4_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def load_split_indices() -> Dict[str, Any]:
    """
    Load split indices from split_indices.json.

    Returns:
        Dict with 'calibration_indices' and 'test_indices' lists
    """
    import json
    if not SPLIT_INDICES_FILE.exists():
        raise FileNotFoundError(f"Split indices file not found: {SPLIT_INDICES_FILE}")

    with open(SPLIT_INDICES_FILE, 'r') as f:
        return json.load(f)


def filter_documents_by_split(documents: list, split: str) -> list:
    """
    Filter documents to only include those in the specified split.

    Args:
        documents: List of documents with 'doc_id' field
        split: Either 'calibration' or 'test'

    Returns:
        Filtered list of documents
    """
    indices = load_split_indices()

    if split == 'calibration':
        valid_ids = set(indices['calibration_indices'])
    elif split == 'test':
        valid_ids = set(indices['test_indices'])
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'calibration' or 'test'")

    return [doc for doc in documents if doc.get('doc_id') in valid_ids]


if __name__ == '__main__':
    # Test configuration
    print("Configuration loaded successfully!")
    print(f"\nData directories:")
    print(f"  CALIBRATION_FILE: {CALIBRATION_FILE}")
    print(f"  TEST_FILE: {TEST_FILE}")
    print(f"\nModels:")
    print(f"  Summarizer: {SUMMARIZER_MODEL}")
    print(f"  Oracle: {ORACLE_MODEL}")
    print(f"  Judge: {JUDGE_MODEL}")
    print(f"\nPhase 3 settings:")
    print(f"  Alpha (fact): {ALPHA_FACT_LEVELS}")
    print(f"  Alpha (omit): {ALPHA_OMIT_LEVELS}")

    # Create directories
    setup_directories()
    print(f"\n✓ Created output directories")
