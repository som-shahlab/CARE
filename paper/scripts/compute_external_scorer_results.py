#!/usr/bin/env python3
"""Scorer-agnostic CRC experiment.

Replaces Phase 2 LLM-judge factuality and coverage scores with three external
scorer families. Importance is held fixed at the LLM-judge signal, so this
experiment tests agnosticism of the two scorers reviewers actually care about
(factuality + coverage), without conflating it with a salience proxy.

Scorer configurations:
  - nli_embed   : DeBERTa NLI (factuality forward, coverage reverse)
  - embed_only  : BERT CLS cosine (factuality + coverage max-sim)
  - alignscore  : AlignScore (factuality forward, coverage reverse)

Runs the same 100-resplit CRC / dev-tuned / Max-F1 calibration + evaluation
pipeline. Shows CRC maintains risk guarantees across scorer families while
Max-F1 violates catastrophically.

Requires GPU. AlignScore ckpt path set via ALIGNSCORE_CKPT env var.
"""
import json
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR

SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA_FACT = 0.15
ALPHA_OMIT = 0.15
GRID_FACT = 0.01
GRID_OMIT = 0.05

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EMBED_MODEL = "bert-base-uncased"
NLI_TOP_K = 5

ALIGNSCORE_CKPT = os.environ.get("ALIGNSCORE_CKPT", "")
ALIGNSCORE_MODEL = os.environ.get("ALIGNSCORE_MODEL", "roberta-base")
ALIGNSCORE_MODE = os.environ.get("ALIGNSCORE_MODE", "nli_sp")

RESULTS_PATH = OUTPUT_DIR / "external_scorer_results.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# External scoring functions
# ---------------------------------------------------------------------------
def load_nli_model():
    print(f"Loading NLI model: {NLI_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
    model.eval()
    return tokenizer, model


def load_embed_model():
    print(f"Loading embedding model: {EMBED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).to(device)
    model.eval()
    return tokenizer, model


def encode_sentences(sentences, tokenizer, model, batch_size=64):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state[:, 0, :]  # CLS token
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def cosine_sim_matrix(emb1, emb2):
    e1 = emb1 / emb1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    e2 = emb2 / emb2.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (e1 @ e2.T).numpy()


def score_nli_factuality(doc, nli_tok, nli_model, batch_size=32):
    """NLI entailment score per summary sentence (max over top-K source sents)."""
    src_sents = doc["source_sentences"]
    gen_sents = doc["generated_sentences"]
    if not gen_sents or not src_sents:
        return [0.5] * len(gen_sents)

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    try:
        all_vecs = tfidf.fit_transform(src_sents + gen_sents)
    except ValueError:
        return [0.5] * len(gen_sents)

    src_vecs = all_vecs[:len(src_sents)]
    gen_vecs = all_vecs[len(src_sents):]
    sim = sklearn_cosine(gen_vecs, src_vecs)

    scores = []
    pairs = []
    pair_map = []

    for j, g in enumerate(gen_sents):
        top_k_idx = np.argsort(sim[j])[-NLI_TOP_K:]
        for idx in top_k_idx:
            pairs.append((src_sents[idx], g))
            pair_map.append(j)

    entail_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        inputs = nli_tok(
            [p[0] for p in batch], [p[1] for p in batch],
            padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Label order for cross-encoder/nli-deberta-v3-base: [contradiction, entailment, neutral]
        entail_scores.extend(probs[:, 1].tolist())

    per_sent = [[] for _ in gen_sents]
    for idx, s in zip(pair_map, entail_scores):
        per_sent[idx].append(s)

    for j in range(len(gen_sents)):
        scores.append(max(per_sent[j]) if per_sent[j] else 0.5)

    return scores


def score_embedding_coverage(doc, embed_tok, embed_model):
    """BERT embedding cosine similarity: max sim per source sentence."""
    src_sents = doc["source_sentences"]
    gen_sents = doc["generated_sentences"]
    if not src_sents or not gen_sents:
        return [0.5] * len(src_sents)

    src_emb = encode_sentences(src_sents, embed_tok, embed_model)
    gen_emb = encode_sentences(gen_sents, embed_tok, embed_model)
    sim = cosine_sim_matrix(src_emb, gen_emb)
    coverage = sim.max(axis=1).tolist()
    return coverage


def score_embedding_factuality(doc, embed_tok, embed_model):
    """BERT embedding cosine similarity for factuality: max sim per summary sentence."""
    src_sents = doc["source_sentences"]
    gen_sents = doc["generated_sentences"]
    if not src_sents or not gen_sents:
        return [0.5] * len(gen_sents)

    src_emb = encode_sentences(src_sents, embed_tok, embed_model)
    gen_emb = encode_sentences(gen_sents, embed_tok, embed_model)
    sim = cosine_sim_matrix(gen_emb, src_emb)
    factuality = sim.max(axis=1).tolist()
    return factuality


def score_nli_coverage(doc, nli_tok, nli_model, batch_size=32):
    """NLI reverse-direction: does the summary entail each source sentence?

    For each source sentence, compute entailment score with premise=full_summary,
    hypothesis=source_sent. High entailment = source content is covered.
    """
    src_sents = doc["source_sentences"]
    gen_sents = doc["generated_sentences"]
    if not src_sents or not gen_sents:
        return [0.5] * len(src_sents)

    full_summary = " ".join(gen_sents)
    # Truncate summary to avoid exceeding model max length
    # (premise will be paired with each source sentence)
    pairs = [(full_summary, s) for s in src_sents]

    entail_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        inputs = nli_tok(
            [p[0] for p in batch], [p[1] for p in batch],
            padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Label order: [contradiction, entailment, neutral]
        entail_scores.extend(probs[:, 1].tolist())

    return entail_scores


def load_alignscore():
    """Lazy-load AlignScore. Requires `pip install alignscore` and a checkpoint."""
    if not ALIGNSCORE_CKPT:
        raise RuntimeError(
            "ALIGNSCORE_CKPT env var not set. Download AlignScore-base.ckpt from "
            "https://github.com/yuh-zha/AlignScore and set ALIGNSCORE_CKPT to its path."
        )
    from alignscore import AlignScore
    print(f"Loading AlignScore: {ALIGNSCORE_MODEL} ({ALIGNSCORE_MODE})")
    return AlignScore(
        model=ALIGNSCORE_MODEL,
        batch_size=32,
        device=str(device),
        ckpt_path=ALIGNSCORE_CKPT,
        evaluation_mode=ALIGNSCORE_MODE,
    )


def score_alignscore_factuality(doc, scorer):
    """AlignScore per summary sentence: context=full source, claim=summary_sent."""
    src_sents = doc["source_sentences"]
    gen_sents = doc["generated_sentences"]
    if not gen_sents or not src_sents:
        return [0.5] * len(gen_sents)
    full_source = " ".join(src_sents)
    contexts = [full_source] * len(gen_sents)
    scores = scorer.score(contexts=contexts, claims=gen_sents)
    return [float(s) for s in scores]


def score_alignscore_coverage(doc, scorer):
    """AlignScore reverse: context=full summary, claim=source_sent. High = covered."""
    src_sents = doc["source_sentences"]
    gen_sents = doc["generated_sentences"]
    if not src_sents or not gen_sents:
        return [0.5] * len(src_sents)
    full_summary = " ".join(gen_sents)
    contexts = [full_summary] * len(src_sents)
    scores = scorer.score(contexts=contexts, claims=src_sents)
    return [float(s) for s in scores]


def score_tfidf_importance(doc):
    """TF-IDF cosine similarity between each source sentence and reference summary."""
    src_sents = doc["source_sentences"]
    ref_text = doc.get("reference_text", "")
    if not src_sents or not ref_text:
        return [0.5] * len(src_sents)

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    try:
        all_vecs = tfidf.fit_transform(src_sents + [ref_text])
    except ValueError:
        return [0.5] * len(src_sents)

    src_vecs = all_vecs[:len(src_sents)]
    ref_vec = all_vecs[-1]
    sims = sklearn_cosine(src_vecs, ref_vec).flatten()
    # Normalize to [0, 1]
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())
    else:
        sims = np.full_like(sims, 0.5)
    return sims.tolist()


# ---------------------------------------------------------------------------
# Calibration + evaluation (reused from compute_resplit_cis.py)
# ---------------------------------------------------------------------------
def compute_factuality_loss(doc, threshold):
    labels = np.array(doc["factuality_labels"])
    probs = np.array(doc["factuality_probs"])
    if len(labels) == 0:
        return 0, 0, 0, 0
    flagged = probs <= threshold
    halluc = labels == 0
    loss = int((halluc & ~flagged).any())
    n_flagged = int(flagged.sum())
    tp = int((flagged & halluc).sum())
    total_halluc = int(halluc.sum())
    return loss, n_flagged, tp, total_halluc


def compute_omission_2d(doc, tau, gamma):
    imp = np.array(doc["importance_probs"])
    cov = np.array(doc["coverage_probs"])
    imp_lab = np.array(doc["importance_labels"])
    cov_lab = np.array(doc["coverage_labels"])
    if len(imp) == 0:
        return 0.0, 0, 0, 0
    surfaced = (imp >= tau) & ((1 - cov) >= gamma)
    true_omit = (imp_lab == 1) & (cov_lab == 0)
    n_true = int(true_omit.sum())
    n_surfaced = int(surfaced.sum())
    if n_true == 0:
        return 0.0, n_surfaced, 0, 0
    n_missed = int((true_omit & ~surfaced).sum())
    tp = int((surfaced & true_omit).sum())
    frac_loss = n_missed / n_true
    return float(frac_loss), n_surfaced, tp, n_true


def calibrate_factuality(cal_docs, alpha, grid=GRID_FACT):
    n = len(cal_docs)
    for lam in np.arange(0.0, 1.0 + grid, grid):
        losses = [compute_factuality_loss(d, lam)[0] for d in cal_docs]
        if (n / (n + 1)) * np.mean(losses) + 1 / (n + 1) <= alpha:
            return float(lam)
    return 1.0


def calibrate_factuality_devset(cal_docs, alpha, grid=GRID_FACT):
    for lam in np.arange(0.0, 1.0 + grid, grid):
        losses = [compute_factuality_loss(d, lam)[0] for d in cal_docs]
        if np.mean(losses) <= alpha:
            return float(lam)
    return 1.0


def calibrate_omission_2d(cal_docs, alpha, grid=GRID_OMIT):
    n = len(cal_docs)
    tau_vals = np.arange(0.0, 1.0 + grid, grid)
    gamma_vals = np.arange(0.0, 1.0 + grid, grid)
    feasible = []
    for tau in tau_vals:
        for gamma in gamma_vals:
            losses = [compute_omission_2d(d, tau, gamma)[0] for d in cal_docs]
            adj = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
            if adj <= alpha:
                wl = np.mean([compute_omission_2d(d, tau, gamma)[1] for d in cal_docs])
                feasible.append((tau, gamma, wl))
    if not feasible:
        return 0.0, 0.0
    feasible.sort(key=lambda x: (x[2], -x[0], -x[1]))
    return float(feasible[0][0]), float(feasible[0][1])


def calibrate_omission_2d_devset(cal_docs, alpha, grid=GRID_OMIT):
    tau_vals = np.arange(0.0, 1.0 + grid, grid)
    gamma_vals = np.arange(0.0, 1.0 + grid, grid)
    feasible = []
    for tau in tau_vals:
        for gamma in gamma_vals:
            losses = [compute_omission_2d(d, tau, gamma)[0] for d in cal_docs]
            if np.mean(losses) <= alpha:
                wl = np.mean([compute_omission_2d(d, tau, gamma)[1] for d in cal_docs])
                feasible.append((tau, gamma, wl))
    if not feasible:
        return 0.0, 0.0
    feasible.sort(key=lambda x: (x[2], -x[0], -x[1]))
    return float(feasible[0][0]), float(feasible[0][1])


def calibrate_factuality_maxf1(cal_docs, grid=GRID_FACT):
    """Pick lambda that maximizes F1 on cal set. No risk constraint."""
    best_lam, best_f1 = 1.0, -1.0
    for lam in np.arange(0.0, 1.0 + grid, grid):
        tp_total, fp_total, fn_total = 0, 0, 0
        for d in cal_docs:
            labels = np.array(d["factuality_labels"])
            probs = np.array(d["factuality_probs"])
            if len(labels) == 0:
                continue
            flagged = probs <= lam
            halluc = labels == 0
            tp_total += int((flagged & halluc).sum())
            fp_total += int((flagged & ~halluc).sum())
            fn_total += int((~flagged & halluc).sum())
        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_lam = float(lam)
    return best_lam


def calibrate_omission_maxf1(cal_docs, grid=GRID_OMIT):
    """Pick (tau, gamma) that maximizes F1 on cal set. No risk constraint."""
    best_tau, best_gamma, best_f1 = 0.0, 0.0, -1.0
    for tau in np.arange(0.0, 1.0 + grid, grid):
        for gamma in np.arange(0.0, 1.0 + grid, grid):
            tp_total, fp_total, fn_total = 0, 0, 0
            for d in cal_docs:
                imp = np.array(d["importance_probs"])
                cov = np.array(d["coverage_probs"])
                imp_lab = np.array(d["importance_labels"])
                cov_lab = np.array(d["coverage_labels"])
                if len(imp) == 0:
                    continue
                surfaced = (imp >= tau) & ((1 - cov) >= gamma)
                true_omit = (imp_lab == 1) & (cov_lab == 0)
                tp_total += int((surfaced & true_omit).sum())
                fp_total += int((surfaced & ~true_omit).sum())
                fn_total += int((~surfaced & true_omit).sum())
            prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
            rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_tau, best_gamma = float(tau), float(gamma)
    return best_tau, best_gamma


def evaluate_test(test_docs, lam, tau, gamma, lam_dev, tau_dev, gamma_dev,
                  lam_f1, tau_f1, gamma_f1):
    """Evaluate both controllers with CRC, dev-tuned, and Max-F1 thresholds."""
    results = {}
    for tag, l, t, g in [("crc", lam, tau, gamma), ("dev", lam_dev, tau_dev, gamma_dev),
                         ("maxf1", lam_f1, tau_f1, gamma_f1)]:
        fact_losses, fact_flagged, fact_tp, fact_total = [], [], [], []
        omit_losses, omit_flagged, omit_tp, omit_total = [], [], [], []
        for d in test_docs:
            fl, nf, tp, th = compute_factuality_loss(d, l)
            fact_losses.append(fl)
            fact_flagged.append(nf)
            fact_tp.append(tp)
            fact_total.append(th)
            ol, ns, otp, ot = compute_omission_2d(d, t, g)
            omit_losses.append(ol)
            omit_flagged.append(ns)
            omit_tp.append(otp)
            omit_total.append(ot)

        s_fact_tp = sum(fact_tp)
        s_fact_tot = sum(fact_total)
        s_omit_tp = sum(omit_tp)
        s_omit_tot = sum(omit_total)
        results[tag] = {
            "fact_viol": float(np.mean(fact_losses)),
            "fact_wl": float(np.mean(fact_flagged)),
            "fact_recall": s_fact_tp / s_fact_tot if s_fact_tot > 0 else 0.0,
            "omit_viol": float(np.mean(omit_losses)),
            "omit_wl": float(np.mean(omit_flagged)),
            "omit_recall": s_omit_tp / s_omit_tot if s_omit_tot > 0 else 0.0,
        }
    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def load_docs(dataset):
    path = DATA_ROOT / dataset / "phase_2" / "calibrated_scores.jsonl"
    docs = [json.loads(line) for line in open(path)]
    return docs


def apply_external_scores(docs, scorer_name, nli_tok, nli_model,
                          embed_tok, embed_model, align_scorer):
    """Replace LLM-judge factuality + coverage probs. Importance held fixed."""
    scored = []
    for doc in tqdm(docs, desc=f"Scoring ({scorer_name})"):
        d = dict(doc)  # shallow copy; importance_probs inherited from LLM judge

        if scorer_name == "nli_embed":
            d["factuality_probs"] = score_nli_factuality(d, nli_tok, nli_model)
            d["coverage_probs"] = score_nli_coverage(d, nli_tok, nli_model)
        elif scorer_name == "embed_only":
            d["factuality_probs"] = score_embedding_factuality(d, embed_tok, embed_model)
            d["coverage_probs"] = score_embedding_coverage(d, embed_tok, embed_model)
        elif scorer_name == "alignscore":
            d["factuality_probs"] = score_alignscore_factuality(d, align_scorer)
            d["coverage_probs"] = score_alignscore_coverage(d, align_scorer)
        else:
            raise ValueError(f"Unknown scorer: {scorer_name}")

        scored.append(d)
    return scored


def run_resplits(docs, n_resplits=N_RESPLITS, seed=SEED):
    """Run 100 resplits with CRC and dev-tuned calibration."""
    rng = random.Random(seed)
    n = len(docs)
    n_cal = int(n * CAL_FRAC)

    per_resplit = []
    for _ in tqdm(range(n_resplits), desc="Resplits"):
        indices = list(range(n))
        rng.shuffle(indices)
        cal_idx = indices[:n_cal]
        test_idx = indices[n_cal:]
        cal_docs = [docs[i] for i in cal_idx]
        test_docs = [docs[i] for i in test_idx]

        lam = calibrate_factuality(cal_docs, ALPHA_FACT)
        tau, gamma = calibrate_omission_2d(cal_docs, ALPHA_OMIT)
        lam_dev = calibrate_factuality_devset(cal_docs, ALPHA_FACT)
        tau_dev, gamma_dev = calibrate_omission_2d_devset(cal_docs, ALPHA_OMIT)
        lam_f1 = calibrate_factuality_maxf1(cal_docs)
        tau_f1, gamma_f1 = calibrate_omission_maxf1(cal_docs)

        result = evaluate_test(test_docs, lam, tau, gamma,
                               lam_dev, tau_dev, gamma_dev,
                               lam_f1, tau_f1, gamma_f1)
        per_resplit.append(result)

    # Bootstrap CI on the mean — matches resplit_cis.json conventions so
    # downstream table scripts can use identical CI logic for LLM-judge and
    # external-scorer results.
    boot_rng = np.random.default_rng(SEED)
    B = 10000

    summary = {}
    for tag in ["crc", "dev", "maxf1"]:
        for metric in ["fact_viol", "fact_wl", "fact_recall",
                       "omit_viol", "omit_wl", "omit_recall"]:
            vals = np.array([r[tag][metric] for r in per_resplit])
            idx = boot_rng.integers(0, len(vals), size=(B, len(vals)))
            boot_means = vals[idx].mean(axis=1)
            summary[f"{tag}_{metric}"] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(np.median(vals)),
                "ci_lower": float(np.percentile(boot_means, 2.5)),
                "ci_upper": float(np.percentile(boot_means, 97.5)),
                "per_split_pct_low": float(np.percentile(vals, 2.5)),
                "per_split_pct_high": float(np.percentile(vals, 97.5)),
            }
    return summary, per_resplit


def main():
    print(f"Device: {device}")

    scorers_to_run = os.environ.get(
        "EXTERNAL_SCORERS", "nli_embed,embed_only,alignscore"
    ).split(",")
    scorers_to_run = [s.strip() for s in scorers_to_run if s.strip()]
    print(f"Scorers: {scorers_to_run}")

    nli_tok, nli_model = (None, None)
    embed_tok, embed_model = (None, None)
    align_scorer = None

    if any(s in scorers_to_run for s in ["nli_embed"]):
        nli_tok, nli_model = load_nli_model()
    if "embed_only" in scorers_to_run:
        embed_tok, embed_model = load_embed_model()
    if "alignscore" in scorers_to_run:
        align_scorer = load_alignscore()

    all_results = {}
    if RESULTS_PATH.exists():
        all_results = json.load(open(RESULTS_PATH))
        print(f"Loaded {len(all_results)} existing results from {RESULTS_PATH}")

    for ds_key, meta in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {meta['label']} ({ds_key})")
        print(f"{'='*60}")

        docs = load_docs(ds_key)
        print(f"Loaded {len(docs)} documents")

        for scorer_name in scorers_to_run:
            key = f"{ds_key}__{scorer_name}"
            if key in all_results and "per_resplit" in all_results[key]:
                print(f"\n--- Scorer: {scorer_name} (cached with per_resplit, skipping) ---")
                continue
            if key in all_results:
                print(f"\n--- Scorer: {scorer_name} (cached but missing per_resplit, recomputing) ---")
            print(f"\n--- Scorer: {scorer_name} ---")
            scored_docs = apply_external_scores(
                docs, scorer_name, nli_tok, nli_model,
                embed_tok, embed_model, align_scorer,
            )

            print("Running resplits...")
            summary, per_resplit = run_resplits(scored_docs)

            all_results[key] = {
                "dataset": ds_key,
                "label": meta["label"],
                "scorer": scorer_name,
                "n_docs": len(docs),
                "summary": summary,
                "per_resplit": per_resplit,
            }

            # Print quick summary
            for tag in ["crc", "dev", "maxf1"]:
                fv = summary[f"{tag}_fact_viol"]["mean"] * 100
                ov = summary[f"{tag}_omit_viol"]["mean"] * 100
                fw = summary[f"{tag}_fact_wl"]["mean"]
                ow = summary[f"{tag}_omit_wl"]["mean"]
                fr = summary[f"{tag}_fact_recall"]["mean"] * 100
                orr = summary[f"{tag}_omit_recall"]["mean"] * 100
                print(f"  {tag.upper():4s}: FactV={fv:5.1f}% FlagF={fw:5.1f} RecF={fr:5.1f}% "
                      f"| OmitV={ov:5.1f}% FlagO={ow:5.1f} RecO={orr:5.1f}%")

            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
