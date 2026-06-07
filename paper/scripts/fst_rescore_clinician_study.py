#!/usr/bin/env python3
"""Re-score the clinician study under LTT-FST thresholds.

For each clinician-study response, recompute the omission surfaced set from
the current phase-2 calibrated probabilities under two calibration regimes:

  (a) Recorded            — the surfaced_indices the clinician actually saw
                            (whatever thresholds were live at study time).
  (b) 2D Joint recompute  — current probs, canonical 2D Joint thresholds.
                            Sanity check: should closely match (a).
  (c) FST recompute       — current probs, LTT-FST (τ+γ ordering) thresholds.
                            This is the rigorous variant.

Per-dataset thresholds (from paper/outputs/fst_smoketest_results.json):
  ACI-Bench   2D Joint (τ=0.40, γ=0.40)   FST (τ=0.50, γ=0.30)
  MIMIC-BHC   2D Joint (τ=0.40, γ=0.50)   FST (τ=0.40, γ=0.50)   (identical)

Outputs:
  - paper/outputs/fst_rescore_clinician.json
  - prints a side-by-side comparison to stdout.

Reuses the same design-faithful permutation test from
paper/scripts/table_14_clinician_study.py (v2 doc-level, v3 within-clinician).
"""
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = PROJECT_ROOT / "paper"
OUTPUT_DIR = PAPER_DIR / "outputs"
CLINICIAN_DIR = PAPER_DIR / "clinician_study"
DATA_ROOT = Path("/share/pi/nigam/projects/conf-summ")

CLINICIANS = [
    ("C1", "Chloe",   "clinician_study_v2_Chloe.json",   "clinician_study_v3_chloe.json"),
    ("C2", "Jenelle", "clinician_study_v2_Jenelle.json", "clinician_study_v3_jenelle.json"),
    ("C3", "Anson",   "clinician_study_v2_anson.json",   "clinician_study_v3_anson.json"),
]

THRESHOLDS = {
    "ACI_Bench":     {"two_d_joint": (0.40, 0.40), "fst": (0.50, 0.30)},
    "MIMIC_IV_BHC":  {"two_d_joint": (0.40, 0.50), "fst": (0.40, 0.50)},
}

PERM_ITERS = 20000
SEED = 0


def load_scored_docs():
    """Map (dataset, doc_id) -> scored doc record."""
    out = {}
    for ds in THRESHOLDS:
        path = DATA_ROOT / ds / "phase_2" / "calibrated_scores.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                out[(ds, d["doc_id"])] = d
    return out


def surfaced_indices(rec, tau, gamma):
    imp = np.asarray(rec["importance_probs"])
    cov = np.asarray(rec["coverage_probs"])
    non_cov = 1.0 - cov
    mask = (imp >= tau) & (non_cov >= gamma)
    return set(np.where(mask)[0].tolist())


def load_rows(scored, regime):
    """regime ∈ {'recorded', 'two_d_joint', 'fst'}.
    Returns one row per clinician response with safety_recall computed
    against the chosen surfaced set."""
    rows = []
    drops = 0
    for label, name, v2_file, v3_file in CLINICIANS:
        for ver, fname in [("v2", v2_file), ("v3", v3_file)]:
            data = json.load(open(CLINICIAN_DIR / fname))
            for r in data["responses"]:
                ds = r["dataset"]
                if regime == "recorded":
                    surfaced = (set(r.get("surfaced_indices", []))
                                if r["condition"] == "highlighted" else set())
                else:
                    rec = scored.get((ds, r["doc_id"]))
                    if rec is None:
                        drops += 1
                        continue
                    tau, gamma = THRESHOLDS[ds][regime]
                    if r["condition"] == "highlighted":
                        surfaced = surfaced_indices(rec, tau, gamma)
                    else:
                        surfaced = set()
                true_omit = set(r.get("true_omission_indices", []))
                yellow = set(r.get("yellow_highlighted_sentences", []))
                detected = true_omit & (yellow | surfaced)
                rows.append(dict(
                    label=label, clinician=name, version=ver,
                    doc_id=r["doc_id"], dataset=ds, condition=r["condition"],
                    n_yellow=len(yellow), n_surfaced=len(surfaced),
                    n_true_omit=len(true_omit), n_detected=len(detected),
                    safety_recall=(len(detected) / len(true_omit)
                                   if true_omit else np.nan),
                    yellow=yellow, true_omit=true_omit, surfaced=surfaced,
                ))
    if drops:
        print(f"  [warn] dropped {drops} rows missing phase-2 record")
    return rows


def attribution(rows):
    """% of clinician-yellowed true omissions that lie within CARE's surfaced set
    (CARE condition only)."""
    num, den = 0, 0
    for r in rows:
        if r["condition"] != "highlighted":
            continue
        yel_omit = r["true_omit"] & r["yellow"]
        den += len(yel_omit)
        num += len(yel_omit & r["surfaced"])
    return num / den if den else 0.0


def per_clinician(rows):
    out = []
    for label, name, _, _ in CLINICIANS:
        mine = [r for r in rows if r["clinician"] == name]
        ctrl = [r for r in mine if r["condition"] == "control"]
        care = [r for r in mine if r["condition"] == "highlighted"]
        def mean(xs, key):
            vals = [x[key] for x in xs
                    if not (isinstance(x[key], float) and np.isnan(x[key]))]
            return float(np.mean(vals)) if vals else 0.0
        out.append(dict(
            label=label, name=name,
            n_ctrl=len(ctrl), n_care=len(care),
            ctrl_recall=mean(ctrl, "safety_recall"),
            care_recall=mean(care, "safety_recall"),
        ))
    return out


def permutation_test(rows, metric="safety_recall", direction="greater",
                     n_iter=PERM_ITERS, seed=SEED):
    rng = np.random.default_rng(seed)
    valid = [r for r in rows
             if not (isinstance(r[metric], float) and np.isnan(r[metric]))]
    vals = np.array([r[metric] for r in valid])
    is_care = np.array([r["condition"] == "highlighted" for r in valid], dtype=bool)
    obs = vals[is_care].mean() - vals[~is_care].mean()

    v2_doc_to_idxs, v2_doc_orig_care = {}, {}
    for i, r in enumerate(valid):
        if r["version"] != "v2":
            continue
        k = (r["dataset"], r["doc_id"])
        v2_doc_to_idxs.setdefault(k, []).append(i)
        v2_doc_orig_care[k] = (r["condition"] == "highlighted")
    v2_docs = list(v2_doc_to_idxs.keys())
    v2_n_care = sum(1 for k in v2_docs if v2_doc_orig_care[k])

    v3_strata = {}
    for i, r in enumerate(valid):
        if r["version"] == "v3":
            v3_strata.setdefault(r["clinician"], []).append(i)

    null = np.empty(n_iter)
    for it in range(n_iter):
        perm = is_care.copy()
        order = rng.permutation(len(v2_docs))
        care_docs = set(v2_docs[i] for i in order[:v2_n_care])
        for k, idxs in v2_doc_to_idxs.items():
            new_cond = (k in care_docs)
            for i in idxs:
                perm[i] = new_cond
        for idxs in v3_strata.values():
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
    return float(obs), float(p)


def aggregate(rows):
    ctrl = [r for r in rows if r["condition"] == "control"]
    care = [r for r in rows if r["condition"] == "highlighted"]
    def mean(xs, key):
        vals = [x[key] for x in xs
                if not (isinstance(x[key], float) and np.isnan(x[key]))]
        return float(np.mean(vals)) if vals else 0.0
    delta, p = permutation_test(rows, "safety_recall", "greater")
    return dict(
        n_ctrl=len(ctrl), n_care=len(care),
        ctrl_recall=mean(ctrl, "safety_recall"),
        care_recall=mean(care, "safety_recall"),
        recall_delta=delta, recall_p=p,
        attribution=attribution(rows),
    )


def main():
    print("FST re-scoring of clinician study (v2 + v3 pooled)")
    print("=" * 72)
    scored = load_scored_docs()

    results = {}
    for regime in ["recorded", "two_d_joint", "fst"]:
        print(f"\nRegime: {regime}")
        rows = load_rows(scored, regime)
        agg = aggregate(rows)
        per_clin = per_clinician(rows)
        results[regime] = dict(aggregate=agg, per_clinician=per_clin)
        n = agg["n_ctrl"] + agg["n_care"]
        print(f"  n={n} ({agg['n_ctrl']} ctrl / {agg['n_care']} CARE)")
        print(f"  ctrl recall = {agg['ctrl_recall']:.4f}   "
              f"CARE recall = {agg['care_recall']:.4f}   "
              f"Δ = {agg['recall_delta']:+.4f}   p = {agg['recall_p']:.4f}")
        print(f"  attribution = {agg['attribution']:.1%}")
        for r in per_clin:
            d = r["care_recall"] - r["ctrl_recall"]
            print(f"    {r['label']} ({r['name']:<7}) "
                  f"ctrl={r['ctrl_recall']:.3f}  CARE={r['care_recall']:.3f}  "
                  f"Δ={d:+.3f}")

    print("\n" + "=" * 72)
    print("Side-by-side")
    print(f"{'Regime':<14} {'ctrl%':>7} {'CARE%':>7} {'Δ pp':>7} {'p':>8} {'attrib':>8}")
    for regime, r in results.items():
        a = r["aggregate"]
        print(f"{regime:<14} {a['ctrl_recall']*100:>7.1f} "
              f"{a['care_recall']*100:>7.1f} "
              f"{a['recall_delta']*100:>+7.1f} "
              f"{a['recall_p']:>8.4f} "
              f"{a['attribution']:>7.1%}")

    # Persist
    out_path = OUTPUT_DIR / "fst_rescore_clinician.json"
    serializable = {}
    for regime, r in results.items():
        serializable[regime] = {
            "aggregate": r["aggregate"],
            "per_clinician": r["per_clinician"],
        }
    serializable["thresholds"] = {
        ds: {k: list(v) for k, v in d.items()}
        for ds, d in THRESHOLDS.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
