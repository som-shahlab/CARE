"""R3 Q2 / R4: decompose the clinician-study endpoint.

Per review: T = oracle true omissions, Y = clinician-highlighted source sentences,
S = CARE-surfaced sentences shown to the clinician (CARE arm only).
Paper metric (review-time coverage): |T ∩ (Y ∪ S)| / |T|, averaged over reviews.
"""
import json, re, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from care.utils import split_summary_into_sentences
from care.filters import is_valid_summary_sentence

D = Path(__file__).resolve().parents[1] / "clinician_study"
rng = np.random.default_rng(0)


def norm(s):
    s = re.sub(r"[*_#`]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def boot(x, B=10000):
    x = np.asarray([v for v in x if v == v], float)
    if not len(x): return (np.nan, np.nan, np.nan)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


rows = []
for c in ["C1", "C2", "C3"]:
    for v in ["v2", "v3"]:
        for r in json.load(open(D / f"clinician_study_{v}_{c}.json"))["responses"]:
            T, Y = set(r["true_omission_indices"]), set(r["yellow_highlighted_sentences"])
            care = r["condition"] == "highlighted"
            S = set(r["surfaced_indices"]) if care else set()
            row = dict(clin=c, ver=v, ds=r["dataset"], care=care, n_src=r["n_src"], time=r["time_seconds"],
                       n_T=len(T), n_Y=len(Y), n_S=len(S), confidence=r.get("confidence"))
            if T:
                row.update(cov_total=len(T & (Y | S)) / len(T), cov_care_only=len(T & S) / len(T),
                           cov_clin_own=len(T & Y) / len(T), cov_incremental=len(T & (Y - S)) / len(T))
            row["y_precision"] = len(Y & T) / len(Y) if Y else np.nan          # clinician highlights that are true omissions
            if care and S:
                row["s_precision"] = len(S & T) / len(S)                        # CARE flags that are true omissions
                row["s_endorsed"] = len(S & Y) / len(S)                         # CARE flags the clinician also highlighted
                row["y_within_s"] = len(Y & S) / len(Y) if Y else np.nan       # clinician highlights inside CARE's flags
            # Hallucination edits: index sentences against the summary this reviewer saw.
            sents = [s for s in split_summary_into_sentences(r["original_llm_summary"]) if is_valid_summary_sentence(s)]
            H, F = set(r.get("true_hallucination_indices", [])), set(r.get("flagged_indices", []))
            if len(sents) == r["n_gen"] and r.get("clinician_summary"):
                cs = norm(r["clinician_summary"])
                removed = {i for i, s in enumerate(sents) if norm(s) not in cs}
                others = set(range(len(sents))) - H
                row.update(n_H=len(H),
                           hall_removed=len(H & removed) / len(H) if H else np.nan,
                           other_removed=len(others & removed) / len(others) if others else np.nan,
                           flag_removed=len(F & removed) / len(F) if (care and F) else np.nan)
            rows.append(row)

arms = {"Control": [r for r in rows if not r["care"]], "CARE": [r for r in rows if r["care"]]}
def line(label, key, fmt="{:.1%}"):
    vals = {a: boot([r.get(key, np.nan) for r in rs]) for a, rs in arms.items()}
    cells = "  ".join(f"{a}: {fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]" if m == m else f"{a}: n/a" for a, (m, lo, hi) in vals.items())
    print(f"  {label:<58} {cells}")

print("Study arms (reviews): " + ", ".join(f"{a} n={len(rs)} (ACI {sum(r['ds']=='ACI_Bench' for r in rs)}, BHC {sum(r['ds']=='MIMIC_IV_BHC' for r in rs)})" for a, rs in arms.items()))
print("Reviews with >=1 oracle true omission: " + ", ".join(f"{a} {sum(r['n_T']>0 for r in rs)}" for a, rs in arms.items()))
print("\nOmission coverage (mean over reviews, 95% bootstrap CI):")
line("Review-time coverage |T∩(Y∪S)|/|T|  (paper endpoint)", "cov_total")
line("Clinician's own highlights |T∩Y|/|T|", "cov_clin_own")
line("CARE flags alone |T∩S|/|T|", "cov_care_only")
line("Incremental beyond CARE |T∩(Y\\S)|/|T|", "cov_incremental")
print("\nHighlight / flag behaviour:")
line("Clinician highlights per review", "n_Y", "{:.1f}")
line("Precision of clinician highlights |Y∩T|/|Y|", "y_precision")
line("CARE flags shown per review", "n_S", "{:.1f}")
line("Precision of CARE flags |S∩T|/|S|", "s_precision")
line("CARE flags the clinician also highlighted |S∩Y|/|S|", "s_endorsed")
line("Clinician highlights inside CARE flags |Y∩S|/|Y|", "y_within_s")
print("\nHallucination handling in edited summaries (sentence absent from clinician's summary):")
line("Oracle-hallucinated sentences removed/rewritten", "hall_removed")
line("Non-hallucinated sentences removed/rewritten", "other_removed")
line("CARE-flagged sentences removed/rewritten", "flag_removed")
print(f"  reviews usable for edit analysis: {sum('other_removed' in r for r in rows)}/{len(rows)}; "
      f"with >=1 oracle hallucination: {sum(r.get('n_H', 0) > 0 for r in rows)}")
print("\nReview time:")
for r in rows: r["t_per_sent"] = r["time"] / r["n_src"] if r["n_src"] else np.nan
line("Seconds per source sentence", "t_per_sent", "{:.2f}")
for r in rows: r["time_min"] = r["time"] / 60
line("Minutes per review", "time_min", "{:.1f}")
print("\nPer clinician (review-time coverage, Control -> CARE | own highlights, Control -> CARE):")
for c in ["C1", "C2", "C3"]:
    g = lambda arm, k: np.nanmean([r.get(k, np.nan) for r in rows if r["clin"] == c and r["care"] == arm])
    print(f"  {c}: {g(False,'cov_total'):.0%} -> {g(True,'cov_total'):.0%}   |   {g(False,'cov_clin_own'):.0%} -> {g(True,'cov_clin_own'):.0%}")
json.dump(rows, open(Path(__file__).with_suffix(".json"), "w"), indent=1, default=float)
