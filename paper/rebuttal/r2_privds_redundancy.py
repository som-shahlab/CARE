"""R2: copy-paste redundancy in Priv-DS sources (E5). Source-only statistics; workload is not recomputed.

Pre-specified before looking at results:
  - Unit: phase-2 source sentences (the units CARE scores and surfaces).
  - Normalization: lowercase, non-alphanumerics -> space, collapse whitespace.
  - Eligible: >= 5 normalized tokens (shorter fragments such as headers or "plan" repeat trivially).
  - Duplicate: another eligible sentence in the SAME admission with token-set Jaccard >= 0.8
    (primary); sensitivity at 0.6 / 0.7 / 0.9 / 1.0 (1.0 = exact after normalization).
  - Repeat copy: a duplicate whose match occurs EARLIER in the admission (i.e. collapsible).
  - Surfaced: Table 9 Priv-DS thresholds, p_imp >= 0.50 and 1 - p_cov >= 0.50.
  - Oracle omission: omission_labels == 1 (important AND not covered).
"""
import json
import re
from pathlib import Path

import numpy as np

from care.config import get_phase2_scores

DATA = get_phase2_scores("PRIV_DS")
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from care_thresholds import thresholds, mode
_, TAU, GAMMA = thresholds()["Priv-DS"]  # camera-ready by default; CARE_THRESHOLDS=submitted gives 0.50, 0.50
MIN_TOK = 5
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]
PRIMARY = 0.8


def norm_tokens(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def doc_matches(sents, thr, surf=None):
    """For each sentence: (has_any_match, has_earlier_match, has_earlier_SURFACED_match) at Jaccard >= thr."""
    toks = [frozenset(norm_tokens(s)) if len(norm_tokens(s)) >= MIN_TOK else None for s in sents]
    n = len(sents)
    any_m, earlier_m, earlier_surf = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    # inverted index on tokens for candidate generation
    index = {}
    for i, t in enumerate(toks):
        if t is None:
            continue
        cands = set()
        for w in t:
            cands.update(index.get(w, ()))
        for j in cands:  # j < i
            u = toks[j]
            inter = len(t & u)
            if inter / (len(t) + len(u) - inter) >= thr:
                any_m[i] = any_m[j] = True
                earlier_m[i] = True
                if surf is not None and surf[j]:
                    earlier_surf[i] = True
        for w in t:
            index.setdefault(w, []).append(i)
    eligible = np.array([t is not None for t in toks])
    return any_m, earlier_m, earlier_surf, eligible


def main():
    rows = [json.loads(l) for l in open(DATA)]
    out = {"n_docs": len(rows), "params": dict(tau=TAU, gamma=GAMMA, min_tok=MIN_TOK, primary=PRIMARY)}
    # per-document cached label arrays
    docs = []
    for r in rows:
        surf = (np.array(r["importance_probs"]) >= TAU) & (1 - np.array(r["coverage_probs"]) >= GAMMA)
        omit = np.array(r["omission_labels"]) == 1
        imp = np.array(r["importance_labels"]) == 1
        docs.append((r, surf, omit, imp))
    out["mean_source_sentences"] = float(np.mean([len(r["source_sentences"]) for r, *_ in docs]))
    out["mean_surfaced"] = float(np.mean([s.sum() for _, s, _, _ in docs]))
    out["mean_oracle_omissions"] = float(np.mean([o.sum() for _, _, o, _ in docs]))

    res = {}
    for thr in THRESHOLDS:
        tot = dict(n=0, elig=0, dup=0, rep=0, surf=0, surf_dup=0, surf_rep=0, omit=0, omit_dup=0, omit_rep=0,
                   rep_surf_and_earlier_surf=0)
        per_doc_dup, per_doc_rep = [], []
        for r, surf, omit, imp in docs:
            any_m, earlier_m, earlier_surf, elig = doc_matches(r["source_sentences"], thr, surf)
            tot["rep_surf_and_earlier_surf"] += int((surf & earlier_surf).sum())
            n = len(any_m)
            tot["n"] += n; tot["elig"] += int(elig.sum())
            tot["dup"] += int(any_m.sum()); tot["rep"] += int(earlier_m.sum())
            tot["surf"] += int(surf.sum()); tot["surf_dup"] += int((surf & any_m).sum()); tot["surf_rep"] += int((surf & earlier_m).sum())
            tot["omit"] += int(omit.sum()); tot["omit_dup"] += int((omit & any_m).sum()); tot["omit_rep"] += int((omit & earlier_m).sum())
            per_doc_dup.append(any_m.mean()); per_doc_rep.append(earlier_m.mean())
        res[str(thr)] = dict(
            totals=tot,
            pooled_dup_rate=tot["dup"] / tot["n"], pooled_repeat_rate=tot["rep"] / tot["n"],
            macro_dup_rate=float(np.mean(per_doc_dup)), macro_repeat_rate=float(np.mean(per_doc_rep)),
            docs_with_any_repeat=int(sum(x > 0 for x in per_doc_rep)),
            share_surfaced_dup=tot["surf_dup"] / tot["surf"], share_surfaced_repeat=tot["surf_rep"] / tot["surf"],
            share_omissions_dup=tot["omit_dup"] / tot["omit"], share_omissions_repeat=tot["omit_rep"] / tot["omit"],
            share_surfaced_collapsible=tot["rep_surf_and_earlier_surf"] / tot["surf"],
        )

    # Consistency within duplicate pairs at the primary threshold: do copies get the same oracle label and surfacing?
    pairs = dict(n=0, same_imp=0, same_omit=0, same_surf=0)
    for r, surf, omit, imp in docs:
        sents = r["source_sentences"]
        toks = [frozenset(norm_tokens(s)) if len(norm_tokens(s)) >= MIN_TOK else None for s in sents]
        index = {}
        for i, t in enumerate(toks):
            if t is None:
                continue
            cands = set()
            for w in t:
                cands.update(index.get(w, ()))
            for j in cands:
                u = toks[j]; inter = len(t & u)
                if inter / (len(t) + len(u) - inter) >= PRIMARY:
                    pairs["n"] += 1
                    pairs["same_imp"] += int(imp[i] == imp[j])
                    pairs["same_omit"] += int(omit[i] == omit[j])
                    pairs["same_surf"] += int(surf[i] == surf[j])
            for w in t:
                index.setdefault(w, []).append(i)
    out["pair_consistency_primary"] = {k: (v if k == "n" else v / pairs["n"]) for k, v in pairs.items()}

    # Descriptive (not a new risk estimate): omission miss rate per admission at the fixed Table 9 thresholds,
    # counting sentences vs counting distinct content (duplicate clusters = connected components at J >= PRIMARY).
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    sent_l, clus_l = [], []
    for r, surf, omit, imp in docs:
        if not omit.any():
            continue
        sents = r["source_sentences"]; n = len(sents)
        toks = [frozenset(norm_tokens(s)) if len(norm_tokens(s)) >= MIN_TOK else None for s in sents]
        ii, jj = [], []
        index = {}
        for i, t in enumerate(toks):
            if t is None:
                continue
            cands = set()
            for w in t:
                cands.update(index.get(w, ()))
            for j in cands:
                u = toks[j]; inter = len(t & u)
                if inter / (len(t) + len(u) - inter) >= PRIMARY:
                    ii.append(i); jj.append(j)
            for w in t:
                index.setdefault(w, []).append(i)
        _, lab = connected_components(csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n)), directed=False)
        sent_l.append((omit & ~surf).sum() / omit.sum())
        oc = set(lab[omit])
        clus_l.append(sum(not surf[lab == c].any() for c in oc) / len(oc))
    out["omission_miss_sentence_vs_content"] = dict(sentence_level=float(np.mean(sent_l)), content_level=float(np.mean(clus_l)),
                                                    n_docs=len(sent_l))
    out["by_threshold"] = res
    json.dump(out, open(Path(__file__).with_suffix(".json") if mode() != "submitted" else Path(__file__).with_name(Path(__file__).stem + "_submitted.json"), "w"), indent=1)

    print(f"Priv-DS: {out['n_docs']} admissions, {out['mean_source_sentences']:.1f} source sents/adm, "
          f"{out['mean_surfaced']:.1f} surfaced (tau=gamma=0.5), {out['mean_oracle_omissions']:.1f} oracle omissions")
    print(f"{'Jacc':>5} | {'dup%':>6} {'repeat%':>8} {'docs w/ rep':>11} | {'surf dup%':>9} {'surf rep%':>9} {'collaps%':>8} | {'omit dup%':>9} {'omit rep%':>9}")
    for thr, x in res.items():
        print(f"{thr:>5} | {100*x['pooled_dup_rate']:6.1f} {100*x['pooled_repeat_rate']:8.1f} {x['docs_with_any_repeat']:>11} | "
              f"{100*x['share_surfaced_dup']:9.1f} {100*x['share_surfaced_repeat']:9.1f} {100*x['share_surfaced_collapsible']:8.1f} | "
              f"{100*x['share_omissions_dup']:9.1f} {100*x['share_omissions_repeat']:9.1f}")
    pc = out["pair_consistency_primary"]
    print(f"Duplicate pairs at J>={PRIMARY}: n={pc['n']}, same oracle importance {100*pc['same_imp']:.1f}%, "
          f"same omission label {100*pc['same_omit']:.1f}%, same surfacing {100*pc['same_surf']:.1f}%")
    oc = out["omission_miss_sentence_vs_content"]
    print(f"Omission miss rate at Table 9 thresholds (all {oc['n_docs']} admissions, descriptive): "
          f"sentence-level {oc['sentence_level']:.3f}, distinct-content level {oc['content_level']:.3f}")


if __name__ == "__main__":
    main()
