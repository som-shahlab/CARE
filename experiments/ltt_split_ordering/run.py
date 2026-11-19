#!/usr/bin/env python3
"""Run the split-ordering LTT-FST experiment and its comparators (see README.md)."""
import argparse
import json
from pathlib import Path

import numpy as np

from ltt import crc_largest, fixed_sequence, scalar_ltt

DATASETS = [  # (label, folder under CARE_DATA_ROOT)
    ("ACI-Bench", "ACI_Bench"),
    ("MIMIC-BHC", "MIMIC_IV_BHC"),
    ("MIMIC-CXR", "MIMIC_III_CXR"),
    ("Priv-DS", "PRIV_DS"),
    ("SumPubMed", "SumPubMed"),
]
SEED, N_RESPLITS, CAL_FRAC = 42, 100, 0.70
ALPHA, DELTA, GRID = 0.15, 0.05, 0.05
METHODS = ["LTT split-order", "Code-FST", "LTT τ+γ (last)", "LTT τ+γ (min-WL)",
           "1D-Imp", "Product", "Union", "1D-Imp HB", "Product HB", "Union HB"]


def build(docs, rnd):
    """Per-document fractional loss and workload for every 2D cell and 1D threshold."""
    grid = np.arange(0.0, 1.0 + GRID, GRID)
    if rnd:
        grid = np.round(grid, 4)
    cells = [(float(t), float(g)) for t in grid for g in grid]
    n = len(docs)
    out = {k: np.zeros((n, len(cells))) for k in ("L2", "W2")}
    out.update({k: np.zeros((n, len(grid))) for k in ("Li", "Wi", "Lp", "Wp", "Lc")})
    per_doc = []
    for i, d in enumerate(docs):
        imp = np.asarray(d["importance_probs"], float)
        ncov = 1.0 - np.asarray(d["coverage_probs"], float)
        if rnd:
            imp, ncov = np.round(imp, 6), np.round(ncov, 6)
        true = (np.asarray(d["importance_labels"]) == 1) & (np.asarray(d["coverage_labels"]) == 0)
        n_true = true.sum()
        frac = (lambda s, t=true, nt=n_true: (t & ~s).sum() / nt) if n_true else (lambda s: 0.0)
        per_doc.append((imp, ncov, frac))
        for c, (t, g) in enumerate(cells):
            s = (imp >= t) & (ncov >= g)
            out["L2"][i, c], out["W2"][i, c] = frac(s), s.sum()
        prod = imp * ncov
        for k, b in enumerate(grid):
            s = imp >= b
            out["Li"][i, k], out["Wi"][i, k] = frac(s), s.sum()
            s = prod >= b
            out["Lp"][i, k], out["Wp"][i, k] = frac(s), s.sum()
            out["Lc"][i, k] = frac(ncov >= b)
    return cells, grid, out, per_doc


def split_ordering(cells, L, W, cal, resplit):
    """The pre-registered rule from README.md; returns the deployed cell index."""
    rng = np.random.default_rng(10_000 + resplit)
    perm = rng.permutation(cal)
    n_ord = len(cal) // 3
    ord_idx, cert_idx = perm[:n_ord], perm[n_ord:]
    r_ord, w_ord = L[ord_idx].mean(0), W[ord_idx].mean(0)
    cand = [c for c in range(len(cells)) if r_ord[c] <= ALPHA]
    cand.sort(key=lambda c: (-w_ord[c], round(cells[c][0] + cells[c][1], 9), cells[c][0], cells[c][1]))
    certified = fixed_sequence(L[cert_idx], cand, ALPHA, DELTA)
    if not certified:
        return cells.index((0.0, 0.0)), False
    w_cert = W[cert_idx].mean(0)
    return min(certified, key=lambda c: w_cert[c]), True


def run_dataset(path, rnd):
    docs = [json.loads(l) for l in open(path)]
    cells, grid, M, per_doc = build(docs, rnd)
    L2, W2 = M["L2"], M["W2"]
    n = len(docs)
    n_cal = int(n * CAL_FRAC)
    zero = cells.index((0.0, 0.0))
    asc = sorted(range(len(cells)), key=lambda c: (round(cells[c][0] + cells[c][1], 9), cells[c][0], cells[c][1]))
    desc = sorted(range(len(cells)), key=lambda c: (-round(cells[c][0] + cells[c][1], 9), -cells[c][0], -cells[c][1]))

    def union_eval(ti, gi, test):
        ls, ws = [], []
        for j in test:
            imp, ncov, frac = per_doc[j]
            s = (imp >= grid[ti]) & (ncov >= grid[gi])
            ls.append(frac(s)); ws.append(s.sum())
        return float(np.mean(ls)), float(np.mean(ws))

    rng = np.random.default_rng(SEED)
    res = {m: [] for m in METHODS}
    n_fallback = 0
    for i in range(N_RESPLITS):
        p = rng.permutation(n)
        cal, test = p[:n_cal], p[n_cal:]

        c, ok = split_ordering(cells, L2, W2, cal, i)
        n_fallback += not ok
        picks = {"LTT split-order": c}

        r_cal = L2[cal].mean(0)
        picks["Code-FST"] = next((c for c in desc if n_cal / (n_cal + 1) * r_cal[c] + 1 / (n_cal + 1) <= ALPHA), zero)
        pref = fixed_sequence(L2[cal], asc, ALPHA, DELTA) or [zero]
        w_cal = W2[cal].mean(0)
        picks["LTT τ+γ (last)"] = pref[-1]
        picks["LTT τ+γ (min-WL)"] = min(pref, key=lambda c: w_cal[c])
        for m, c in picks.items():
            res[m].append((float(L2[test, c].mean()), float(W2[test, c].mean())))

        for name, sel in (("", lambda Lm, a: crc_largest(Lm, a)), (" HB", lambda Lm, a, d=DELTA: scalar_ltt(Lm, a, d))):
            k = sel(M["Li"][cal], ALPHA); res["1D-Imp" + name].append((float(M["Li"][test, k].mean()), float(M["Wi"][test, k].mean())))
            k = sel(M["Lp"][cal], ALPHA); res["Product" + name].append((float(M["Lp"][test, k].mean()), float(M["Wp"][test, k].mean())))
        res["Union"].append(union_eval(crc_largest(M["Li"][cal], ALPHA / 2), crc_largest(M["Lc"][cal], ALPHA / 2), test))
        res["Union HB"].append(union_eval(scalar_ltt(M["Li"][cal], ALPHA / 2, DELTA / 2),
                                          scalar_ltt(M["Lc"][cal], ALPHA / 2, DELTA / 2), test))

    summary = {}
    for m, v in res.items():
        a = np.array(v)
        summary[m] = {"viol_mean": float(a[:, 0].mean()), "workload_mean": float(a[:, 1].mean()),
                      "frac_resplits_viol_gt_alpha": float((a[:, 0] > ALPHA).mean()),
                      "per_resplit": v}
    summary["_meta"] = {"n_docs": n, "n_cal": n_cal, "split_order_fallbacks": n_fallback}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[0, 1], default=0, help="1 = float-grid fix")
    args = ap.parse_args()
    from care.config import CANONICAL_DATA_ROOT as root  # honours CARE_DATA_ROOT
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    results = {}
    for label, folder in DATASETS:
        print(f"running {label} ...", flush=True)
        results[label] = run_dataset(root / folder / "phase_2" / "calibrated_scores.jsonl", args.round)

    tag = f"round{args.round}"
    json.dump({"config": dict(seed=SEED, n_resplits=N_RESPLITS, alpha=ALPHA, delta=DELTA, grid=GRID, round=args.round),
               "results": results}, open(out_dir / f"results_{tag}.json", "w"), indent=1)

    hb_methods = ["LTT split-order", "LTT τ+γ (last)", "LTT τ+γ (min-WL)", "1D-Imp HB", "Product HB", "Union HB"]
    lines = [f"# Results ({tag}; α={ALPHA}, δ={DELTA}, {N_RESPLITS} resplits)", "",
             "Cells: mean test violation % / mean surfaced source sentences per doc / % of resplits with test violation > α", "",
             "| Dataset | " + " | ".join(METHODS) + " | lowest among HB-guaranteed |",
             "|---|" + "---|" * (len(METHODS) + 1)]
    for label, s in results.items():
        cells = [f"{100*s[m]['viol_mean']:.1f} / {s[m]['workload_mean']:.1f} / {100*s[m]['frac_resplits_viol_gt_alpha']:.0f}%" for m in METHODS]
        best = min(hb_methods, key=lambda m: s[m]["workload_mean"])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {best} |")
    lines.append("")
    lines.append("Split-order fallbacks to (0,0): " + ", ".join(f"{k} {v['_meta']['split_order_fallbacks']}" for k, v in results.items()))
    (out_dir / f"results_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
