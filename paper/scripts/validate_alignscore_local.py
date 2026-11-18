#!/usr/bin/env python3
"""Check alignscore_local against scores produced by the official AlignScore package.

Compares factuality scores (context = full source, claim = summary sentence) for the first
N documents of a dataset with paper/rebuttal/cache/<DS>__alignscore_fact.json.
Usage: python validate_alignscore_local.py [--dataset MIMIC_III_CXR] [--n 10] [--device cpu]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignscore_local import AlignScore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="MIMIC_III_CXR")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
root = Path(os.environ["CARE_DATA_ROOT"])
ref = json.load(open(Path(__file__).resolve().parents[1] / "rebuttal" / "cache" / f"{a.dataset}__alignscore_fact.json"))
docs = [json.loads(l) for l in open(root / a.dataset / "phase_2" / "calibrated_scores.jsonl")]
docs = [d for d in docs if str(d["doc_id"]) in ref][: a.n]
sc = AlignScore(device=a.device, ckpt_path=os.environ.get("ALIGNSCORE_CKPT", str(Path.home() / "models/alignscore/AlignScore-base.ckpt")))
diffs = []
for d in docs:
    src = " ".join(d["source_sentences"])
    mine = sc.score([src] * len(d["generated_sentences"]), d["generated_sentences"])
    diffs += list(np.abs(np.array(mine) - np.array(ref[str(d["doc_id"])])))
diffs = np.array(diffs)
print(f"{a.dataset}: {len(docs)} docs, {len(diffs)} sentences | max |diff| = {diffs.max():.2e}, "
      f"mean = {diffs.mean():.2e}, share > 1e-3: {(diffs > 1e-3).mean():.3f}")
