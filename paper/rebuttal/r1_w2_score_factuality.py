"""E10 helper: per-sentence factuality scores from external detectors, computed with the SUBMITTED
scoring functions (git main:paper/scripts/compute_external_scorer_results.py, passed as --ext).

Writes paper/rebuttal/cache/<dataset>__<scorer>_fact.json  {doc_id: [score per generated sentence]}.
Only factuality is scored (E10 is about the hallucination controller). Resumable.

Usage: python r1_w2_score_factuality.py --ext /path/ext_main.py --scorer nli|alignscore --datasets ACI_Bench,...
"""
import argparse, importlib.util, json, os
from pathlib import Path
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--ext", required=True)
ap.add_argument("--scorer", choices=["nli", "alignscore"], required=True)
ap.add_argument("--datasets", default="ACI_Bench,MIMIC_III_CXR,MIMIC_IV_BHC")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("ext", a.ext)
ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
import torch
ext.device = torch.device(os.environ.get("E10_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
print("device", ext.device)

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
if a.scorer == "nli":
    tok, model = ext.load_nli_model()
    score = lambda d: ext.score_nli_factuality(d, tok, model, batch_size=16)
else:
    os.environ.setdefault("ALIGNSCORE_CKPT", str(Path.home() / "models/alignscore/AlignScore-base.ckpt"))
    ext.ALIGNSCORE_CKPT = os.environ["ALIGNSCORE_CKPT"]
    sc = ext.load_alignscore()
    score = lambda d: ext.score_alignscore_factuality(d, sc)

for ds in a.datasets.split(","):
    out = CACHE / f"{ds}__{a.scorer}_fact.json"
    cache = json.load(open(out)) if out.exists() else {}
    docs = ext.load_docs(ds)
    todo = [d for d in docs if str(d["doc_id"]) not in cache]
    for i, d in enumerate(tqdm(todo, desc=f"{ds} {a.scorer}")):
        cache[str(d["doc_id"])] = [float(x) for x in score(d)]
        if (i + 1) % 20 == 0:
            json.dump(cache, open(out, "w"))
    json.dump(cache, open(out, "w"))
    print(ds, "done", len(cache))
