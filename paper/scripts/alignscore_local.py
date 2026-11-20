"""Minimal local AlignScore inference (nli_sp mode) for the released AlignScore-base checkpoint.

Drop-in replacement for `alignscore.AlignScore(...).score(contexts, claims)` used by
compute_external_scorer_results.py when the `alignscore` package is not installed. It follows the
reference implementation (Zha et al., 2023): the context is split into sentence chunks of about
350 words, each claim into sentences; every (chunk, claim-sentence) pair is scored by the
3-way alignment head; the claim score is the mean over claim sentences of the max over chunks
of P(aligned).

Validated against scores produced by the official package (paper/rebuttal/cache/*_alignscore_fact.json)
by validate_alignscore_local.py before use.
"""
from typing import List

import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoConfig, AutoModel, AutoTokenizer


class _AlignModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        h = self.base_model.config.hidden_size
        self.tri_layer = torch.nn.Linear(h, 3)
        self.bin_layer = torch.nn.Linear(h, 2)
        self.reg_layer = torch.nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask):
        pooled = self.base_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.tri_layer(pooled)


class AlignScore:
    def __init__(self, model: str = "roberta-base", batch_size: int = 32, device: str = "cpu",
                 ckpt_path: str = "", evaluation_mode: str = "nli_sp", chunk_words: int = 350):
        assert evaluation_mode == "nli_sp", "only nli_sp is implemented"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.chunk_words = chunk_words
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = _AlignModel(model)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        sd = {k: v for k, v in sd.items() if not k.startswith("mlm_head.")}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not [m for m in missing if "position_ids" not in m], missing
        self.model.to(self.device).eval()
        self.max_len = 512

    @torch.no_grad()
    def _tri_aligned(self, premises: List[str], hypos: List[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(premises), self.batch_size):
            p, h = premises[i:i + self.batch_size], hypos[i:i + self.batch_size]
            try:
                enc = self.tokenizer(p, h, truncation="only_first", padding="max_length",
                                     max_length=self.max_len, return_tensors="pt")
            except Exception:
                # A claim alone exceeds max_length (e.g. a long dialogue turn scored for coverage):
                # truncate the longer side instead of only the context.
                enc = self.tokenizer(p, h, truncation="longest_first", padding="max_length",
                                     max_length=self.max_len, return_tensors="pt")
            enc = enc.to(self.device)
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            out.append(torch.softmax(logits, dim=-1)[:, 0].float().cpu())
        return torch.cat(out) if out else torch.zeros(0)

    def _chunks(self, premise: str) -> List[str]:
        sents = sent_tokenize(premise) or [""]
        n_chunk = len(premise.strip().split()) // self.chunk_words + 1
        per = max(len(sents) // n_chunk, 1)
        return [" ".join(sents[i:i + per]) for i in range(0, len(sents), per)]

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        scores = []
        cache = {}
        for ctx, claim in zip(contexts, claims):
            if ctx not in cache:
                cache[ctx] = self._chunks(ctx)
            chunks = cache[ctx]
            hyps = sent_tokenize(claim) or [""]
            prem = [c for c in chunks for _ in hyps]
            hyp = [h for _ in chunks for h in hyps]
            s = self._tri_aligned(prem, hyp).view(len(chunks), len(hyps))
            scores.append(float(s.max(dim=0).values.mean()))
        return scores
