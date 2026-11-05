"""Score every benchmark item at Pythia checkpoints on CPU (pilot).

    python -m evals.score_cpu --model EleutherAI/pythia-410m --steps 512 2000 ... [--limit N]

For each item and each answer choice, computes the summed log-probability of the choice
given the prompt, exactly as lm-evaluation-harness does for multiple-choice tasks. Then:
  y    = 1 if the highest-scoring choice is the correct one (lm-eval "acc")
  logp = log-probability of the correct choice
Writes data/evals/pilot/<model>/step<N>.parquet in the evaluation-table schema, one
checkpoint at a time, and deletes each downloaded checkpoint afterwards to save disk.
Reruns skip checkpoints that are already scored.

This is the CPU pilot (Pythia 410M, a subset of checkpoints). The full study uses
`evals.run_checkpoint` with lm-evaluation-harness on a GPU.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, provenance, write_parquet
from evals.benchmarks import load_items
from evals.schema import EVAL_COLUMNS


def score_items(model, tok, items, batch_tokens: int = 6000) -> list[dict]:
    import torch

    # every (item, choice) is one sequence: prompt + continuation
    seqs = []
    for i, it in enumerate(items):
        ctx = tok(it["prompt"]).input_ids
        for j, ch in enumerate(it["choices"]):
            cont = tok(ch).input_ids
            ids = (ctx + cont)[-2048:]
            seqs.append((i, j, ids, len(cont)))
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]))
    ll = np.zeros(len(seqs))
    k = 0
    with torch.inference_mode():
        while k < len(order):
            maxlen = len(seqs[order[k]][2])
            bs = 1
            while k + bs < len(order) and (bs + 1) * len(seqs[order[k + bs]][2]) <= batch_tokens:
                bs += 1
            idx = order[k:k + bs]
            maxlen = max(len(seqs[t][2]) for t in idx)
            batch = torch.full((len(idx), maxlen), tok.pad_token_id or 0, dtype=torch.long)
            for r, t in enumerate(idx):
                ids = seqs[t][2]
                batch[r, :len(ids)] = torch.tensor(ids)
            logits = model(batch).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for r, t in enumerate(idx):
                ids, ncont = seqs[t][2], seqs[t][3]
                n = len(ids)
                pos = torch.arange(n - ncont - 1, n - 1)
                tgt = torch.tensor(ids[n - ncont:])
                ll[t] = logprobs[r, pos, tgt].sum().item()
            k += bs
    by_item: dict[int, list[float]] = {}
    for t, (i, j, _, _) in enumerate(seqs):
        by_item.setdefault(i, []).append(ll[t])
    rows = []
    for i, it in enumerate(items):
        scores = by_item[i]
        rows.append({"benchmark": it["benchmark"], "item_id": it["item_id"],
                     "y": int(int(np.argmax(scores)) == it["answer"]), "logp": float(scores[it["answer"]])})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N items per benchmark (smoke test)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    items = load_items()
    if args.limit:
        counts: dict[str, int] = {}
        kept = []
        for it in items:
            counts[it["benchmark"]] = counts.get(it["benchmark"], 0) + 1
            if counts[it["benchmark"]] <= args.limit:
                kept.append(it)
        items = kept
    name = args.model.split("/")[-1]
    out_dir = REPO_ROOT / "data" / "evals" / "pilot" / name
    cache = REPO_ROOT / "data" / "hf_cache"
    tok = AutoTokenizer.from_pretrained(args.model, revision=f"step{args.steps[-1]}", cache_dir=cache)
    prov = provenance({"model": args.model, "steps": args.steps, "limit": args.limit, "n_items": len(items)})
    for step in args.steps:
        out = out_dir / f"step{step}.parquet"
        if out.exists() and not args.limit:
            print(f"step{step}: already scored")
            continue
        t0 = time.time()
        model = GPTNeoXForCausalLM.from_pretrained(args.model, revision=f"step{step}", cache_dir=cache, torch_dtype=torch.float32)
        model.eval()
        rows = score_items(model, tok, items)
        df = pd.DataFrame(rows)
        df["model"], df["data_version"], df["step"] = name, ("deduped" if "deduped" in name else "standard"), step
        df["tier"], df["t0"], df["n"] = None, None, None
        write_parquet(df[EVAL_COLUMNS], out, prov)
        acc = df.groupby("benchmark")["y"].mean().round(3).to_dict()
        print(f"step{step}: {len(df)} items in {time.time() - t0:.0f}s, accuracy {acc}", flush=True)
        del model
        if not args.keep_cache:
            for d in cache.glob(f"models--{args.model.replace('/', '--')}/blobs"):
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
