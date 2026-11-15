"""Date every Pile copy of the found pilot questions to the Pythia training step that read it.

    python -m exposure.date_items [--max-copies 50]

For each pilot item whose question was found in the Pile (tiers confirmed and
question_only), and for every copy infini-gram finds (up to --max-copies):

  infini-gram copy -> Pythia's document (window search, exposure.pile_order)
                   -> the 2,049-token training samples that hold that document
                   -> training position and step
                   -> read that training sample and confirm the question's tokens are in it

Writes data/exposure/pilot_dates.parquet (one row per copy and training sample, with
`verified`) and data/exposure/pilot_first_exposure.parquet (one row per item: t0, the
number of verified copies, and copies seen by each pilot checkpoint). Only verified
samples count. This is Phase 0 step 3 of AGENTS.md: it tests the step-mapping method
end to end on real items.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config, provenance, write_parquet
from evals.benchmarks import load_items
from exposure.pile_order import (BATCH, DOC_WINDOW, TRAINED_SAMPLES, TokenizedPile, TrainingOrder,
                                 find_subsequence, read_training_sample)
from exposure.retrieval import InfiniGramClient
from exposure.search_items import clean_text
from exposure.step_mapping import copy_count_at, first_exposure_step

PILOT_STEPS = [512, 1000, 3000, 13000, 33000, 53000, 73000, 93000, 113000, 143000]


def needle_of(tok, text: str) -> np.ndarray:
    """GPT-NeoX tokens of the text as it would appear mid-document, minus its first and
    last token, which change with the surrounding characters."""
    ids = tok(" " + text).input_ids
    return np.asarray(ids[1:-1] if len(ids) > 4 else ids, dtype=np.uint16)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exposure.yaml")
    ap.add_argument("--max-copies", type=int, default=50)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", cache_dir=REPO_ROOT / "data" / "hf_cache_tok")
    client = InfiniGramClient.from_config(load_config(args.config))
    pile, order = TokenizedPile(), TrainingOrder()
    tiers = pd.read_parquet(REPO_ROOT / "data" / "exposure" / "pilot_tiers.parquet")
    found = tiers[tiers["tier"].isin(["confirmed", "question_only"])]
    items = {(it["benchmark"], it["item_id"]): it for it in load_items()}

    rows = []
    for r in found.itertuples():
        q = clean_text(items[(r.benchmark, r.item_id)]["question"])
        needle = needle_of(tok, q)
        hit = client.request("find", query=q)
        ranks = [(s, k) for s, (lo, hi) in enumerate(hit.get("segment_by_shard", [])) for k in range(lo, hi)][: args.max_copies]
        for s, rank in ranks:
            d = client.request("get_doc_by_rank", query=q, s=s, rank=rank, max_disp_len=200)
            meta = json.loads(d.get("metadata") or "{}")
            base = {"benchmark": r.benchmark, "item_id": r.item_id, "tier": r.tier, "ig_doc": int(d["doc_ix"]),
                    "source": f'{meta.get("path")}:{meta.get("linenum")}',
                    "pile_set": meta.get("metadata", {}).get("meta", {}).get("pile_set_name")}
            toks, starts = pile.window(base["ig_doc"] - DOC_WINDOW, base["ig_doc"] + DOC_WINDOW)
            docs = sorted({base["ig_doc"] - DOC_WINDOW + int(np.searchsorted(starts, h, side="right")) - 1
                           for h in find_subsequence(toks, needle)})
            if not docs:
                rows.append({**base, "neox_doc": -1, "sample": -1, "position": -1, "step": -1, "verified": False})
                continue
            for doc in docs:
                for sample in order.samples_of_doc(doc):
                    pos = order.training_position(sample)
                    trained = pos < TRAINED_SAMPLES
                    ok = trained and bool(find_subsequence(read_training_sample(pos), needle))
                    rows.append({**base, "neox_doc": doc, "sample": sample, "position": pos,
                                 "step": pos // BATCH + 1 if trained else -1, "verified": ok})
        print(f"{r.benchmark} {r.item_id}: {len(ranks)} copies, "
              f"{sum(x['verified'] for x in rows if x['item_id'] == r.item_id)} verified training samples", flush=True)

    dates = pd.DataFrame(rows)
    prov = provenance({"max_copies": args.max_copies, "doc_window": DOC_WINDOW}, args.config)
    write_parquet(dates, "data/exposure/pilot_dates.parquet", prov)

    per_item = []
    for (b, i), g in found.groupby(["benchmark", "item_id"]):
        v = dates[(dates["benchmark"] == b) & (dates["item_id"] == i) & dates["verified"]] if len(dates) else dates
        batches = (v["step"] - 1).to_numpy() if len(v) else np.array([], dtype=int)
        per_item.append({"benchmark": b, "item_id": i, "tier": g["tier"].iloc[0],
                         "copies_found": int(g["q_count"].iloc[0]), "copies_verified": int(len(v)),
                         "t0": first_exposure_step(batches),
                         **{f"n_step{s}": int(c) for s, c in zip(PILOT_STEPS, copy_count_at(batches, PILOT_STEPS))}})
    first = pd.DataFrame(per_item)
    write_parquet(first, "data/exposure/pilot_first_exposure.parquet", prov)
    print(first.to_string(index=False))


if __name__ == "__main__":
    main()
