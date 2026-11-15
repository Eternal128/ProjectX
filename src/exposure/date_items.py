"""Date every Pile copy of the found pilot questions to the Pythia training step that read it.

    python -m exposure.date_items [--source pilot|lambada] [--max-copies 50]

For each pilot item whose question was found in the Pile (tiers confirmed and
question_only), and for every copy infini-gram finds (up to --max-copies):

  infini-gram copy -> Pythia's document (window search, exposure.pile_order)
                   -> the 2,049-token training samples that hold that document
                   -> training position and step
                   -> read that training sample and confirm the question's tokens are in it

Writes data/exposure/<source>_dates.parquet (one row per copy and training sample, with
`verified`) and data/exposure/<source>_first_exposure.parquet (one row per item: t0, the
number of verified copies, whether it passes the timing filter, and copies seen by each
pilot checkpoint). `--source lambada` dates the LAMBADA passages the coverage probe found. Only verified
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
                                 find_neox_docs, find_subsequence, read_training_sample)
from exposure.retrieval import InfiniGramClient
from exposure.search_items import clean_text
from exposure.step_mapping import copy_count_at, first_exposure_step

PILOT_STEPS = [512, 1000, 3000, 13000, 33000, 53000, 73000, 93000, 113000, 143000]


def needle_of(tok, text: str) -> np.ndarray:
    """GPT-NeoX tokens of the text as it would appear mid-document, minus its first and
    last token, which change with the surrounding characters."""
    ids = tok(" " + text).input_ids
    return np.asarray(ids[1:-1] if len(ids) > 4 else ids, dtype=np.uint16)


def targets(source: str) -> list[dict]:
    """Items to date: {benchmark, item_id, tier, query, copies_found}."""
    exp = REPO_ROOT / "data" / "exposure"
    if source == "pilot":
        tiers = pd.read_parquet(exp / "pilot_tiers.parquet")
        found = tiers[tiers["tier"].isin(["confirmed", "question_only"])]
        items = {(it["benchmark"], it["item_id"]): it for it in load_items()}
        return [{"benchmark": r.benchmark, "item_id": r.item_id, "tier": r.tier, "copies_found": int(r.q_count),
                 "query": clean_text(items[(r.benchmark, r.item_id)]["question"])} for r in found.itertuples()]
    if source == "lambada":
        from exposure.probe_coverage import download, LAMBADA_URL, windows

        probe = pd.read_parquet(exp / "probe_coverage.parquet")
        hits = probe[(probe["benchmark"] == "lambada") & (probe["count"] > 0)]
        lam = [json.loads(x) for x in download(LAMBADA_URL, "lambada_test_en.jsonl").read_text().splitlines() if x.strip()]
        return [{"benchmark": "lambada", "item_id": r.item_id, "tier": "window", "copies_found": int(r.count),
                 "query": windows(clean_text(lam[int(r.item_id)]["text"]))["last"]} for r in hits.itertuples()]
    raise ValueError(source)


def passes_timing(t0, checkpoints, total_steps: int, cfg: dict) -> bool:
    """AGENTS.md Phase 1 step 5: at least N checkpoints before t0, and t0 within the middle of training."""
    if t0 is None or (isinstance(t0, float) and np.isnan(t0)):
        return False
    tf = cfg["timing_filter"]
    before = sum(1 for c in checkpoints if c < t0)
    return before >= tf["min_checkpoints_before_t0"] and tf["t0_min_fraction"] <= t0 / total_steps <= tf["t0_max_fraction"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exposure.yaml")
    ap.add_argument("--source", choices=["pilot", "lambada"], default="pilot",
                    help="pilot: found SciQ / ARC-Easy / PIQA questions; lambada: LAMBADA passages found by the coverage probe")
    ap.add_argument("--max-copies", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    cfg = load_config(args.config)
    models = load_config("configs/models.yaml")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", cache_dir=REPO_ROOT / "data" / "hf_cache_tok")
    client = InfiniGramClient.from_config(cfg)
    pile, order = TokenizedPile(), TrainingOrder()
    todo = targets(args.source)

    _ = order._doc_positions  # build the reverse lookup once, before the worker threads

    def date_one(t: dict) -> list[dict]:
        out = []
        q, needle = t["query"], needle_of(tok, t["query"])
        hit = client.request("find", query=q)
        ranks = [(s, k) for s, (lo, hi) in enumerate(hit.get("segment_by_shard", [])) for k in range(lo, hi)][: args.max_copies]
        for s, rank in ranks:
            d = client.request("get_doc_by_rank", query=q, s=s, rank=rank, max_disp_len=200)
            meta = json.loads(d.get("metadata") or "{}")
            base = {"benchmark": t["benchmark"], "item_id": t["item_id"], "tier": t["tier"], "ig_doc": int(d["doc_ix"]),
                    "source": f'{meta.get("path")}:{meta.get("linenum")}',
                    "pile_set": meta.get("metadata", {}).get("meta", {}).get("pile_set_name")}
            docs = find_neox_docs(pile, base["ig_doc"], needle)
            if not docs:
                out.append({**base, "neox_doc": -1, "sample": -1, "position": -1, "step": -1, "verified": False})
                continue
            for doc in docs:
                for sample in order.samples_of_doc(doc):
                    pos = order.training_position(sample)
                    trained = pos < TRAINED_SAMPLES
                    ok = trained and bool(find_subsequence(read_training_sample(pos), needle))
                    out.append({**base, "neox_doc": doc, "sample": sample, "position": pos,
                                "step": pos // BATCH + 1 if trained else -1, "verified": ok})
        n_ok = len({x["position"] for x in out if x["verified"]})
        print(f"{t['benchmark']} {t['item_id']}: {len(ranks)} copies, {n_ok} verified training samples", flush=True)
        return out

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as pool:  # network-bound: overlap the waits
        rows = [r for out in pool.map(date_one, todo) for r in out]

    # two infini-gram copies in one document lead to the same training samples: count each sample once
    dates = pd.DataFrame(rows).drop_duplicates(["benchmark", "item_id", "neox_doc", "sample"]).reset_index(drop=True)
    prov = provenance({"source": args.source, "max_copies": args.max_copies, "doc_window": DOC_WINDOW}, args.config)
    write_parquet(dates, f"data/exposure/{args.source}_dates.parquet", prov)

    per_item = []
    for t in todo:
        v = dates[(dates["benchmark"] == t["benchmark"]) & (dates["item_id"] == t["item_id"]) & dates["verified"]]
        batches = (v["step"] - 1).to_numpy()
        t0 = first_exposure_step(batches)
        per_item.append({"benchmark": t["benchmark"], "item_id": t["item_id"], "tier": t["tier"],
                         "copies_found": t["copies_found"], "copies_verified": int(len(v)), "t0": t0,
                         "passes_timing": passes_timing(t0, models["checkpoints"], models["training"]["total_steps"], cfg),
                         **{f"n_step{s}": int(c) for s, c in zip(PILOT_STEPS, copy_count_at(batches, PILOT_STEPS))}})
    first = pd.DataFrame(per_item)
    write_parquet(first, f"data/exposure/{args.source}_first_exposure.parquet", prov)
    print(first.drop(columns=[c for c in first.columns if c.startswith("n_step")]).to_string(index=False))
    dated = first["t0"].notna()
    print(f"dated {int(dated.sum())} of {len(first)} items; {int(first['passes_timing'].sum())} pass the timing filter")


if __name__ == "__main__":
    main()
