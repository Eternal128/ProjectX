"""When did each training run read each found item? (the seed check)

    python -m exposure.seed_dates [--sources pilot lambada] [--seeds 1 2 ... 9]

EleutherAI retrained Pythia with 9 more seeds (PolyPythias); each run reads the same
documents in a different order. `EleutherAI/pile-preshuffled-seeds` publishes each
run's index maps (seed0 is the original Pythia run, seed 1234). Document numbers are the
same in every run, so a copy's place in the data is fixed: (document, token offset).
Only the order changes.

For every copy confirmed by `exposure.date_items` we locate the item's token offset in
its document once (a few KB of range reads). Then, per run, the sample that holds that
offset is found by index arithmetic alone, and its training position gives the step.
Check: for seed0 this must reproduce the positions that date_items confirmed by reading
Pythia's actual training batches.

Each seed's maps (2.8 GB) are downloaded, used and deleted in turn. Writes
data/exposure/seed_dates.parquet (item, copy, seed, position, step) and
data/exposure/seed_first_exposure.parquet (item, seed, t0, copies read in training).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, provenance, write_parquet
from exposure.date_items import needle_of, targets
from exposure.pile_order import BATCH, MAPS, TRAINED_SAMPLES, TokenizedPile, TrainingOrder, find_subsequence
from exposure.step_mapping import first_exposure_step

SEED_URL = "https://huggingface.co/datasets/EleutherAI/pile-preshuffled-seeds/resolve/main"
STEP_DIR = REPO_ROOT / "data" / "step_mapping"


def maps_name(seed: int) -> str:
    # seed0 holds the original run (seed 1234); seedK holds seed K
    return MAPS if seed == 0 else MAPS.replace("_1234s", f"_{seed}s")


def fetch_maps(seed: int):
    d = STEP_DIR / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    for kind in ("doc_idx", "sample_idx", "shuffle_idx"):
        f = d / f"{maps_name(seed)}_{kind}.npy"
        if not f.exists() or f.stat().st_size == 0:
            subprocess.run(["curl", "-sS", "-L", "--retry", "5", "-C", "-", "-o", str(f),
                            f"{SEED_URL}/seed{seed}/{f.name}"], check=True)
    return d


def sample_holding(order: TrainingOrder, doc: int, offset: int) -> int | None:
    """The unshuffled sample whose span starts at or before (doc position, offset), last such."""
    best = None
    for j in order.doc_position(doc).tolist():
        for s in order.samples_of_doc(doc):
            jf, of = (int(v) for v in order.sample_idx[s])
            jl, ol = (int(v) for v in order.sample_idx[s + 1])
            if (jf, of) <= (j, offset) <= (jl, ol):
                best = s if best is None else min(best, s)
    return best


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=["pilot", "lambada"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--keep-maps", action="store_true")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", cache_dir=REPO_ROOT / "data" / "hf_cache_tok")
    pile = TokenizedPile()

    # 1. every confirmed copy: (item, document, token offset of the item inside it)
    copies = []
    for source in args.sources:
        dates = pd.read_parquet(REPO_ROOT / "data" / "exposure" / f"{source}_dates.parquet")
        ok = dates[dates["verified"]].drop_duplicates(["benchmark", "item_id", "neox_doc", "position"])
        query = {(t["benchmark"], t["item_id"]): t["query"] for t in targets(source)}
        for r in ok.itertuples():
            toks, _ = pile.window(int(r.neox_doc), int(r.neox_doc) + 1)
            for off in find_subsequence(toks, needle_of(tok, query[(r.benchmark, r.item_id)])):
                copies.append({"source": source, "benchmark": r.benchmark, "item_id": r.item_id,
                               "neox_doc": int(r.neox_doc), "offset": int(off), "seed0_position": int(r.position)})
    copies = pd.DataFrame(copies).drop_duplicates(["benchmark", "item_id", "neox_doc", "offset"])
    print(f"{len(copies)} copies of {copies[['benchmark', 'item_id']].drop_duplicates().shape[0]} items", flush=True)

    rows = []
    part_dir = REPO_ROOT / "data" / "exposure" / "seed_dates_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    for seed in [0, *args.seeds]:
        part = part_dir / f"seed{seed}.parquet"
        if part.exists() and seed != 0:  # finished earlier: reuse, so a stopped run resumes
            rows += pd.read_parquet(part).to_dict("records")
            print(f"seed {seed}: already done", flush=True)
            continue
        order = TrainingOrder(fetch_maps(seed) if seed else STEP_DIR / "seed0", name=maps_name(seed))
        for c in copies.itertuples():
            s = sample_holding(order, c.neox_doc, c.offset)
            pos = order.training_position(s) if s is not None else -1
            rows.append({"benchmark": c.benchmark, "item_id": c.item_id, "neox_doc": c.neox_doc, "offset": c.offset,
                         "seed": seed, "position": pos,
                         "step": pos // BATCH + 1 if 0 <= pos < TRAINED_SAMPLES else -1})
        pd.DataFrame([r for r in rows if r["seed"] == seed]).to_parquet(part, index=False)
        if seed == 0:
            # the check: index arithmetic must agree with the batches date_items read
            mine = pd.DataFrame([r for r in rows if r["seed"] == 0]).merge(copies, on=["benchmark", "item_id", "neox_doc", "offset"])
            agree = int((mine["position"] == mine["seed0_position"]).sum())
            print(f"seed0 check: {agree} of {len(mine)} copies land at the confirmed training position", flush=True)
        else:
            print(f"seed {seed}: done", flush=True)
            if not args.keep_maps:
                shutil.rmtree(STEP_DIR / f"seed{seed}", ignore_errors=True)
        del order
        import gc

        gc.collect()

    df = pd.DataFrame(rows)
    prov = provenance({"sources": args.sources, "seeds": [0, *args.seeds]})
    write_parquet(df, "data/exposure/seed_dates.parquet", prov)
    per = []
    for (b, i, seed), g in df.groupby(["benchmark", "item_id", "seed"]):
        trained = g[g["step"] > 0]
        per.append({"benchmark": b, "item_id": i, "seed": seed, "copies": int(len(g)), "copies_trained": int(len(trained)),
                    "t0": first_exposure_step((trained["step"] - 1).to_numpy())})
    first = pd.DataFrame(per)
    write_parquet(first, "data/exposure/seed_first_exposure.parquet", prov)
    spread = first.groupby(["benchmark", "item_id"])["t0"].agg(["min", "max", "count"])
    print(f"items: {len(spread)}; median spread of first-read step across runs: "
          f"{int((spread['max'] - spread['min']).median()):,} steps")


if __name__ == "__main__":
    main()
