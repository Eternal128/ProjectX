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
from pathlib import Path

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
    """Download a run's three map files, resuming any partial file until it has the size the
    Hub lists (an interrupted transfer otherwise leaves a short file behind)."""
    import json
    import urllib.request

    d = STEP_DIR / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/EleutherAI/pile-preshuffled-seeds/tree/main/seed{seed}"
    with urllib.request.urlopen(api, timeout=60) as r:
        sizes = {Path(x["path"]).name: int(x["size"]) for x in json.load(r)}
    for kind in ("doc_idx", "sample_idx", "shuffle_idx"):
        f = d / f"{maps_name(seed)}_{kind}.npy"
        for _ in range(5):
            if f.exists() and f.stat().st_size == sizes[f.name]:
                break
            subprocess.run(["curl", "-sS", "-L", "--retry", "5", "-C", "-", "-o", str(f),
                            f"{SEED_URL}/seed{seed}/{f.name}"], check=False)
        if not f.exists() or f.stat().st_size != sizes[f.name]:
            raise RuntimeError(f"could not download {f.name}")
    return d


def check_or_repair_shuffle(seed: int, d) -> str:
    """The published shuffle must be a permutation. If it is not (seed 2's file is mostly
    zeros), rebuild it with GPT-NeoX's recipe (exposure.index_maps) and accept the rebuild
    only if it agrees with every surviving non-zero entry of the published file."""
    f = d / f"{maps_name(seed)}_shuffle_idx.npy"
    pub = np.load(f, mmap_mode="r")
    if (np.bincount(np.asarray(pub), minlength=len(pub)) == 1).all():
        return "published file is a permutation"
    from exposure.index_maps import build

    _, _, rebuilt = build(np.load(STEP_DIR / "standard_sizes.npy"), num_samples=147_164_160, seed=seed)
    ok = np.asarray(pub) != 0
    agree = int((np.asarray(pub)[ok] == rebuilt[ok]).sum())
    if agree != int(ok.sum()):
        raise RuntimeError(f"seed {seed}: rebuilt shuffle disagrees with {int(ok.sum()) - agree} published entries")
    np.save(f, rebuilt)
    return f"published file broken ({int((~ok).sum()):,} zero entries); rebuilt, agrees with all {agree:,} others"


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
        maps_dir = fetch_maps(seed) if seed else STEP_DIR / "seed0"
        print(f"seed {seed}: {check_or_repair_shuffle(seed, maps_dir)}", flush=True)
        order = TrainingOrder(maps_dir, name=maps_name(seed))
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
