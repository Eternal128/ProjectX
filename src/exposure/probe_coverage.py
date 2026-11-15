"""How many benchmark items appear in the Pile at all? A coverage probe (pilot, Phase 0).

    python -m exposure.probe_coverage [--sample 500]

The pilot search (`exposure.search_items`) asks whether each whole question appears in the
Pile word for word. This probe asks two wider questions, to inform the go / no-go count
(AGENTS.md, Phase 0 step 4) before any result is looked at:

  windows     For every pilot question of at least 13 words: do its first 13 words, or its
              last 13 words, appear in the Pile? This is the long-n-gram rule of Phase 1,
              and it catches copies that differ only outside the window (a different
              prefix such as "Question:", a reworded ending).
  lambada     A seeded random sample of LAMBADA test passages: do the passage's last 13
              words (which end in the word the model must predict) appear in the Pile?
  triviaqa    A seeded random sample of TriviaQA validation questions (rc.nocontext): does
              the whole question appear, and does it appear with its answer nearby? Same
              rule as the pilot search.

Writes data/exposure/probe_coverage.parquet (one row per item and check) and prints the
share found per benchmark with a 95% Wilson interval. Exact matching only: a copy that is
reworded anywhere inside the window is missed, so every share here is a lower bound.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config, provenance, write_parquet
from evals.benchmarks import load_items
from exposure.retrieval import InfiniGramClient
from exposure.search_items import clean_text

WINDOW = 13
HF = "https://huggingface.co/datasets"
LAMBADA_URL = f"{HF}/EleutherAI/lambada_openai/resolve/main/data/lambada_test_en.jsonl"
TRIVIAQA_URL = f"{HF}/mandarjoshi/trivia_qa/resolve/main/rc.nocontext/validation-00000-of-00001.parquet"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def windows(text: str, size: int = WINDOW) -> dict[str, str]:
    w = text.split()
    if len(w) < size:
        return {}
    return {"first": " ".join(w[:size]), "last": " ".join(w[-size:])}


def download(url: str, name: str):
    import requests

    path = REPO_ROOT / "data" / "raw" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exposure.yaml")
    ap.add_argument("--sample", type=int, default=500, help="items sampled from LAMBADA and from TriviaQA")
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args(argv)
    client = InfiniGramClient.from_config(load_config(args.config))
    rows: list[dict] = []
    t_start = time.time()

    def add(bench, item_id, check, query):
        n = client.count(clean_text(query))
        rows.append({"benchmark": bench, "item_id": str(item_id), "check": check, "count": n})
        if len(rows) % 500 == 0:
            print(f"{len(rows)} queries, {time.time() - t_start:.0f}s", flush=True)
        return n

    # 1. 13-word windows of every pilot question long enough to have one
    for it in load_items():
        for side, q in windows(clean_text(it["question"])).items():
            add(it["benchmark"], it["item_id"], f"window_{side}", q)

    rng = np.random.default_rng(args.seed)
    # 2. LAMBADA: last 13 words of the passage, ending in the target word
    lam = [json.loads(line) for line in download(LAMBADA_URL, "lambada_test_en.jsonl").read_text().splitlines() if line.strip()]
    for i in sorted(rng.choice(len(lam), size=min(args.sample, len(lam)), replace=False).tolist()):
        w = windows(clean_text(lam[i]["text"]))
        if w:
            add("lambada", i, "window_last", w["last"])

    # 3. TriviaQA: whole question, then question with its answer nearby
    tq = pd.read_parquet(download(TRIVIAQA_URL, "triviaqa_rc_nocontext_validation.parquet"), columns=["question_id", "question", "answer"])
    tq = tq.drop_duplicates("question").reset_index(drop=True)
    for i in sorted(rng.choice(len(tq), size=min(args.sample, len(tq)), replace=False).tolist()):
        r = tq.iloc[i]
        q = clean_text(r["question"])
        if add("triviaqa", r["question_id"], "question", q) > 0:
            add("triviaqa", r["question_id"], "question_and_answer", f"{q} AND {clean_text(r['answer']['value'])}")

    df = pd.DataFrame(rows)
    prov = provenance({"window": WINDOW, "sample": args.sample, "seed": args.seed, "lambada": LAMBADA_URL, "triviaqa": TRIVIAQA_URL}, args.config)
    write_parquet(df, "data/exposure/probe_coverage.parquet", prov)
    print(summarise(df).to_string(index=False))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Share of items found per benchmark: any window hit, or (TriviaQA) question / question+answer."""
    out = []
    for bench, g in df.groupby("benchmark"):
        if bench == "triviaqa":
            q = g[g["check"] == "question"]
            n = q["item_id"].nunique()
            for label, ids in [("question found", set(q.loc[q["count"] > 0, "item_id"])),
                               ("question and answer found", set(g.loc[(g["check"] == "question_and_answer") & (g["count"] > 0), "item_id"]))]:
                lo, hi = wilson(len(ids), n)
                out.append({"benchmark": bench, "check": label, "found": len(ids), "items": n, "share": len(ids) / n, "ci_low": lo, "ci_high": hi})
        else:
            hit = g.groupby("item_id")["count"].max() > 0
            lo, hi = wilson(int(hit.sum()), len(hit))
            out.append({"benchmark": bench, "check": "13-word window found", "found": int(hit.sum()), "items": len(hit),
                        "share": float(hit.mean()), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out).round(4)


if __name__ == "__main__":
    main()
