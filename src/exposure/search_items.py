"""Search the Pile for every benchmark item with infini-gram (pilot).

    python -m exposure.search_items [--limit N]

Per item, two exact-text queries against the Pile index (configs/exposure.yaml):
  question   the full question text
  qa         the question and the correct answer together, as a CNF "AND" query that
             counts places where both appear within a short window of each other

Pilot tiers (the semantic "likely" tier needs an embedding pass and is not in the pilot):
  confirmed      question (at least `min_question_words` words) found together with its answer
  question_only  question found, answer not nearby
  clean          question not found anywhere in the Pile
Short questions are left out of both groups ("ambiguous"): a short generic phrase can
match without the item having been seen.

Writes data/exposure/pilot_matches.parquet and data/exposure/pilot_tiers.parquet.
A missed copy (reworded, reformatted) lands in "clean", which biases effects toward zero.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from common.config import load_config, provenance, write_parquet
from evals.benchmarks import load_items
from exposure.retrieval import InfiniGramClient


def tier_of(question: str, q_count: int, qa_count: int, min_words: int) -> str:
    if len(question.split()) < min_words:
        return "ambiguous"
    if qa_count > 0:
        return "confirmed"
    if q_count > 0:
        return "question_only"
    return "clean"


def clean_text(s: str) -> str:
    return " ".join(s.replace(" AND ", " and ").replace(" OR ", " or ").split())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exposure.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-question-words", type=int, default=8)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    client = InfiniGramClient.from_config(cfg)
    items = load_items()
    if args.limit:
        items = items[: args.limit]

    rows, t_start = [], time.time()
    for k, it in enumerate(items):
        q, a = clean_text(it["question"]), clean_text(it["answer_text"])
        q_count = client.count(q)
        qa_count = client.count(f"{q} AND {a}") if q_count > 0 else 0
        rows.append({"benchmark": it["benchmark"], "item_id": it["item_id"], "question_words": len(q.split()),
                     "q_count": q_count, "qa_count": qa_count,
                     "tier": tier_of(q, q_count, qa_count, args.min_question_words)})
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(items)} items, {time.time() - t_start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    prov = provenance({"exposure": cfg, "min_question_words": args.min_question_words, "n_items": len(items)}, args.config)
    write_parquet(df, "data/exposure/pilot_tiers.parquet", prov)
    print(df.groupby(["benchmark", "tier"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
