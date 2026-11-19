"""Download the benchmark questions and write one normalised item table.

    python -m evals.benchmarks

Sources (public, reachable without Hugging Face):
  SciQ      test, 1,000 items   AI2 public datasets, CC BY-NC 3.0
  ARC-Easy  test, 2,376 items   AI2 public datasets, CC BY-SA 4.0
  PIQA      validation, 1,838   the PIQA authors' GitHub (its test labels are not public)

Each item is stored with the exact zero-shot prompt and answer continuations used by
lm-evaluation-harness 0.4 for these tasks, so per-item scores are comparable with the
published Pythia results. LAMBADA (test, 5,153 passages, from EleutherAI's
lambada_openai on Hugging Face) is built only when asked for, into its own file:

    python -m evals.benchmarks --benchmarks lambada --out data/raw/items_lambada.jsonl

Its single continuation is the passage's last word; an item is right when every token of
that word is the model's top choice (lm-evaluation-harness `lambada_openai`, metric acc).

Writes data/raw/benchmarks/<name>.jsonl (raw downloads, gitignored) and
data/raw/items.jsonl (one row per item: benchmark, item_id, prompt, choices, answer, text).
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from common.config import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"
SOURCES = {
    "sciq": "https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip",
    "arc_easy": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip",
    "piqa": "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/",
    "lambada": "https://huggingface.co/datasets/EleutherAI/lambada_openai/resolve/main/data/lambada_test_en.jsonl",
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def _cached(name: str, fetch) -> list[dict]:
    path = RAW / "benchmarks" / f"{name}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = fetch()
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return [json.loads(line) for line in path.read_text().splitlines()]


def fetch_sciq() -> list[dict]:
    z = zipfile.ZipFile(io.BytesIO(_get(SOURCES["sciq"])))
    name = next(n for n in z.namelist() if n.endswith("/test.json") and "__MACOSX" not in n)
    return json.loads(z.read(name))


def fetch_arc_easy() -> list[dict]:
    z = zipfile.ZipFile(io.BytesIO(_get(SOURCES["arc_easy"])))
    name = next(n for n in z.namelist() if n.endswith("ARC-Easy/ARC-Easy-Test.jsonl") and "__MACOSX" not in n)
    return [json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]


def fetch_piqa() -> list[dict]:
    goals = [json.loads(line) for line in _get(SOURCES["piqa"] + "valid.jsonl").decode().splitlines() if line.strip()]
    labels = [int(x) for x in _get(SOURCES["piqa"] + "valid-labels.lst").decode().split()]
    assert len(goals) == len(labels)
    return [{**g, "label": lab} for g, lab in zip(goals, labels)]


def fetch_lambada() -> list[dict]:
    return [json.loads(line) for line in _get(SOURCES["lambada"]).decode().splitlines() if line.strip()]


# ---- prompts, matching lm-evaluation-harness 0.4 task configs ----

def items_sciq(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        choices = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
        out.append({
            "benchmark": "sciq", "item_id": f"sciq-{i}",
            "prompt": f"{r['support'].lstrip()}\nQuestion: {r['question']}\nAnswer:",
            "choices": [f" {c}" for c in choices], "answer": 3,
            "question": r["question"], "answer_text": r["correct_answer"],
        })
    return out


def items_arc_easy(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        labels = [c["label"] for c in r["question"]["choices"]]
        texts = [c["text"] for c in r["question"]["choices"]]
        ans = labels.index(r["answerKey"])
        out.append({
            "benchmark": "arc_easy", "item_id": r["id"],
            "prompt": f"Question: {r['question']['stem']}\nAnswer:",
            "choices": [f" {t}" for t in texts], "answer": ans,
            "question": r["question"]["stem"], "answer_text": texts[ans],
        })
    return out


def items_piqa(rows: list[dict]) -> list[dict]:
    return [{
        "benchmark": "piqa", "item_id": f"piqa-{i}",
        "prompt": f"Question: {r['goal']}\nAnswer:",
        "choices": [f" {r['sol1']}", f" {r['sol2']}"], "answer": r["label"],
        "question": r["goal"], "answer_text": [r["sol1"], r["sol2"]][r["label"]],
    } for i, r in enumerate(rows)]


def items_lambada(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        context, word = r["text"].rsplit(" ", 1)
        out.append({
            "benchmark": "lambada", "item_id": f"lambada-{i}", "prompt": context,
            "choices": [f" {word}"], "answer": 0, "greedy": True,
            "question": context, "answer_text": word,
        })
    return out


BUILDERS = {"sciq": (fetch_sciq, items_sciq), "arc_easy": (fetch_arc_easy, items_arc_easy), "piqa": (fetch_piqa, items_piqa),
            "lambada": (fetch_lambada, items_lambada)}
PILOT = ["sciq", "arc_easy", "piqa"]


def load_items(path: str | Path = RAW / "items.jsonl") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+", default=PILOT, choices=list(BUILDERS))
    ap.add_argument("--out", default=str(RAW / "items.jsonl"))
    ap.add_argument("--probe-sample", action="store_true",
                    help="LAMBADA: keep only the passages sampled by exposure.probe_coverage, whose exposure is known")
    args = ap.parse_args(argv)
    items = []
    for name in args.benchmarks:
        fetch, build = BUILDERS[name]
        rows = build(_cached(name, fetch))
        if args.probe_sample and name == "lambada":
            import pandas as pd

            probe = pd.read_parquet(REPO_ROOT / "data" / "exposure" / "probe_coverage.parquet")
            keep = {f"lambada-{i}" for i in probe.loc[probe["benchmark"] == "lambada", "item_id"]}
            rows = [r for r in rows if r["item_id"] in keep]
        print(f"{name}: {len(rows)} items")
        items += rows
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.write_text("".join(json.dumps(r) + "\n" for r in items))
    print(f"wrote {out} ({len(items)} items)")


if __name__ == "__main__":
    main()
