# CPU research pilot

A first real measurement, small enough to run on a CPU, that produces data the website can show.
Branch: `james/research-pilot`.

## What it measures

- **Model:** Pythia 410M (standard Pile), one of the six study models.
- **Checkpoints:** 10, chosen from steps where EleutherAI also published accuracy, so our scoring
  can be checked against theirs: 512, 1000, 3000, 13000, 33000, 53000, 73000, 93000, 113000, 143000.
- **Items:** every question in SciQ test (1,000), ARC-Easy test (2,376) and PIQA validation (1,838),
  5,214 in all, downloaded from their public sources (`src/evals/benchmarks.py`).
- **Per item and checkpoint:** correct or not, and the log-probability of the right answer
  (`src/evals/score_cpu.py`), in the same zero-shot format as lm-evaluation-harness.
- **Exposure:** an infini-gram search of the Pile for each question, alone and together with its
  answer (`src/exposure/search_items.py`), giving the groups confirmed / question only / clean.

## What it can and cannot show

It can show whether questions found in Pythia's training text are answered better, and how their
scores move over training, compared with questions that were not found. It cannot place each
question at the training step it was first read: that needs a search of the preshuffled training
files (hundreds of GB), so the event study and the exposure IRT model wait for the full study.
The semantic "likely" tier is also left for later. Missed copies count as clean, which biases any
difference toward zero.

## Requirements

Network access to `huggingface.co`, `cdn-lfs.huggingface.co`, `cdn-lfs-us-1.hf.co`,
`cas-bridge.xethub.hf.co` and `api.infini-gram.io`; `pip install torch transformers safetensors`.
About 1 GB of disk per checkpoint while it is scored (deleted afterwards).

## Commands

```bash
cd src
python -m evals.benchmarks                       # 5,214 items -> data/raw/items.jsonl
python -m evals.score_cpu --steps 143000 --limit 50   # smoke test, compare with published accuracy
python -m evals.score_cpu --steps 512 1000 3000 13000 33000 53000 73000 93000 113000 143000
python -m exposure.search_items                  # Pile search -> data/exposure/pilot_tiers.parquet
python -m figures.site_pilot                     # -> site/data/pilot.json for the website
```

Scoring skips checkpoints already done, so it can be stopped and restarted.
