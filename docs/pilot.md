# CPU research pilot

A first real measurement, small enough to run on a CPU, that produces data the website can show.
Run 2026-11-15 on branch `james/pilot-results` (4 CPU cores, no GPU).

## What it measures

- **Model:** Pythia 410M (standard Pile), one of the six study models.
- **Checkpoints:** 10, chosen from steps where EleutherAI also published accuracy, so our scoring
  can be checked against theirs: 512, 1000, 3000, 13000, 33000, 53000, 73000, 93000, 113000, 143000.
- **Items:** every question in SciQ test (1,000), ARC-Easy test (2,376) and PIQA validation (1,838),
  5,214 in all (`src/evals/benchmarks.py`); plus 500 LAMBADA test passages, the seeded sample whose
  Pile exposure the coverage probe measured (`--probe-sample`).
- **Per item and checkpoint:** correct or not, and the log-probability of the right answer
  (`src/evals/score_cpu.py`), in the same zero-shot format as lm-evaluation-harness.
- **Exposure:** infini-gram searches of the Pile (`src/exposure/search_items.py`,
  `src/exposure/probe_coverage.py`), then the training step at which each found item was read
  (`src/exposure/date_items.py`, method in `docs/step_mapping.md`).

## Results

### 1. Our scoring reproduces EleutherAI's published accuracy

| Test | Checkpoints scored | Largest gap to the published accuracy |
| --- | --- | --- |
| SciQ (1,000) | 6 (512 to 53000) | 0.4 points |
| ARC-Easy (2,376) | 6 (512 to 53000) | 0.1 points |
| PIQA (1,838) | 6 (512 to 53000) | 0.2 points |
| LAMBADA (500-passage sample) | all 10 | 2.0 points (sampling: published is on all 5,153) |

The per-checkpoint table is in `site/data/pilot.json` (`accuracy`) and on the website. The last 4
checkpoints of SciQ / ARC-Easy / PIQA (73000 to 143000) were not reached: the scoring run is
capped at 2 hours per background job in this cloud session. To finish (about 2.5 hours on 4
cores; it resumes and skips the 6 done):

```bash
cd src && python -m evals.score_cpu --steps 512 1000 3000 13000 33000 53000 73000 93000 113000 143000
python -m figures.site_pilot
```

### 2. Almost none of the SciQ, ARC-Easy and PIQA questions are in the Pile; a quarter of LAMBADA is

| Test | Check | Found | Share (95% CI) |
| --- | --- | --- | --- |
| SciQ, ARC-Easy, PIQA | whole question word for word | 14 of 5,214 (2 with the answer nearby) | 0.3% |
| ARC-Easy | first or last 13 words of the question | 8 of 1,554 | 0.5% (0.3 to 1.0) |
| SciQ | same | 4 of 449 | 0.9% (0.4 to 2.3) |
| PIQA | same | 1 of 121 | 0.8% (0.1 to 4.5) |
| TriviaQA (500 sampled) | question, and question with its answer nearby | 4, and 0 | 0.8%, and 0% (0 to 0.8) |
| **LAMBADA (500 sampled)** | **last 13 words of the passage, ending in the answer word** | **117** | **23.4% (19.9 to 27.3)** |

Most PIQA goals (1,042) and some SciQ and ARC-Easy questions are too short to search
meaningfully and are left out ("ambiguous"). Exact matching misses reworded copies, so every share
is a lower bound and missed copies push any effect toward zero.

### 3. Found items can be dated to the training step that read them

All 14 found SciQ / ARC-Easy / PIQA questions were dated: 27 of their 32 Pile copies were confirmed
inside the exact training batch Pythia read (the other 5 fall after training ends or across a sample
boundary). First reads range from step 7,917 to step 132,700. Example: ARC-Easy `MCAS_2003_8_17`
was read at step 23,734 and again at step 133,946. LAMBADA dating: see `docs/prereg_changes.md`.

### 4. Found LAMBADA passages are not answered better (descriptive)

| Step | Found in the Pile (n = 117) | Not found (n = 383) |
| --- | --- | --- |
| 3,000 | 28.2% (20.5 to 36.8) | 25.1% (21.1 to 29.2) |
| 13,000 | 40.2% (31.6 to 48.7) | 43.1% (38.1 to 47.8) |
| 53,000 | 45.3% (36.8 to 53.8) | 45.7% (40.7 to 50.7) |
| 143,000 | 48.7% (39.3 to 58.1) | 52.7% (48.0 to 57.7) |

Accuracy with 95% bootstrap intervals over passages. The mean log-probability of the answer word is
about 0.2 nats lower for found passages late in training. The intervals overlap at every
checkpoint. This compares different passages, so it is not a causal estimate: found passages may
be harder, and many were read late. The event study (each passage before vs after its own first
read, against matched clean passages) is what separates these, and it is the next step.

## What it cannot show

The pilot compares groups of items; it does not estimate what reading an item did to the model.
The event study and the exposure IRT model need dated items, matched clean items and denser
checkpoints, and they wait for the analysis code and thresholds to be fixed on a development split
(AGENTS.md). The semantic "likely" tier is also left for later. Missed copies count as clean, which
biases any difference toward zero.

## Requirements

Network access to `huggingface.co`, `cas-bridge.xethub.hf.co` and `api.infini-gram.io`;
`pip install torch transformers safetensors pandas pyarrow requests`. About 2 GB of disk per
checkpoint while it is scored (deleted afterwards), and 2.8 GB for the training-order index maps.
The public infini-gram API rate-limits sustained traffic; the client backs off and caches.

## Commands

```bash
cd src
python -m evals.benchmarks                                    # 5,214 items -> data/raw/items.jsonl
python -m evals.score_cpu --steps 143000 --limit 50           # smoke test, compare with published accuracy
python -m evals.score_cpu --steps 512 1000 3000 13000 33000 53000 73000 93000 113000 143000
python -m exposure.search_items                               # whole-question Pile search -> pilot_tiers.parquet
python -m exposure.probe_coverage                             # 13-word windows, LAMBADA and TriviaQA samples
python -m evals.benchmarks --benchmarks lambada --probe-sample --out data/raw/items_lambada_probe.jsonl
python -m evals.score_cpu --steps 512 1000 3000 13000 33000 53000 73000 93000 113000 143000 \
    --items ../data/raw/items_lambada_probe.jsonl --tag lambada
python -m exposure.date_items --source pilot                  # date found questions (needs the index maps)
python -m exposure.date_items --source lambada
python -m figures.site_pilot                                  # -> site/data/pilot.json for the website
```

The index maps for dating are downloaded once into `data/step_mapping/seed0/` (see
`docs/step_mapping.md`). Scoring skips checkpoints already done, so it can be stopped and restarted.
