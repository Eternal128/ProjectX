# Step mapping: Pile position to Pythia training step

**Status: open (Phase 0, week 1). Owner: data lead.**

Infini-gram returns positions in the Pile, not Pythia training steps. This document
records how we map a matched document or token span to the step where Pythia trained
on it, and what that costs. Nothing below is verified yet; facts marked *(verify)*
were written from memory.

## What the code assumes

`src/exposure/step_mapping.py` implements the arithmetic once a position in the
**preshuffled training stream** is known:

- Training stream is a sequence of fixed-length sequences of 2049 tokens *(verify)*.
- Batch size is 1024 sequences, about 2M tokens per step *(verify)*.
- 143,000 steps in total *(verify)*.
- Batch `b` (0-based) is consumed by optimizer step `b + 1`, so checkpoint `stepN`
  has seen batches `0 .. N-1`. **Verify this off-by-one** against the Pythia repo
  before the first real run.
- `t0 = first_batch + 1`; `E(t) = 1` iff `t >= t0`; `n(t)` = occurrences in batches `< t`.

These values live in `configs/models.yaml` under `training`.

## Sources to check

- Pythia preshuffled, pre-tokenized datasets on Hugging Face (standard and deduplicated
  Pile) *(verify exact dataset names)*.
- Batch tooling in the `EleutherAI/pythia` GitHub repo (batch viewer / memmap dataset
  utilities) *(verify path and interface)*.
- Infini-gram docs: Pile index name, what a `find` result returns (shard, suffix-array
  rank, document index) *(verify)*.

## Options

### (a) Search the preshuffled token files directly

Tokenize candidate n-grams with the GPT-NeoX tokenizer and scan the preshuffled token
files with a multi-pattern matcher (e.g. Aho-Corasick over token ids). Each hit's token
offset maps directly to a batch.

- Storage: TBD (size of the preshuffled token files for each of standard / deduped).
- Time: TBD (one or more full passes over ~300B tokens per data version).
- Pros: gives every occurrence, so copy counts `n` come for free; no dependency on
  document indices. Cons: heavy I/O; tokenization-boundary effects on matches.

### (b) Map Pile document indices to the shuffled order

If infini-gram returns Pile document indices and the Pythia tooling exposes the
document-to-shuffled-sequence permutation, look up each matched document's position.

- Storage: TBD. Time: TBD.
- Pros: cheap per query. Cons: only works if the permutation is recoverable; documents
  are packed and split across sequences, so the span offset inside the document matters;
  infini-gram's Pile index may not share document order with Pythia's tokenized Pile.

## Decision

TBD. Record: chosen method, why, measured storage and time cost, and a validation check
(e.g. for a handful of known matches, decode the batch at the computed step and confirm
the text is there).
