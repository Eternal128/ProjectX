# Step mapping: Pile position to Pythia training step

**Status: resolved for standard Pythia (2026-11-15), verified on real items. For the
deduplicated runs the training order is rebuilt and verified; finding a matched passage's
document in the deduplicated Pile is still open.** Code: `src/exposure/pile_order.py`,
`src/exposure/index_maps.py`, `src/exposure/date_items.py`.

Infini-gram returns positions in the Pile, not Pythia training steps. This document records
how we map a matched passage to the step where Pythia trained on it, what it costs, and how
we checked it.

## Decision: option (b), map documents through GPT-NeoX's index maps

Every piece is published by EleutherAI, so no training data has to be downloaded in full.

| Step | Source | What it gives |
| --- | --- | --- |
| 1. Find the passage | infini-gram, index `v4_piletrain_llama` (`find`, then `get_doc_by_rank`) | Pile document number, source file and line (e.g. `14.jsonl:1124868`) |
| 2. Pin Pythia's document | `EleutherAI/pythia_pile_idxmaps` (Pythia's tokenized, unshuffled Pile), range reads | The same document in Pythia's numbering |
| 3. Find its training samples | `EleutherAI/pile-preshuffled-seeds`, folder `seed0` (seed 1234): `doc_idx`, `sample_idx`, `shuffle_idx` | Which 2,049-token samples hold the document, and each sample's training position |
| 4. Verify | `EleutherAI/pile-standard-pythia-preshuffled`, one 4,098-byte range read per sample | The passage's tokens really are in the batch Pythia trained on at that step |

Training position `i` is in batch `i // 1024`; batch `b` is consumed by optimizer step
`b + 1` (`src/exposure/step_mapping.py`). Only samples that pass step 4 count.

### Facts measured on the published files

- **Tokenized Pile:** 210,604,984 documents (index header). Infini-gram's Pile train index
  numbers its documents slightly differently: Pythia's number is lower by **1,339 at
  document 99.4M and 2,572 at document 195.8M**, consistent with empty documents dropped
  during tokenization. We therefore read a window of ±3,000 documents (17 to 20 MB) and
  find the document that holds the passage's tokens. About 4 to 5 seconds per copy here.
- **Index maps (seed 1234):** 2.79 GB in total (`doc_idx` 842 MB, `sample_idx` 1.30 GB,
  `shuffle_idx` 649 MB). They describe one epoch of 162,165,686 samples. Pythia trained on
  the first **146,432,000** training positions (143,000 steps × 1,024), so a document can
  land after the end of training and is then never read.
- **Preshuffled training file:** 601.8 GB per data version (20 shards of 30 GB plus
  78 MB) = exactly 146,432,000 samples × 2,049 tokens × 2 bytes.

### Validation

ARC-Easy item `MCAS_2003_8_17` ("Which of the following symbiotic relationships is
considered parasitic?") has 2 copies in the Pile, both in Pile-CC. The method predicts
the training samples for each, and reading those samples back from the preshuffled file
finds the question's tokens in exactly one sample per copy:

| Copy (infini-gram doc → Pythia doc) | Samples holding the document | Verified sample | Step |
| --- | --- | --- | --- |
| 195,793,549 → 195,790,977 | 4 | position 24,302,757 | **23,734** |
| 99,410,851 → 99,409,512 | 4 (3 after training ends) | position 137,159,922 | **133,946** |

The neighbouring samples of the same documents do not contain the question, so the check
is not passing by chance. First exposure `t0` for this item is step 23,734 (17% of
training), and it was read again at step 133,946. `python -m exposure.date_items` runs
the same check on every found pilot question.

## Option (a), scanning the training file, is not practical here

Searching the 601.8 GB preshuffled file directly needs one full pass per data version.
Downloads in our cloud container are capped at about **7 MB/s**, the same with 1 or 8
parallel connections (measured on 500 MB and 8 × 200 MB range reads), so one pass would
take about **24 hours** per version. On a machine with a 100 MB/s link it would take
under 2 hours, and it would still be useful as an independent check of copy counts.

## Rebuilt index maps (needed for the deduplicated runs)

EleutherAI publishes index maps for the standard Pile only. `src/exposure/index_maps.py`
rebuilds them from per-document token counts with GPT-NeoX's recipe (seed 1234). Checks:

- **Standard, against the published maps:** rebuilt `doc_idx` (210,604,984), `sample_idx`
  (162,165,687 × 2) and `shuffle_idx` (162,165,686) are **byte-for-byte identical** to the
  published seed0 files (`python -m exposure.index_maps --check`, 44 s).
- **Samples rebuilt from the unshuffled Pile** (`reconstruct_sample`) equal Pythia's
  preshuffled training file at positions 0, 24,302,757, 77,777,777 and 146,431,999.
- **Deduplicated:** the deduplicated Pile has 134,318,121 documents, so the recipe needs
  2 epochs (268,636,242 entries in `doc_idx`, 202,314,857 samples, of which training used
  the first 146,432,000). Samples rebuilt with these maps equal Pythia's
  `pile-deduped-pythia-preshuffled` file at positions 0, 1, 50,000,000, 100,000,123 and
  146,431,999. The maps take 3.5 GB and rebuild in about a minute
  (`python -m exposure.index_maps --deduped`), so they are not stored.

## The 9 other training runs (PolyPythias, the seed check)

`EleutherAI/pile-preshuffled-seeds` also publishes the index maps of the 9 PolyPythias
reruns (`seed1` … `seed9`, 2.8 GB each). Document numbers are shared, so once a copy's place
in the Pile is known (document and token offset), its training step in any run is index
arithmetic (`python -m exposure.seed_dates`; maps downloaded and deleted one run at a time).

- **Check:** for the original run the arithmetic puts 208 of 211 copies at exactly the
  training position confirmed by reading Pythia's batches (the other 3 are documents that
  hold the passage twice).
- **seed2's published shuffle file is broken.** 149,810,208 of its 162,165,686 entries are
  0, so it is not a permutation and every item looked first-read at step 1. Rebuilding the
  seed-2 maps with `exposure.index_maps` gives a permutation that agrees with **all
  12,355,478** surviving entries, so the rebuild is used. `seed_dates` checks every run's
  shuffle this way; seeds 0, 1 and 3 to 9 are valid as published.
- **Result (128 dated items: 14 pilot questions and 114 LAMBADA passages from the sample):**
  each run reads 119 to 128 of them within training (about 10% of each run's shuffled
  order falls after step 143,000, as expected). The same item is first read at very
  different steps in different runs: median spread across runs 112,925 steps (IQR 97,180
  to 126,937); 80% of the variance in first-read step for LAMBADA passages is within
  passage, across runs; runs 0 and 1 correlate at 0.16. In each run 77 to 94 of the 114
  LAMBADA passages pass the timing filter, and 112 pass it in at least 5 of the 10 runs.

So the seed check has what it needs: the same passage read early in some runs and late in
others, which compares a passage with itself instead of with other passages.
PolyPythias runs exist for sizes up to 410M only.

## Still open

- **Finding a passage's document in the deduplicated Pile.** Infini-gram indexes the
  standard Pile, and the deduplicated Pile numbers its 134M documents differently. A test
  on 12 deduplicated documents spread across the file, looked up in infini-gram, gave
  standard positions that roughly rise with the deduplicated position but not cleanly: a
  kept document often has several copies in the standard Pile, so the order cannot be
  read off one lookup. Two ways forward: (1) a binary search over deduplicated document
  numbers that uses all copies of each probe (about 30 lookups per item), or (2) scanning
  the deduplicated training file directly for the matched tokens (option a) on a machine
  with a fast link. The dedup comparison is a secondary check (AGENTS.md), so this does
  not block the main analysis.
- **Off-by-one.** Whether batch `b` is first reflected in checkpoint `step b` or `step b+1`
  is still unconfirmed. It shifts `t0` by one step, far below the checkpoint spacing.
- **Copies split across two samples.** A passage cut by a sample boundary fails the check
  and is not counted, which biases copy counts slightly down. In the pilot, 27 of the 32
  Pile copies of the 14 found questions were confirmed inside a training batch; the other
  5 fall after the end of training or across a boundary.
