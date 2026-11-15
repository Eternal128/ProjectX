"""Rebuild GPT-NeoX's training index maps (doc_idx, sample_idx, shuffle_idx) in numpy.

EleutherAI publishes Pythia's index maps for the standard Pile only
(`EleutherAI/pile-preshuffled-seeds`). The deduplicated runs used the same recipe, the
same seed (1234) and a different document list, so we rebuild them. The recipe follows
GPT-NeoX's `megatron/data/gpt2_dataset.py` (`_build_index_mappings`):

  num_epochs  smallest e with (e * tokens_per_epoch - 1) // seq_length >= num_samples
  doc_idx     the document ids repeated num_epochs times, int32, shuffled in place by
              np.random.RandomState(seed)
  sample_idx  for k = 0 .. (num_epochs * tokens_per_epoch - 1) // seq_length: the
              (doc_idx position, token offset) holding stream token k * seq_length, where
              the stream is the documents in doc_idx order (GPT-NeoX's C++
              `build_sample_idx`, written here as a vectorised search)
  shuffle_idx arange(number of samples), uint32, shuffled by the same generator next

`check_against_published` rebuilds the standard maps and compares them with the
published ones byte for byte; only if they match do we trust a rebuilt deduplicated map.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common.config import REPO_ROOT

OUT = REPO_ROOT / "data" / "step_mapping"


def num_epochs(tokens_per_epoch: int, seq_length: int, num_samples: int) -> int:
    e, total = 0, 0
    while True:
        e += 1
        total += tokens_per_epoch
        if (total - 1) // seq_length >= num_samples:
            return e


def build(sizes: np.ndarray, num_samples: int, seq_length: int = 2048, seed: int = 1234):
    sizes = np.asarray(sizes)
    tokens_per_epoch = int(sizes.sum(dtype=np.int64))
    epochs = num_epochs(tokens_per_epoch, seq_length, num_samples)
    rng = np.random.RandomState(seed=seed)
    doc_idx = np.tile(np.arange(len(sizes), dtype=np.int32), epochs)
    rng.shuffle(doc_idx)
    # sample boundaries: stream token k * seq_length lives in the doc_idx position whose span holds it
    ends = np.cumsum(sizes[doc_idx], dtype=np.int64)
    n = (epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.empty((n + 1, 2), dtype=np.int32)
    chunk = 8_000_000  # bounded memory: a few hundred MB per chunk
    for a in range(0, n + 1, chunk):
        pos = np.arange(a, min(a + chunk, n + 1), dtype=np.int64) * seq_length
        j = np.searchsorted(ends, pos, side="right")
        sample_idx[a:a + len(pos), 0] = j
        sample_idx[a:a + len(pos), 1] = pos - (ends[j] - sizes[doc_idx[j]])
    del ends
    shuffle_idx = np.arange(n, dtype=np.uint32)
    rng.shuffle(shuffle_idx)
    return doc_idx, sample_idx, shuffle_idx


def check_against_published(maps_dir: Path = OUT / "seed0") -> bool:
    from exposure.pile_order import MAPS

    sizes = np.load(OUT / "standard_sizes.npy")
    t = time.time()
    doc_idx, sample_idx, shuffle_idx = build(sizes, num_samples=147_164_160)
    print(f"rebuilt standard maps in {time.time() - t:.0f}s")
    ok = True
    for name, mine in [("doc_idx", doc_idx), ("sample_idx", sample_idx), ("shuffle_idx", shuffle_idx)]:
        theirs = np.load(maps_dir / f"{MAPS}_{name}.npy", mmap_mode="r")
        same = theirs.shape == mine.shape and theirs.dtype == mine.dtype and bool(np.array_equal(theirs, mine))
        print(f"{name}: published {theirs.shape} {theirs.dtype}, rebuilt {mine.shape} {mine.dtype}, identical: {same}")
        ok &= same
    return ok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="rebuild the standard maps and compare with the published ones")
    ap.add_argument("--deduped", action="store_true", help="build the deduplicated maps (num_samples as for standard)")
    args = ap.parse_args(argv)
    if args.check:
        print("standard maps match the published ones" if check_against_published() else "MISMATCH: do not use rebuilt maps")
    if args.deduped:
        d = OUT / "deduped_seed1234"
        d.mkdir(parents=True, exist_ok=True)
        doc_idx, sample_idx, shuffle_idx = build(np.load(OUT / "deduped_sizes.npy"), num_samples=147_164_160)
        for name, a in [("doc_idx", doc_idx), ("sample_idx", sample_idx), ("shuffle_idx", shuffle_idx)]:
            np.save(d / f"{name}.npy", a)
        print(f"wrote {d}: doc_idx {doc_idx.shape}, sample_idx {sample_idx.shape}, shuffle_idx {shuffle_idx.shape}")


if __name__ == "__main__":
    main()
