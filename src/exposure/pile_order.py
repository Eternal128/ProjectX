"""Date when Pythia read a Pile document: from an infini-gram match to a training step.

This is option (b) of docs/step_mapping.md, which turned out to be cheap because
EleutherAI publishes every piece needed:

1. infini-gram (`v4_piletrain_llama`) finds a passage and returns its Pile document
   number, source file and line.
2. Pythia's tokenized, unshuffled Pile (`EleutherAI/pythia_pile_idxmaps`, 210,604,984
   documents) holds the same documents in the same order, minus a few thousand empty
   ones, so the document number can be off by a little. We read a window of documents
   around it (HTTP range requests, a few MB) and find the one that holds the passage.
3. GPT-NeoX's index maps for Pythia's run (`EleutherAI/pile-preshuffled-seeds`, folder
   seed0, seed 1234; about 2.8 GB) give the document order (doc_idx), how documents
   were cut into 2,049-token samples (sample_idx) and the shuffle of samples
   (shuffle_idx). Training position i = inverse_shuffle[sample]; step batch = i // 1024.
4. Check: read the 2,049-token sample at position i from the preshuffled training file
   (`EleutherAI/pile-standard-pythia-preshuffled`, 4,098 bytes per sample) and confirm
   the passage's tokens are in it. Only a verified sample dates an item.

Batch b is consumed by optimizer step b + 1 (`exposure.step_mapping`); the off-by-one
does not matter at checkpoint spacing but is kept consistent.
"""
from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

from common.config import REPO_ROOT

HF = "https://huggingface.co/datasets/EleutherAI"
IDXMAP = f"{HF}/pythia_pile_idxmaps/resolve/main/pile_20B_tokenizer_text_document"
IDXMAP_SHARD_BYTES = 5_000_000_000
PRESHUFFLED = f"{HF}/pile-standard-pythia-preshuffled/resolve/main/document"
DEDUP_IDXMAP = f"{HF}/pythia_deduped_pile_idxmaps/resolve/main/pile_0.87_deduped_text_document"
DEDUP_PRESHUFFLED = f"{HF}/pile-deduped-pythia-preshuffled/resolve/main/document"
PRESHUFFLED_SHARD_BYTES = 30_000_000_000
SEQ_TOKENS = 2049
SAMPLE_BYTES = SEQ_TOKENS * 2  # uint16 tokens
BATCH = 1024
TRAINED_SAMPLES = 143_000 * BATCH
DOC_WINDOW = 3000  # infini-gram vs GPT-NeoX document numbering drift (empty documents)
# measured drift: 1,339 at doc 99,410,851 and 2,572 at doc 195,793,549, about 1.32e-5 per
# document; search a narrow window around the predicted drift first, the full window after
DRIFT_PER_DOC = 1.32e-5
NARROW_WINDOW = 300


def find_neox_docs(pile: "TokenizedPile", ig_doc: int, needle: np.ndarray) -> list[int]:
    """Pythia's document numbers that hold `needle`, near infini-gram's document `ig_doc`."""
    guess = ig_doc - int(round(ig_doc * DRIFT_PER_DOC))
    for lo, hi in ((guess - NARROW_WINDOW, guess + NARROW_WINDOW), (ig_doc - DOC_WINDOW, ig_doc + DOC_WINDOW)):
        lo = max(0, lo)
        toks, starts = pile.window(lo, hi)
        docs = sorted({lo + int(np.searchsorted(starts, h, side="right")) - 1 for h in find_subsequence(toks, needle)})
        if docs:
            return docs
    return []
MAPS = "pile_20B_tokenizer_text_document_train_0_indexmap_147164160ns_2048sl_1234s"


def _get(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end) of a remote file."""
    import requests

    for attempt in range(6):
        r = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
        if r.status_code == 206 and len(r.content) == end - start:
            return r.content
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
    raise RuntimeError(f"range read failed: {url} {start}-{end}")


def _read_sharded(prefix: str, shard_bytes: int, n_shards: int, start: int, end: int) -> bytes:
    """Read [start, end) of a file published as equal-size shards `prefix-0000k-of-000NN.bin`."""
    out = b""
    while start < end:
        k = start // shard_bytes
        stop = min(end, (k + 1) * shard_bytes)
        out += _get(f"{prefix}-{k:05d}-of-{n_shards:05d}.bin", start - k * shard_bytes, stop - k * shard_bytes)
        start = stop
    return out


class TokenizedPile:
    """Remote reader for Pythia's unshuffled tokenized Pile (GPT-NeoX mmap format)."""

    def __init__(self, prefix: str = IDXMAP, n_shards: int = 132):
        self.prefix, self.n_shards = prefix, n_shards
        head = _get(f"{prefix}.idx", 0, 34)
        assert head[:9] == b"MMIDIDX\x00\x00", "not a GPT-NeoX index file"
        self.n_docs = struct.unpack("<Q", head[18:26])[0]
        self._sizes_at = 34
        self._pointers_at = 34 + 4 * self.n_docs

    def sizes(self, lo: int, hi: int) -> np.ndarray:
        return np.frombuffer(_get(f"{self.prefix}.idx", self._sizes_at + 4 * lo, self._sizes_at + 4 * hi), dtype=np.int32)

    def pointers(self, lo: int, hi: int) -> np.ndarray:
        return np.frombuffer(_get(f"{self.prefix}.idx", self._pointers_at + 8 * lo, self._pointers_at + 8 * hi), dtype=np.int64)

    def window(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        """Tokens of documents lo..hi-1 (contiguous) and each document's start offset in them."""
        lo, hi = max(0, lo), min(self.n_docs, hi)
        sizes = self.sizes(lo, hi).astype(np.int64)
        p0 = int(self.pointers(lo, lo + 1)[0])
        raw = _read_sharded(self.prefix, IDXMAP_SHARD_BYTES, self.n_shards, p0, p0 + 2 * int(sizes.sum()))
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        return np.frombuffer(raw, dtype=np.uint16), starts


def find_subsequence(hay: np.ndarray, needle: np.ndarray) -> list[int]:
    """Start offsets of `needle` in `hay` (both 1-D integer arrays)."""
    n = len(needle)
    if n == 0 or len(hay) < n:
        return []
    cand = np.flatnonzero(hay[: len(hay) - n + 1] == needle[0])
    for k in range(1, n):
        if not len(cand):
            break
        cand = cand[hay[cand + k] == needle[k]]
    return cand.tolist()


class TrainingOrder:
    """Pythia's training order (standard Pile, seed 1234) from GPT-NeoX index maps on disk."""

    def __init__(self, maps_dir: str | Path | None = None):
        d = Path(maps_dir) if maps_dir else REPO_ROOT / "data" / "step_mapping" / "seed0"
        self.doc_idx = np.load(d / f"{MAPS}_doc_idx.npy", mmap_mode="r")
        self.sample_idx = np.load(d / f"{MAPS}_sample_idx.npy", mmap_mode="r")
        shuffle = np.load(d / f"{MAPS}_shuffle_idx.npy")
        # int32 is enough (162M samples, 210M documents) and halves memory next to the scoring runs
        self.inverse = np.empty(len(shuffle), dtype=np.int32)
        self.inverse[shuffle] = np.arange(len(shuffle), dtype=np.int32)
        del shuffle
        self._sample_doc = np.asarray(self.sample_idx[:, 0])

    @property
    def _doc_positions(self):
        """Where each document sits in doc_idx, built once. Pythia's standard run is a single
        epoch, so doc_idx is a permutation; with several epochs fall back to a scan."""
        if not hasattr(self, "_pos_cache"):
            d = np.asarray(self.doc_idx)
            if len(d) == d.max() + 1:
                inv = np.empty(len(d), dtype=np.int32)
                inv[d] = np.arange(len(d), dtype=np.int32)
                self._pos_cache = inv
            else:
                self._pos_cache = None
        return self._pos_cache

    @lru_cache(maxsize=None)
    def doc_position(self, doc: int) -> np.ndarray:
        inv = self._doc_positions
        if inv is not None:
            return np.array([inv[doc]])
        return np.flatnonzero(np.asarray(self.doc_idx) == doc)

    def samples_of_doc(self, doc: int) -> list[int]:
        """Unshuffled sample indices that contain any token of document `doc`."""
        out: list[int] = []
        for j in self.doc_position(doc).tolist():
            a = int(np.searchsorted(self._sample_doc, j, side="left"))
            b = int(np.searchsorted(self._sample_doc, j, side="right"))
            out.extend(range(max(a - 1, 0), min(b, len(self._sample_doc) - 1)))
        return out

    def training_position(self, sample: int) -> int:
        return int(self.inverse[sample])


def read_training_sample(position: int, preshuffled: str = PRESHUFFLED) -> np.ndarray:
    """The 2,049 tokens Pythia trained on at training position `position` (batch = position // 1024)."""
    raw = _read_sharded(preshuffled, PRESHUFFLED_SHARD_BYTES, 20, position * SAMPLE_BYTES, (position + 1) * SAMPLE_BYTES)
    return np.frombuffer(raw, dtype=np.uint16)


def reconstruct_sample(position: int, doc_idx, sample_idx, shuffle_idx, pile: TokenizedPile) -> np.ndarray:
    """Rebuild the training sample at `position` from the unshuffled Pile and the index maps,
    the way GPT-NeoX's dataset does: documents doc_idx[f..l], from offset f to offset l inclusive."""
    s = int(shuffle_idx[position])
    (jf, of), (jl, ol) = (int(v) for v in sample_idx[s]), (int(v) for v in sample_idx[s + 1])
    parts = []
    for j in range(jf, jl + 1):
        d = int(doc_idx[j])
        toks, _ = pile.window(d, d + 1)
        lo = of if j == jf else 0
        hi = ol + 1 if j == jl else len(toks)
        parts.append(toks[lo:hi])
    return np.concatenate(parts)
