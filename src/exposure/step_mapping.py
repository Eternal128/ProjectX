"""Map positions in Pythia's training stream to training steps, and compute
first-exposure step ``t0`` and running copy counts ``n``.

Conventions (see docs/step_mapping.md, which records how positions are obtained):

* The preshuffled training stream is a sequence of fixed-length sequences of
  ``sequence_length`` tokens; batch ``b`` (0-based) holds sequences
  ``b*batch_size .. (b+1)*batch_size - 1``.
* Batch ``b`` is consumed by optimizer step ``b + 1``, so checkpoint ``stepN``
  has seen batches ``0 .. N-1``.
* ``t0`` is the first checkpoint step whose weights include the item:
  ``t0 = first_batch + 1``. Exposure indicator ``E(t) = 1`` iff ``t >= t0``.
* ``n(t)`` is the number of occurrences in batches ``< t`` (copies seen by
  checkpoint ``t``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingLayout:
    total_steps: int = 143_000
    batch_size_sequences: int = 1024
    sequence_length_tokens: int = 2049

    @classmethod
    def from_config(cls, models_cfg: dict) -> "TrainingLayout":
        t = models_cfg["training"]
        return cls(t["total_steps"], t["batch_size_sequences"], t["sequence_length_tokens"])

    def batch_of_sequence(self, seq_index):
        return np.asarray(seq_index) // self.batch_size_sequences

    def batch_of_token(self, token_offset):
        """Batch index of a token offset in the flattened preshuffled stream."""
        return self.batch_of_sequence(np.asarray(token_offset) // self.sequence_length_tokens)


def first_exposure_step(batch_indices) -> int | None:
    """``t0`` from the batch indices where an item occurs, or None if never seen."""
    b = np.asarray(batch_indices)
    return None if b.size == 0 else int(b.min()) + 1


def copy_count_at(batch_indices, checkpoint_steps) -> np.ndarray:
    """Running copy count ``n`` at each checkpoint: occurrences in batches < step."""
    b = np.sort(np.asarray(batch_indices))
    return np.searchsorted(b, np.asarray(checkpoint_steps), side="left")


def exposure_indicator(t0: int | None, checkpoint_steps) -> np.ndarray:
    steps = np.asarray(checkpoint_steps)
    if t0 is None:
        return np.zeros_like(steps, dtype=np.int8)
    return (steps >= t0).astype(np.int8)


def passes_timing_filter(
    t0: int | None,
    checkpoint_steps,
    total_steps: int,
    min_checkpoints_before_t0: int = 3,
    t0_min_fraction: float = 0.10,
    t0_max_fraction: float = 0.90,
) -> bool:
    """AGENTS.md timing filter: at least ``min_checkpoints_before_t0`` checkpoints
    strictly before ``t0``, and ``t0`` between the given fractions of training."""
    if t0 is None:
        return False
    steps = np.asarray(checkpoint_steps)
    n_before = int(np.sum(steps < t0))
    return (
        n_before >= min_checkpoints_before_t0
        and t0_min_fraction * total_steps <= t0 <= t0_max_fraction * total_steps
    )
