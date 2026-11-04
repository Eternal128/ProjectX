import numpy as np

from exposure.step_mapping import (TrainingLayout, copy_count_at, exposure_indicator,
                                   first_exposure_step, passes_timing_filter)


def test_layout_batch_math():
    lay = TrainingLayout(total_steps=143_000, batch_size_sequences=1024, sequence_length_tokens=2049)
    assert lay.batch_of_sequence(0) == 0
    assert lay.batch_of_sequence(1023) == 0
    assert lay.batch_of_sequence(1024) == 1
    assert lay.batch_of_token(2049 * 1024 - 1) == 0
    assert lay.batch_of_token(2049 * 1024) == 1


def test_t0_and_exposure_indicator():
    t0 = first_exposure_step([500, 1999, 20])
    assert t0 == 21  # batch 20 is consumed by step 21
    assert first_exposure_step([]) is None
    np.testing.assert_array_equal(exposure_indicator(21, [1, 20, 21, 100]), [0, 0, 1, 1])
    np.testing.assert_array_equal(exposure_indicator(None, [1, 2]), [0, 0])


def test_copy_counts():
    # occurrences in batches 5, 5, 10 -> seen by checkpoint step N if batch < N
    np.testing.assert_array_equal(copy_count_at([10, 5, 5], [5, 6, 10, 11, 1000]), [0, 2, 2, 3, 3])


def test_timing_filter():
    ckpts = [512, 1000, 2000, 3000, 20000, 50000, 143000]
    assert passes_timing_filter(20000, ckpts, 143_000)
    assert not passes_timing_filter(None, ckpts, 143_000)
    assert not passes_timing_filter(10000, ckpts, 143_000)            # before 10% of training
    assert not passes_timing_filter(130000, ckpts, 143_000)           # after 90%
    assert not passes_timing_filter(20000, [1000, 2000, 143000], 143_000)  # only 2 checkpoints before
