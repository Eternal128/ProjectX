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


def test_find_subsequence_returns_every_start():
    import numpy as np
    from exposure.pile_order import find_subsequence

    hay = np.array([5, 1, 2, 3, 9, 1, 2, 3, 1, 2], dtype=np.uint16)
    assert find_subsequence(hay, np.array([1, 2, 3], dtype=np.uint16)) == [1, 5]
    assert find_subsequence(hay, np.array([1, 2, 4], dtype=np.uint16)) == []
    assert find_subsequence(hay[:2], np.array([1, 2, 3], dtype=np.uint16)) == []


def _toy_order():
    """Three documents of 5, 3 and 6 tokens, shuffled to order [2, 0, 1], cut into samples
    of 4 tokens + 1 overlap, as GPT-NeoX does; samples then shuffled."""
    import numpy as np
    from exposure.pile_order import TrainingOrder

    o = TrainingOrder.__new__(TrainingOrder)
    o.doc_idx = np.array([2, 0, 1])
    # stream positions: doc2 = 0..5, doc0 = 6..10, doc1 = 11..13 ; samples start every 4 tokens
    o.sample_idx = np.array([[0, 0], [0, 4], [1, 2], [2, 1], [2, 3]])
    shuffle = np.array([3, 0, 2, 1])
    o.inverse = np.empty(len(shuffle), dtype=np.int64)
    o.inverse[shuffle] = np.arange(len(shuffle))
    o._sample_doc = o.sample_idx[:, 0]
    return o


def test_samples_of_doc_cover_every_token_of_the_document():
    o = _toy_order()
    # sample k covers stream tokens 4k .. 4k+4 (inclusive, one token of overlap)
    assert o.samples_of_doc(2) == [0, 1]        # doc 2, tokens 0..5
    assert o.samples_of_doc(0) == [1, 2]        # doc 0, tokens 6..10
    assert o.samples_of_doc(1) == [2, 3]        # doc 1, tokens 11..13


def test_training_position_inverts_the_shuffle():
    o = _toy_order()
    assert [o.training_position(s) for s in range(4)] == [1, 3, 2, 0]
