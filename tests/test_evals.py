import pandas as pd
import pytest

from evals.extract import sample_to_row, samples_to_rows
from evals.schema import EVAL_COLUMNS, validate_eval_frame


def test_multiple_choice_extraction_uses_target_choice():
    s = {"doc_id": 7, "target": 2, "acc": 1.0,
         "filtered_resps": [["-3.0", "False"], ["-2.5", "False"], ["-1.25", "True"], ["-4.0", "False"]]}
    assert sample_to_row(s, "multiple_choice") == {"item_id": "7", "y": 1, "logp": -1.25}


def test_loglikelihood_extraction():
    s = {"doc_id": 3, "target": " word", "acc": 0, "filtered_resps": [(-0.7, False)]}
    assert sample_to_row(s, "loglikelihood") == {"item_id": "3", "y": 0, "logp": -0.7}


def test_generation_tasks_need_gold_loglikelihood():
    with pytest.raises(NotImplementedError):
        sample_to_row({"doc_id": 0, "target": "x", "acc": 1, "filtered_resps": ["x"]}, "generate_until")


def test_schema_validation():
    rows = samples_to_rows([{"doc_id": 0, "target": 0, "acc": 1, "filtered_resps": [[-1.0, True]]}],
                           model="pythia-410m", data_version="standard", step=1000,
                           benchmark="arc_easy", output_type="multiple_choice")
    df = pd.DataFrame(rows, columns=EVAL_COLUMNS)
    validate_eval_frame(df)
    with pytest.raises(ValueError):
        validate_eval_frame(df.assign(y=2))
    with pytest.raises(ValueError):
        validate_eval_frame(pd.concat([df, df]))
