"""Turn lm-evaluation-harness per-sample logs into evaluation rows.

For each sample we need binary correctness ``y`` and the log-probability of the
correct answer ``logp``. lm-eval logs ``filtered_resps`` as (loglikelihood,
is_greedy) pairs, one per choice for multiple-choice tasks and one for
loglikelihood tasks. Values may arrive as strings after JSON serialisation.

Generation tasks (TriviaQA, SQuAD) do not log a gold-answer loglikelihood; they
need a companion loglikelihood task that scores the gold answer. Until that
exists, extraction raises NotImplementedError for them.
"""
from __future__ import annotations

from typing import Any, Iterable


def _ll(pair: Any) -> float:
    if isinstance(pair, (list, tuple)):
        pair = pair[0]
    return float(pair)


def _target_index(sample: dict[str, Any]) -> int:
    target = sample["target"]
    if isinstance(target, (list, tuple)):
        if len(target) != 1:
            raise ValueError(f"expected a single target, got {target!r}")
        target = target[0]
    if isinstance(target, str) and target.strip().lstrip("-").isdigit():
        target = int(target)
    if not isinstance(target, int):
        raise ValueError(f"cannot map target {target!r} to a choice index")
    return target


def sample_to_row(sample: dict[str, Any], output_type: str, correct_metric: str = "acc") -> dict[str, Any]:
    resps = sample["filtered_resps"]
    if output_type == "multiple_choice":
        logp = _ll(resps[_target_index(sample)])
    elif output_type == "loglikelihood":
        logp = _ll(resps[0])
    else:
        raise NotImplementedError(
            f"output_type={output_type!r}: add a gold-answer loglikelihood task to get logp"
        )
    y = int(round(float(sample[correct_metric])))
    return {"item_id": str(sample["doc_id"]), "y": y, "logp": logp}


def samples_to_rows(
    samples: Iterable[dict[str, Any]],
    *,
    model: str,
    data_version: str,
    step: int,
    benchmark: str,
    output_type: str,
) -> list[dict[str, Any]]:
    rows = []
    for s in samples:
        row = sample_to_row(s, output_type)
        row.update(model=model, data_version=data_version, step=int(step), benchmark=benchmark,
                   tier=None, t0=None, n=None)
        rows.append(row)
    return rows
