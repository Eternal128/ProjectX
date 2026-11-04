# Pre-registration change log

The research questions and hypotheses in `AGENTS.md` are fixed after week 2. Any later
change to a hypothesis, threshold, filter, or the analysis plan is logged here with a
date and a reason, **before** looking at the results it would affect.

Never tune thresholds or drop items to make a hypothesis pass. A failed hypothesis is a
valid result.

| Date | What changed | Old | New | Reason | Who |
| --- | --- | --- | --- | --- | --- |
| 2026-11-04 | Initial provisional thresholds committed in `configs/` (no data seen) | — | see `configs/exposure.yaml`, `configs/event_study.yaml`, `configs/benchmarks.yaml` | Repository setup. To be fixed on the development split before any main-split result. | repo setup |

## Phase 0 go / no-go decision

To be filled in at the end of week 1 (AGENTS.md, Phase 0 step 4).

- Confirmed exposed items passing the timing filter: **TBD**
- Threshold: 200 (`configs/exposure.yaml: go_no_go.min_confirmed_after_timing_filter`)
- Decision: **TBD** (if below 200: move the `likely` tier into the main analysis and drop
  the ability interaction term `delta_2`)

## Benchmark above-chance check

To be filled in during week 1 (AGENTS.md, Phase 0 step 2). One row per candidate.

| Benchmark | Chance | Final-checkpoint accuracy (pythia-410m) | Keep / drop |
| --- | --- | --- | --- |
| TriviaQA | 0 | TBD (not in published results) | TBD |
| SQuAD | 0 | TBD (not in published results) | TBD |
| LAMBADA | 0 | 0.516 (published) | keep, provisional |
| ARC-Easy | 0.25 | 0.521 (published) | keep, provisional |
| SciQ | 0.25 | 0.811 (published) | keep, provisional |
| PIQA | 0.5 | 0.668 (published) | keep, provisional |

"Published" numbers are EleutherAI's zero-shot results for `pythia-410m` at step 143000 (Pythia
repository, commit `a19eecb`; `python -m evals.pythia_published`). Under the rule in
`configs/benchmarks.yaml` all four pass for all six study models, and MMLU passes for none, which
confirms its exclusion. Logged 2026-11-04; to be confirmed with our own evaluation in week 1.
