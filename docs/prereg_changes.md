# Pre-registration change log

The research questions and hypotheses in `AGENTS.md` are fixed after week 2. Any later
change to a hypothesis, threshold, filter, or the analysis plan is logged here with a
date and a reason, **before** looking at the results it would affect.

Never tune thresholds or drop items to make a hypothesis pass. A failed hypothesis is a
valid result.

| Date | What changed | Old | New | Reason | Who |
| --- | --- | --- | --- | --- | --- |
| 2026-11-04 | Initial provisional thresholds committed in `configs/` (no data seen) | — | see `configs/exposure.yaml`, `configs/event_study.yaml`, `configs/benchmarks.yaml` | Repository setup. To be fixed on the development split before any main-split result. | repo setup |
| 2026-11-15 | infini-gram request rate in `configs/exposure.yaml` | 5 per second | 2 per second, with back-off | The public API rate-limited (429) after about 2,000 queries. Operational only; no threshold or result affected. | pilot run |

## Phase 0 go / no-go decision

To be decided by the team at the end of week 1 (AGENTS.md, Phase 0 step 4). Evidence from the
CPU pilot (2026-11-15, Pythia 410M standard; `docs/pilot.md`, `docs/step_mapping.md`):

| Benchmark | Confirmed items found | Dated to a training step | Pass the timing filter |
| --- | --- | --- | --- |
| SciQ, ARC-Easy, PIQA (all 5,214) | 2 (question and answer); 14 with the question alone | 14 of 14 | 2 confirmed; 11 of the 14 |
| LAMBADA (seeded sample of 500 of 5,153) | 117 (last 13 words, ending in the answer word) | 114 of 117 | **94** |

- Confirmed exposed items passing the timing filter: **877 for LAMBADA, counted**, plus 2 from
  SciQ / ARC-Easy / PIQA. The 500-passage sample had estimated about 969 (94 of 500, 95% Wilson
  interval 805 to 1,157 when scaled to 5,153 passages). Measured for the standard 410M run; the
  timing filter uses the 30 study checkpoints in `configs/models.yaml`.
- **Counted directly, complete (2026-11-17):** all 5,153 LAMBADA passages searched; **1,169**
  (22.7%, 95% CI 21.6 to 23.9%) have their last 13 words in the Pile; 1,107 are dated to a
  training step (1,736 of 1,902 copies confirmed inside a training batch); **877 pass the timing
  filter** (17.0% of LAMBADA, 95% CI 16.0 to 18.1%). 168 were first read in the first 10% of
  training and 62 in the last 10%. Copies per passage: 1 to 14, mostly 1 (608) or 2 (439).
- Threshold: 200 (`configs/exposure.yaml: go_no_go.min_confirmed_after_timing_filter`)
- What the rule implies: the count is above 200, so the pre-registered fallback (move the
  `likely` tier in, drop `delta_2`) is **not** triggered. Nearly all exposed items come from
  LAMBADA, so the event study and the exposure IRT model would in practice be LAMBADA analyses;
  SciQ, ARC-Easy and PIQA serve mostly as clean comparison items.
- Things the team should weigh before deciding:
  - Copies per item are few (1 to 5, mostly 1 or 2; 184 of 198 copies confirmed in a training
    batch), so H3's dose term `delta_1` will be estimated from a narrow range.
  - The deduplicated runs are not yet countable (`docs/step_mapping.md`, still open).
  - Copies are almost all in BookCorpus2 and Books3, so exposure is confounded with "comes from a
    book in the Pile"; matched clean items should be drawn from LAMBADA passages too.
- Decision: **TBD (team)**. Nothing in the hypotheses or thresholds has been changed.

## Proposed additions (to decide and log before the week-2 freeze)

Two additions came out of the team discussion on 2026-11-07. Neither is pre-registered yet;
if adopted, each needs a hypothesis and threshold written here before any result it affects
is looked at.

1. **Seed check.** Compare each exposed passage with itself across the 10 Pythia training runs
   (the original plus 9 PolyPythias reruns, sizes up to 410M), which read it at different
   steps. Feasibility (2026-11-15, `docs/step_mapping.md`): all 128 found pilot items are
   dated in every run; median spread of first-read step 112,925 steps; 80% of LAMBADA
   first-read variance is within passage.
2. **Planted-question test.** Continue training a small Pythia from a mid-training checkpoint
   with chosen questions inserted at known steps and copy counts, and check the event study
   recovers the planted effect and estimate how much missed copies shrink it. Not started.

## Benchmark above-chance check

To be filled in during week 1 (AGENTS.md, Phase 0 step 2). One row per candidate.

| Benchmark | Chance | Final-checkpoint accuracy (pythia-410m) | Keep / drop |
| --- | --- | --- | --- |
| TriviaQA | 0 | TBD (not in published results) | TBD |
| SQuAD | 0 | TBD (not in published results) | TBD |
| LAMBADA | 0 | 0.516 (published); 0.518 ours on the 500-passage sample | keep, provisional |
| ARC-Easy | 0.25 | 0.521 (published) | keep, provisional |
| SciQ | 0.25 | 0.811 (published) | keep, provisional |
| PIQA | 0.5 | 0.668 (published) | keep, provisional |

"Published" numbers are EleutherAI's zero-shot results for `pythia-410m` at step 143000 (Pythia
repository, commit `a19eecb`; `python -m evals.pythia_published`). Under the rule in
`configs/benchmarks.yaml` all four pass for all six study models, and MMLU passes for none, which
confirms its exclusion. Logged 2026-11-04; to be confirmed with our own evaluation in week 1.
