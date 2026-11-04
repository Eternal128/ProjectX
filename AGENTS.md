# AGENTS.md

Instructions for humans and coding agents working in this repository. Read this file fully before writing code.

## Project

**Seen or Solved? Decomposing Language Model Progress into Memorization and Generalization**
Project X 2026 (UofT AI), Research stream. Theme "What is Progress?"

Benchmark scores mix two things, recalling items seen in training and getting more capable. This project measures how much of Pythia's benchmark progress over training is explained by exposure to benchmark items. It does this with two tools.

1. **Exposure detection.** Find which benchmark items appear in the Pile, and the training step where Pythia first saw each one.
2. **Exposure IRT plus an event study.** An Item Response Theory model with a term for how much easier an item gets after exposure, checked by a difference-in-differences event study around each exposure step.

Closest prior work is Lesci et al. (2024), *Causal Estimation of Memorisation Profiles* (ACL), which runs difference-in-differences on Pythia for training sequences. We differ by studying benchmark items, their effect on measured progress, and by adding the exposure IRT model. Cite it in anything we write.

## Research questions and pre-registered hypotheses

Do not edit this table after week 2 except to log changes in `docs/prereg_changes.md` with a date and reason.

| # | Question | Hypothesis |
| --- | --- | --- |
| RQ1 | Does IRT ability tell a different story of progress than raw accuracy across Pythia's training? | H1. Accuracy and ability disagree on when progress speeds up or stalls. |
| RQ2 | Does seeing an item in training cause a jump in performance on it? | H2. Log-probability of the correct answer rises sharply right after exposure, relative to matched clean items, with no pre-trend. |
| RQ3 | How much of measured progress is explained by exposure? | H3. The exposure term is positive and grows with copy count within standard Pythia. |
| RQ4 | Does the exposure effect depend on ability? | H4. The effect changes with ability, so memorization and generalization do not simply add up. |
| RQ5 | Do exposure gains transfer to paraphrased items? | H5. Gains transfer only partly to paraphrases. |

Every hypothesis is a valid result if it fails. Never tune thresholds or drop items to make a hypothesis pass.

## Models and benchmarks

**Models.** `EleutherAI/pythia-410m`, `EleutherAI/pythia-1.4b`, `EleutherAI/pythia-2.8b`, and the matching `-deduped` versions. Checkpoints are loaded with `revision="stepN"`. Use about 30 checkpoints per model, denser early in training. Verify the available checkpoint list on the Hugging Face model cards before hardcoding it.

**Benchmarks.** Each benchmark must have been public before the Pile was released (end of 2020), and Pythia must score clearly above chance on it.

- Candidates are TriviaQA, SQuAD, LAMBADA, ARC-Easy, SciQ, PIQA.
- Excluded on purpose are GSM8K (released 2021, cannot be in the Pile) and MMLU (released months before the Pile, near chance at this scale).
- Week 1 task is to confirm above-chance scores on the final checkpoints and drop any benchmark that fails.

## Repository layout

```
.
├── AGENTS.md                 # this file
├── README.md                 # short public summary, links to the proposal
├── configs/                  # YAML configs for every run (models, checkpoints, benchmarks, thresholds)
├── data/
│   ├── raw/                  # downloaded benchmarks, never edited by hand
│   ├── exposure/             # candidate matches, tiers, first-exposure steps, copy counts
│   └── evals/                # per-item, per-checkpoint results (parquet)
├── src/
│   ├── exposure/             # retrieval, embedding check, tiering, step mapping
│   ├── evals/                # checkpoint evaluation wrappers
│   ├── irt/                  # standard IRT and exposure IRT models (NumPyro)
│   ├── event_study/          # matching, difference-in-differences, pre-trend test
│   └── figures/              # one script per figure
├── notebooks/                # exploration only, nothing a result depends on
├── tests/                    # unit tests, including IRT parameter recovery on simulated data
└── docs/
    ├── proposal.md           # export of the proposal doc
    └── prereg_changes.md     # dated log of any change to hypotheses or thresholds
```

Large data stays out of git. Commit configs and small manifests, and put bulk files in `data/` (gitignored) with a script that regenerates them.

## Work plan

Gates are hard stops. Do not start the next phase until the gate passes or the fallback is written down.

### Phase 0, week 1. Feasibility

1. Set up the environment and a smoke test that evaluates one Pythia checkpoint on one benchmark.
2. Check above-chance scores on final checkpoints for every candidate benchmark.
3. **Resolve the step-mapping question.** Infini-gram returns positions in the Pile, not Pythia training steps. Pythia's training order comes from its preshuffled, pre-tokenized dataset on Hugging Face and the batch tooling in the `EleutherAI/pythia` GitHub repo. Find the cheapest way to map a matched document or token span to the step where it was trained. Options to compare are (a) searching the preshuffled token files directly for the candidate n-grams with a multi-pattern matcher, and (b) mapping Pile document indices to the shuffled order if the tooling supports it. Write the chosen method and its storage and time cost in `docs/step_mapping.md`.
4. **Go or no-go count.** Count Confirmed exposed items that pass the timing filter below. The main analysis needs at least 200. If fewer, move the Likely tier into the main analysis and drop the ability interaction term. Record the count and decision in `docs/prereg_changes.md`.

### Phase 1, weeks 1 to 4. Exposure detection (`src/exposure/`)

1. **Candidate retrieval.** Query infini-gram over the Pile with long n-grams (13 or more tokens) from the question and from the answer, plus shorter n-grams for a wider candidate set. Check the infini-gram docs for the correct Pile index name.
2. **Semantic check.** Embed each candidate passage and the item with an open sentence-embedding model and store cosine similarity.
3. **Tiering.**
   - `confirmed` is a long exact match covering both question and answer.
   - `likely` is high semantic similarity with no exact match.
   - `clean` is no candidate above a low similarity threshold.
4. **Step mapping and copy counts.** Record first-exposure step `t0` and running copy count `n` for every confirmed item, for both the standard and deduplicated runs.
5. **Timing filter.** Keep confirmed items with at least three checkpoints before `t0` and `t0` between 10% and 90% of training. Report all results stratified by `t0`.
6. **Hand validation.** Two people label a random sample of about 100 items per tier. Report precision per tier.

Missed exposure puts seen items in the clean group, which biases effects toward zero. State this in every write-up.

### Phase 2, weeks 1 to 5. Checkpoint evaluation (`src/evals/`)

1. Use EleutherAI's `lm-evaluation-harness` with per-sample logging.
2. For every (model, checkpoint, item) store both binary correctness `y` and the log-probability of the correct answer `logp`.
3. Save as parquet with columns `model, data_version, step, benchmark, item_id, y, logp, tier, t0, n`.
4. Fixed seeds and pinned library versions in `configs/`. Start with Pythia 410M if GPU access is delayed.

**Inputs must not be mixed.** IRT uses binary `y` only. The event study uses continuous `logp` only.

### Phase 3, weeks 6 to 8. Models (`src/irt/`, `src/event_study/`)

**Exposure IRT.** For item `i`, model `m`, checkpoint `t`,

```
P(y_imt = 1) = sigmoid( a_i * (theta_mt - b_i + delta_imt * E_imt) )
delta_imt    = delta_0 + delta_1 * log(1 + n_it) + delta_2 * theta_mt
```

- `theta_mt` is the ability of one (model, checkpoint) pair, prior Normal(0, 1) to fix the scale.
- `b_i` is base difficulty and `a_i` is discrimination, constrained positive.
- `E_imt` is 1 once model `m` has seen item `i` before checkpoint `t`, else 0.
- `delta_2` is the ability interaction. If anything fails to fit, drop it first.

**Fitting order.** Move to the next step only when the previous one passes.

1. MAP with a single shared delta, on simulated data with known parameters. Test that it recovers them (`tests/test_irt_recovery.py`).
2. Add the dose and ability terms, MAP, recovery test again.
3. SVI in NumPyro on real data for approximate uncertainty.
4. Full NUTS on a random subset of items, only to confirm the SVI intervals.

Also fit standard 2PL IRT (no exposure term) for comparison.

**Event study.**

1. Match each confirmed item to clean items from the same benchmark with similar base difficulty.
2. Difference-in-differences on `logp` around `t0`,
   `Effect(k) = [mean_exp(t0+k) - mean_exp(t0-1)] - [mean_clean(t0+k) - mean_clean(t0-1)]`
3. **Pre-trend test.** Check whether exposed items were already rising faster before `t0`. If so, do not read that group's estimate causally, and say so.
4. Bootstrap 95% confidence intervals over items.

**Dose.** The main dose measure is copy count `n` within standard Pythia. The deduplicated run is a secondary check only, because deduplication changed the whole data distribution.

### Phase 4, weeks 9 to 10. Robustness

1. Sensitivity check adding the `likely` tier to each side in turn.
2. **Transfer check.** Paraphrase a sample of confirmed items (new wording, new names or numbers where present) and compare gains.
3. Exposure share of progress from the counterfactual,
   `M = (dAcc_observed - dAcc_E0) / dAcc_observed`,
   where `dAcc_E0` is predicted accuracy change with every `E` set to 0.

### Weeks 11 to 12

Week 11 is stretch goals (temporal holdouts on OLMo 2, Llama 3.1, Qwen 2.5, or crosscoders on two Pythia checkpoints). Nothing in the core results may depend on them. Week 12 is the write-up and talk.

## Figures

1. **Main figure.** Event study, mean `logp` change for exposed vs matched clean items across checkpoints before and after `t0`. This must be readable without trusting any fitted model.
2. **Second figure.** Observed accuracy over training next to the exposure-removed counterfactual curve, gap shaded.

One script per figure in `src/figures/`, reading only from `data/`. Every figure must be reproducible with one command.

## Conventions

- Every run is driven by a config file, and outputs record the config hash and git commit.
- Never look at final results before the analysis code and thresholds are fixed. Write analysis code against a small development split first.
- Notebooks are for exploration. Anything a result depends on lives in `src/` with a test.
- Commit messages say what changed and why. Open a pull request for any change to `src/irt/` or `src/event_study/`, reviewed by the statistics lead.
- Keep compute in check. The core project should need about 50 to 100 GPU hours, mostly checkpoint evaluation. Flag anything that looks like it will exceed that.

## Team roles

| Role | Owns |
| --- | --- |
| Data lead | Retrieval, embedding check, tiers, step mapping |
| Validation lead | Hand validation, matched clean items, paraphrases |
| Evaluation lead | Checkpoint evaluation, compute |
| Statistics lead | Exposure IRT, event study, counterfactual, figures |

## References

- Lesci, P., Meister, C., Hofmann, T., Vlachos, A. and Pimentel, T. (2024). Causal Estimation of Memorisation Profiles. ACL. https://aclanthology.org/2024.acl-long.834.pdf
- Biderman, S. et al. (2023). Pythia. A Suite for Analyzing Large Language Models Across Training and Scaling. ICML.
- Biderman, S. et al. (2023). Emergent and Predictable Memorization in Large Language Models. NeurIPS.
- Carlini, N. et al. (2023). Quantifying Memorization Across Neural Language Models. ICLR.
- Liu, J. et al. (2024). Infini-gram. Scaling Unbounded n-gram Language Models to a Trillion Tokens. COLM.
- Mirzadeh, I. et al. (2024). GSM-Symbolic. arXiv.
- Polo, F. M. et al. (2024). tinyBenchmarks. ICML.
- Schaeffer, R., Miranda, B. and Koyejo, S. (2023). Are Emergent Abilities of Large Language Models a Mirage? NeurIPS.

All references except Lesci et al. were written from memory. Verify before citing.
