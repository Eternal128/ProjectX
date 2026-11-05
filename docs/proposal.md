# Seen or Solved? Decomposing Language Model Progress into Memorization and Generalization

**Project X 2026 (UofT AI), Research stream.** Theme: *What is Progress?*
**Status:** research proposal, 4 November 2026. Hypotheses are pre-registered (section 2); any later change
is logged with a date and reason in [`prereg_changes.md`](prereg_changes.md).

---

## Summary

Benchmark curves are how the field answers "what is progress?" for language models. Pythia 2.8B's
accuracy on SciQ climbs from 21.9% to 82.9% in its first 13,000 training steps, about 9% of training,
and adds only 5.3 more points over the remaining 91% (EleutherAI's published results). But a score
rises for two reasons it cannot tell apart: the model may have become more capable, or it may have
seen the test items during training and recalled them. **Because Pythia's training data and its order
are public, we can date the moment each benchmark question was first read and measure how much of the
progress curve that reading explains.**

Contributions:

- **Exposure dating.** Find benchmark items in the Pile and date their first exposure to the training
  step, with hand-validated precision per tier (sections 5.1 to 5.3).
- **A causal effect of exposure.** An event study around each item's first exposure, with a pre-trend
  test (section 5.6).
- **The share of progress that is exposure.** An exposure-aware IRT model and its counterfactual,
  tested for dose and ability effects and transfer to paraphrases (sections 5.5, 5.7, 5.8).

Pythia is unusually well suited to this. Every model in the suite was trained on the same public
corpus (the Pile) in the same fixed, published order, with 154 checkpoints saved along the way. That
lets us find benchmark items in the training data, name the training step at which each one was
first seen, and watch the model's performance on that item at checkpoints before and after.

We combine two tools. **Exposure detection** finds benchmark items in the Pile and maps each match to
the step where Pythia first trained on it. An **exposure IRT model**, checked by a model-free
**difference-in-differences event study**, then estimates how much easier an item becomes after
exposure and what share of measured progress that accounts for.

The closest prior work is Lesci et al. (2024), who run difference-in-differences on Pythia for
training sequences. We study benchmark items instead, their effect on measured progress, and add the
exposure IRT model.

---

## 1. Motivation

Benchmark curves over training are the standard picture of "progress". If part of a curve's rise is
recall of items that happened to appear in training data, the curve overstates capability, and the
overstatement can grow or shrink as training proceeds. Contamination studies usually ask a static
question (*is this item in the training data?*). We ask a dynamic one: *when the model sees an item,
does its performance on that item jump, and how much of the curve do such jumps add up to?*

Answering it requires three things that rarely come together: the training data, its exact order,
and many intermediate checkpoints. Pythia provides all three.

## 2. Research questions and pre-registered hypotheses

| # | Question | Hypothesis |
| --- | --- | --- |
| RQ1 | Does IRT ability tell a different story of progress than raw accuracy across Pythia's training? | **H1.** Accuracy and ability disagree on when progress speeds up or stalls. |
| RQ2 | Does seeing an item in training cause a jump in performance on it? | **H2.** Log-probability of the correct answer rises sharply right after exposure, relative to matched clean items, with no pre-trend. |
| RQ3 | How much of measured progress is explained by exposure? | **H3.** The exposure term is positive and grows with copy count within standard Pythia. |
| RQ4 | Does the exposure effect depend on ability? | **H4.** The effect changes with ability, so memorization and generalization do not simply add up. |
| RQ5 | Do exposure gains transfer to paraphrased items? | **H5.** Gains transfer only partly to paraphrases. |

Every hypothesis is a valid result if it fails. Thresholds are never tuned, and items are never
dropped, to make a hypothesis pass.

## 3. Related work

- **Lesci et al. (2024), *Causal Estimation of Memorisation Profiles* (ACL).** Difference-in-differences
  on Pythia checkpoints to estimate how much a model memorises the sequences it trains on. Our design
  borrows their causal framing; we differ in studying benchmark items rather than training
  sequences, in measuring the effect on benchmark progress, and in the exposure IRT model.
- **Biderman et al. (2023), Pythia.** The model suite, checkpoints and reproducible data order we rely on.
- **Biderman et al. (2023); Carlini et al. (2023).** Memorization in language models: what predicts it
  and how it scales.
- **Liu et al. (2024), Infini-gram.** Exact n-gram search over trillion-token corpora, used for
  candidate retrieval in the Pile.
- **Polo et al. (2024), tinyBenchmarks.** IRT applied to language-model benchmarks.
- **Schaeffer et al. (2023).** Apparent "emergence" can be an artefact of the metric, one reason we model
  continuous log-probability as well as binary accuracy.

Only Lesci et al. has been checked against the source so far; the other references are to be
verified before any citation in a write-up.

## 4. Setting

### 4.1 Models

Pythia 410M, 1.4B and 2.8B, each in two versions: trained on the standard Pile and on the deduplicated
Pile. Checkpoints load from Hugging Face with `revision="stepN"`.

From the Pythia repository README (checked 4 November 2026):

- All model sizes are trained on the exact same data in the exact same order.
- Each model saw 299,892,736,000 tokens: 143,000 steps at a batch of 2,097,152 tokens, that is 1,024
  sequences of 2,048 tokens (stored as 2,049-token rows so targets can be shifted).
- The standard models see just under one epoch of the Pile; the deduplicated models see about 1.5
  epochs of the 207B-token deduplicated Pile.
- 154 checkpoints per model: steps 0, 1, 2, 4, 8, …, 512, 1000, and every 1,000 steps after that.
- The preshuffled, pre-tokenized training data is public (`EleutherAI/pile-standard-pythia-preshuffled`
  and `EleutherAI/pile-deduped-pythia-preshuffled`), with a batch viewer that reads rows
  `step × 1024` to `(step + 1) × 1024` for a given training step.

### 4.2 Checkpoints

We evaluate 30 checkpoints per model, denser early in training, all inside the documented
schedule: 512, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000, 16000, 20000, 24000, 28000, 33000,
38000, 43000, 49000, 55000, 61000, 68000, 75000, 82000, 90000, 98000, 106000, 114000, 122000, 130000,
137000 and 143000 (`configs/models.yaml`). Before the main run we confirm each revision exists on the
Hugging Face hub (`python -m evals.list_checkpoints`).

### 4.3 Benchmarks

Each benchmark must have been public before the Pile was released (end of 2020), and Pythia must score
clearly above chance on it: at least 5 points above chance with a one-sided test against chance at
p < 0.01 (`configs/benchmarks.yaml`).

Candidates: TriviaQA (2017), SQuAD (2016), LAMBADA (2016), ARC-Easy (2018), SciQ (2017), PIQA (2019),
with years from each paper's first arXiv version. Excluded on purpose: GSM8K (released 2021, so it
cannot be in the Pile) and MMLU (near chance at this scale; see below).

**Above-chance check from published results.** EleutherAI publishes zero-shot accuracy for every
Pythia model at 27 checkpoints (lm-evaluation-harness, Apache-2.0, commit `a19eecb`). They cover four
of our candidates and MMLU. Final-checkpoint accuracy:

| Benchmark | Chance | 410m | 410m-deduped | 1.4b | 1.4b-deduped | 2.8b | 2.8b-deduped | Passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAMBADA | 0.00 | 0.516 | 0.524 | 0.616 | 0.619 | 0.647 | 0.651 | all six |
| PIQA | 0.50 | 0.668 | 0.675 | 0.709 | 0.720 | 0.739 | 0.741 | all six |
| SciQ | 0.25 | 0.811 | 0.826 | 0.865 | 0.865 | 0.882 | 0.882 | all six |
| ARC-Easy | 0.25 | 0.521 | 0.517 | 0.607 | 0.617 | 0.644 | 0.635 | all six |
| MMLU | 0.25 | 0.255 | 0.253 | 0.269 | 0.266 | 0.261 | 0.262 | none |

LAMBADA's chance level is taken as zero because the answer is an open-vocabulary word. MMLU is the
unweighted mean of its 57 subjects; no model clears the 5-point margin, which confirms the
exclusion. TriviaQA and SQuAD are not in the published results and are checked in week 1 with our own
evaluation. These are EleutherAI's aggregate numbers, used only for this screening step; the study
itself uses our own per-item evaluations. (`python -m evals.pythia_published` regenerates the table.)

## 5. Method

### 5.1 Exposure detection

1. **Candidate retrieval.** Query infini-gram over the Pile with long n-grams (13 or more tokens) from
   each question and each answer, plus one span that joins the end of the question to the answer, and
   shorter n-grams for a wider candidate set.
2. **Semantic check.** Embed each candidate passage and the item with an open sentence-embedding model
   and store their cosine similarity.
3. **Tiers.** `confirmed`: a long exact match covering both question and answer. `likely`: high semantic
   similarity with no exact match. `clean`: no candidate above a low similarity threshold. Items in
   between are `ambiguous` and left out of both groups.
4. **Hand validation.** Two people label a random sample of about 100 items per tier; we report precision
   per tier.

Missed exposure puts seen items in the clean group, which biases every effect toward zero. We state
this wherever results are reported.

### 5.2 Step mapping

Infini-gram returns positions in the Pile, not training steps. Training step *b* consumes rows
*1024·b* to *1024·(b+1) − 1* of the preshuffled data, so a matched token span maps to the batch that
contains it. Checkpoint `stepN` has seen batches 0 to N − 1, so we define the first-exposure step as
**t0 = first batch + 1**, the exposure indicator as **E(t) = 1 when t ≥ t0**, and the running copy count
**n(t)** as the number of occurrences in batches before *t*. The off-by-one is confirmed against the
batch viewer before the first real run.

Two ways to locate matches in training order are compared in week 1: (a) search the preshuffled token
files directly with a multi-pattern matcher, which also yields copy counts; or (b) map Pile document
indices to the shuffled order, if the tooling allows. The choice and its storage and time cost are
recorded in [`step_mapping.md`](step_mapping.md).

### 5.3 Timing filter

A confirmed item enters the event study only if at least three evaluated checkpoints come before its
t0 and t0 falls between 10% and 90% of training. All results are reported stratified by t0.

### 5.4 Checkpoint evaluation

EleutherAI's lm-evaluation-harness with per-sample logging. For every model, checkpoint and item we
store binary correctness **y** and the log-probability of the correct answer **logp**, in one table with
columns `model, data_version, step, benchmark, item_id, y, logp, tier, t0, n`. Seeds and library
versions are pinned. **The inputs are never mixed:** the IRT model uses y only; the event study uses
logp only.

### 5.5 Exposure IRT

For item *i*, model *m* and checkpoint *t*:

```
P(y_imt = 1) = sigmoid( a_i · (θ_mt − b_i + δ_imt · E_imt) )
δ_imt        = δ_0 + δ_1 · log(1 + n_it) + δ_2 · θ_mt
```

θ_mt is the ability of one model at one checkpoint, with a Normal(0, 1) prior that fixes the scale;
b_i is base difficulty; a_i > 0 is discrimination; δ_1 is the dose term (copy count) and δ_2 the
ability interaction, the first term dropped if anything fails to fit. A standard 2PL model without the
exposure term is fitted for comparison (RQ1).

Fitting proceeds one step at a time, each gated on the previous: (1) MAP with a single shared δ on
simulated data, with a parameter-recovery test; (2) dose and ability terms added, recovery tested
again; (3) stochastic variational inference on real data for approximate uncertainty; (4) full NUTS
on a random subset of items, only to confirm the variational intervals.

### 5.6 Event study

Each confirmed item is matched to clean items from the same benchmark with similar base difficulty,
measured as mean logp over checkpoints in the first 10% of training (before any eligible t0). With
event time *k* counted in checkpoints from exposure:

```
Effect(k) = [mean_exp(t0+k) − mean_exp(t0−1)] − [mean_clean(t0+k) − mean_clean(t0−1)]
```

with 95% confidence intervals from a bootstrap over items. **Pre-trend test:** if exposed items were
already rising faster than their matches before t0 (the bootstrap interval of the mean pre-period
effect excludes zero), that group's estimate is not read causally, and we say so.

### 5.7 Exposure share of progress

From the fitted IRT model, the counterfactual accuracy with every E set to zero gives

```
M = (ΔAcc_observed − ΔAcc_E0) / ΔAcc_observed
```

the share of the observed accuracy gain explained by exposure (RQ3).

### 5.8 Robustness

- Add the `likely` tier to each side (exposed, then clean) in turn.
- **Transfer (RQ5):** paraphrase a sample of confirmed items with new wording, names and numbers, and
  compare their gains with the originals'.
- **Dose:** the main dose measure is copy count within standard Pythia. The deduplicated run is a
  secondary check only, because deduplication changed the whole data distribution, not just the copies.

## 6. Preliminary work

**Code.** The repository contains the full pipeline: exposure detection and step mapping, the
evaluation wrapper and table schema, the IRT models in NumPyro, the event study, and one script per
figure. Every run is driven by a config file and records its config hash and git commit. 27 unit
tests pass, including the IRT recovery tests required by fitting steps 1 and 2.

**Benchmark curves.** The published results above also give accuracy over training at 27 checkpoints
for each benchmark and model. They rise steadily for all four candidate benchmarks and stay flat near
chance for MMLU. We use them only to describe the benchmarks; no threshold or hypothesis depends on
them.

**Method checks on simulated data.** Our own code, run on simulated data with known effects
(`python -m figures.site_methods`):

- *Event study.* 120 exposed and 360 clean simulated items on the real 30-checkpoint schedule, with a
  planted jump of 1.5 in log-probability at exposure and no pre-trend. The estimated effect after
  exposure is between 1.41 and 1.50 at every event time from 0 to 8; before exposure it stays within
  ±0.06 of zero, and the pre-trend test does not fire.
- *Pre-trend detection.* The same panel with a planted pre-exposure rise: the test flags it (mean
  pre-period effect −0.62, 95% interval −0.68 to −0.56).
- *Exposure IRT.* 200 simulated items, two models and 20 checkpoints with a planted δ_0 = 1.0: MAP
  recovers δ_0 = 0.985, with correlations of 0.91 for item difficulty and 0.97 for ability.

These show the estimators recover what is planted. They say nothing yet about Pythia.

## 7. Threats to validity

- **Missed exposure** (paraphrased, reformatted or translated copies) puts seen items in the clean
  group and biases effects toward zero. Hand validation of the clean tier bounds this.
- **False matches** in the confirmed tier (generic questions with common answers) dilute the effect.
  The confirmed tier requires question and answer together; hand validation reports its precision.
- **Selection by timing.** The timing filter keeps items first seen mid-training, which may differ from
  items seen early or late. Results are stratified by t0.
- **Matching.** Matched clean items may differ from exposed ones in ways difficulty does not capture;
  the pre-trend test is the main guard.
- **Harness details.** Prompt format and scoring affect accuracy. We pin versions, and use log-probability
  as well as accuracy so that conclusions do not rest on a threshold metric.
- **Deduplication** changes the whole training distribution, so standard-versus-deduplicated
  comparisons are secondary evidence only.

## 8. Go / no-go gate

The main analysis needs at least **200 confirmed exposed items that pass the timing filter**. If fewer,
the `likely` tier moves into the main analysis and the ability interaction δ_2 is dropped. The count
and decision are recorded in `prereg_changes.md` at the end of week 1.

## 9. Timeline

| Weeks | Phase | Main outputs |
| --- | --- | --- |
| 1 | Feasibility | Smoke test; above-chance check for TriviaQA and SQuAD; step-mapping method and cost; go / no-go count |
| 1–4 | Exposure detection | Candidates, similarity, tiers, t0 and copy counts for both data versions; hand validation |
| 1–5 | Checkpoint evaluation | y and logp for 6 models × 30 checkpoints × all items |
| 6–8 | Models | Exposure IRT (recovery first, then real data); standard 2PL; event study with pre-trend tests |
| 9–10 | Robustness | Likely-tier sensitivity; paraphrase transfer; exposure share of progress |
| 11 | Stretch | Held-out questions on newer models, or crosscoders on two checkpoints (nothing core depends on these) |
| 12 | Write-up | Paper and talk; every figure rebuilt from data with one command |

Gates are hard stops: a phase does not start until the previous gate passes or its fallback is
written down.

## 10. Compute

About 50 to 100 GPU hours, almost all of it checkpoint evaluation; we start with Pythia 410M while
GPU access is arranged and flag anything likely to exceed the budget. Exposure detection uses the
public infini-gram service and CPU-side search of the preshuffled data.

## 11. Team

| Role | Owns |
| --- | --- |
| Data lead | Retrieval, embedding check, tiers, step mapping |
| Validation lead | Hand validation, matched clean items, paraphrases |
| Evaluation lead | Checkpoint evaluation, compute |
| Statistics lead | Exposure IRT, event study, counterfactual, figures |

## 12. Deliverables

1. **Main figure:** the event study, mean logp change for exposed against matched clean items across
   checkpoints before and after exposure. It must be readable without trusting any fitted model.
2. **Second figure:** observed accuracy over training next to the exposure-removed counterfactual, gap
   shaded.
3. The code, configs and a one-command rebuild of every figure; the project website
   (https://projectx-psi-five.vercel.app/).

## References

- Lesci, P., Meister, C., Hofmann, T., Vlachos, A. and Pimentel, T. (2024). Causal Estimation of
  Memorisation Profiles. ACL. https://aclanthology.org/2024.acl-long.834.pdf
- Biderman, S. et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and
  Scaling. ICML.
- Biderman, S. et al. (2023). Emergent and Predictable Memorization in Large Language Models. NeurIPS.
- Carlini, N. et al. (2023). Quantifying Memorization Across Neural Language Models. ICLR.
- Liu, J. et al. (2024). Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens. COLM.
- Mirzadeh, I. et al. (2024). GSM-Symbolic. arXiv.
- Polo, F. M. et al. (2024). tinyBenchmarks. ICML.
- Schaeffer, R., Miranda, B. and Koyejo, S. (2023). Are Emergent Abilities of Large Language Models a
  Mirage? NeurIPS.
- EleutherAI. Pythia repository and published evaluation results, commit `a19eecb`.
  https://github.com/EleutherAI/pythia

All references except Lesci et al. and the Pythia repository were written from memory and are to be
verified before citing.
