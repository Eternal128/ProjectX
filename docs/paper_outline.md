# Paper outline

The skeleton of the final paper, written to `docs/writing_guide.md`. Text in [brackets] is filled in
from results; nothing in brackets may be guessed. Target: 9 pages plus references and appendix,
NeurIPS style.

## Title (pick one before writing)

- *Seen or Solved? How Much of a Language Model's Progress Is Memory*
- *Seen or Solved? Measuring the Share of Benchmark Progress Explained by Exposure in Pythia*
- *When the Model Read the Test: Exposure Effects on Benchmark Progress Across Pythia's Training*

## Method name (pick one)

A short, memorable name for the exposure IRT model plus event study, used consistently:
**XIRT** (eXposure IRT), **SEEN** (Step-level Exposure Effect estimatioN), or **TRACE** (Training-Read
Attribution of Contaminated Evaluations). Decide with the team before the first draft.

## Abstract (150 to 200 words)

1. Theme and stakes: benchmark curves are how the field measures progress; part of a curve's rise may
   be recall of test items the model read during training.
2. Gap: contamination checks say whether an item was seen, not how much of the progress curve that
   explains or when.
3. Method, one sentence, named: we locate each benchmark item in Pythia's training data, date the step
   it was first read, and estimate its effect with an event study and an exposure-aware IRT model.
4. Headline: exposure explains [X% (95% CI a to b)] of Pythia [size]'s accuracy gain on [benchmarks]
   between steps [s1] and [s2].
5. One supporting result: [jump of d nats at first exposure, no pre-trend] / [effect grows with copies].
6. One robustness result: [holds on deduplicated models / paraphrases retain c%].
7. Meaning: what benchmark curves can and cannot tell us about progress.

## 1. Introduction

**Paragraph 1, stakes (draft, real numbers).** Pythia 2.8B's accuracy on SciQ climbs from 21.9% to
82.9% in its first 13,000 training steps, about 9% of training, and adds only 5.3 more points over the
remaining 91% (EleutherAI's published results). Curves like this are how we read progress. But Pythia
learned from the Pile, a public web-scale corpus, and benchmark questions circulate on the web. [How
much of a curve like this is the model recalling questions it read?]

**Paragraph 2, gap.** Static contamination checks ask whether a test item appears in training data.
They cannot say when the model read it, whether reading it changed the model's answer, or how much of
a progress curve such reading adds up to. [Cite contamination work; Lesci et al. (2024) estimate
memorisation of training sequences causally, but not for benchmark items or their effect on measured
progress.]

**Paragraph 3, idea (one bolded sentence).** **Because Pythia's training data and its order are public,
we can date the moment each benchmark question was first read and measure what reading it did to the
model's score.**

**Contributions (3 bullets, each ending with a pointer).**
- [Exposure dating]: we find [N] benchmark items in the Pile and date their first exposure to the
  training step, with hand-validated precision of [p%] (Section 3, Table 1).
- [Causal effect]: an event study around each item's first exposure shows [result], with [no / a]
  pre-trend (Section 5.1, Figure 2).
- [Share of progress]: an exposure-aware IRT model attributes [X%] of measured progress to exposure,
  [growing / not growing] with copy count and [depending / not depending] on ability (Section 5.2 to
  5.4, Figure 3, Table 2).

**Figure 1 (teaser, page 1 or 2).** Accuracy over training for seen vs unseen items on one benchmark.

## 2. Related work (half a page)

Benchmark contamination and its detection; memorization in Pythia (Biderman et al. 2023; Lesci et al.
2024); item response theory for evaluating language models (Polo et al. 2024, tinyBenchmarks); metric
choice and apparent emergence (Schaeffer et al. 2023). One sentence per thread on how we differ.

## 3. Setting and data (1 page)

- Pythia: sizes, standard vs deduplicated, 154 checkpoints, 1,024 × 2,048 tokens per step, fixed order.
- Benchmarks: which, why (public before the Pile, above chance), with **Table 1** (items, chance,
  final accuracy, tiers, tier precision).
- Exposure detection: retrieval, semantic check, tiers, step mapping, timing filter; hand validation.

## 4. Method (1.5 pages)

- Event study: matching, the effect equation, the pre-trend test, bootstrap intervals. **Figure 4**:
  method diagram.
- Exposure IRT: the model equation, priors, fitting order with recovery tests (appendix).
- Exposure share of progress: the counterfactual and M.

## 5. Results (2.5 pages; each subsection opens with its one-sentence answer)

1. Does reading a question cause a jump? **Figure 2** (event study), pre-trend result.
2. How large, and does it grow with copies? **Table 2** (IRT estimates vs naive baseline and standard IRT).
3. How much of progress is exposure? **Figure 3** (observed vs exposure-removed curve, gap shaded).
4. Does the effect depend on ability? (H4)
5. Does it transfer to paraphrases? (H5)

## 6. Robustness (half a page)

**Table 3**: likely tier on each side; deduplicated models; t0 strata; second seed or split.

## 7. Discussion and limitations (half a page)

What the share means for reading benchmark curves as progress. Limitations, each named: missed exposure
biases toward zero; exact-match detection misses paraphrased copies; model sizes studied; any failed
hypothesis, reported as a finding.

## 8. Conclusion

Two sentences: the finding, and what evaluators should do differently.

## Appendix

A. Configs and settings (from `configs/`, with hashes). B. Hand-validation protocol and agreement.
C. Per-benchmark tables. D. Method checks on simulated data (recovery of planted effects).
E. Pre-registration log and every deviation.
