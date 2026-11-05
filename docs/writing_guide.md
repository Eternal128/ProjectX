# Writing guide: how the paper must be written

This guide sets the standard for every write-up of this project: the final paper, the proposal, the
talk and the website's research sections. It is distilled from four ProjectX winning papers and then
raised above them. The goal is a paper that is **better than every one of them**, not merely like them.

## 1. What we studied

| Year | Team | Paper | Theme fit |
| --- | --- | --- | --- |
| 2020 | University of Toronto | *Forecasting Black Sigatoka Infection Risks with Latent Neural ODEs* | Climate: infectious crop disease |
| 2020 | University of Michigan | *The Devil is in the Details: Spatial and Temporal Super-Resolution of Global Climate Models using Deep Learning* | Climate: weather and natural disaster |
| 2020 | Cornell University | *Ride-hailing emissions modeling and reduction through ride demand prediction* | Climate: energy efficiency |
| 2024 | McGill University | *Scavenging Hyena: Distilling Transformers into Long Convolution Models* (arXiv 2401.17574) | Sustainable AI |

The McGill paper is the closest to ours: Pythia 70M, the Pile, lm-evaluation-harness, ARC-Easy, PIQA
and SciQ, all run in 5 hours on one RTX 3090. A winning paper can be built from exactly our toolkit.

## 2. What every winner does

1. **Real stakes in the first paragraph, with a number, tied to the year's theme.** "St. Vincent faced
   a 90% decline in banana production." "Transportation is the largest contributor of US greenhouse gas
   emissions." McGill opens on GPT-4's energy cost and closes on "sustainable AI".
2. **A named gap, then 2 to 4 contribution bullets.** One sentence says what does not yet exist; the
   bullets say exactly what this paper adds.
3. **A known method adapted to a new field with one specific twist,** scoped to modest compute.
   Nobody invents machine learning from scratch. A memorable method name helps (Toronto's "MR. NODE").
4. **Missing data or compute worked around openly,** and the workaround becomes a contribution:
   simulated disease data from a published model (Toronto), a learned degradation model for unpaired
   data (Michigan), three public datasets joined with stated assumptions (Cornell).
5. **Tables against a baseline with standard metrics.** RNN and LSTM; bicubic upsampling; XGBoost and
   the best published model; the teacher model.
6. **Stress tests or extra evaluations:** data dropped at 30 to 90% plus a second region (Toronto),
   7 months of later, unseen data (Cornell), quantitative scores plus a downstream detector plus a blind
   human study with a domain expert (Michigan).
7. **An impact number or a use case:** coal-equivalent emissions saved, days of warning, "two orders
   of magnitude" less computing, a diagram of a climatologist using the tool.
8. **Named limitations, including results that did not work.** Michigan reports that plain upscaling
   beats its model on two metrics; McGill's student model loses badly on SciQ (0.357 vs 0.640) and they
   say so.
9. **Research-paper packaging:** NeurIPS-style layout, 7 to 10 pages, data shown early, an architecture
   or method figure, code links, and in the best case an appendix of settings.

## 3. Where the winners fell short (and we will not)

| Weakness seen | Example | Our standard |
| --- | --- | --- |
| Overclaiming | McGill: "pioneering", "surpasses pre-training" while its own table is mixed | Every claim is backed by a number in a table or figure. Banned words: pioneering, novel (unless shown), revolutionary, groundbreaking, state-of-the-art (unless compared). |
| One run, no uncertainty | McGill and Toronto report single runs; Toronto reports MSE without spread | Every estimate has a 95% interval (bootstrap over items) or a standard error; IRT reports posterior intervals; key numbers are repeated with a second seed or split. |
| Correlation read as cause | Cornell's 10.6% reduction is a counterfactual from its own forecasts, never validated | Causal claims only where the design supports them: the event study with a passed pre-trend test. Anything else is called "associated with". |
| Weak or no baseline for the main claim | Michigan's mAP is 1.39 vs 2.11 on 5% of the data | Our main estimate is compared with a naive baseline (raw before/after) and with standard IRT, and the gap is reported. |
| Hypotheses chosen after the fact | None of the four pre-registered anything | Our hypotheses and thresholds are pre-registered; every deviation is logged in `prereg_changes.md` and reported in the paper. |
| Data quality not measured | Toronto's labels are simulated; Cornell's deadheading is assumed | Hand-validated precision per exposure tier, reported in the paper. |
| Generalization untested | Most use one dataset or one model size | Results across benchmarks, model sizes and standard vs deduplicated training, stratified by when items were seen. |
| Reproducibility partial | "Leave your comment and request access" to a Google Drive (Michigan) | One command per figure, configs with hashes, public data, pinned versions. |
| Thin related work | Two to four citations for the key claim | Position against contamination, memorization and IRT-for-evaluation literature, with Lesci et al. (2024) as the closest work. |

## 4. Our paper, section by section

Target: 9 pages plus references and appendix, NeurIPS style.

1. **Title.** States the finding or the question, plus the object. Short and memorable.
2. **Abstract (150 to 200 words).** Theme sentence; gap; method in one sentence with its name; the
   headline number with its interval; one robustness result; one sentence on what it means.
3. **Introduction (1 page).**
   - Paragraph 1: stakes for "What is progress?", with a real number from the published Pythia curves.
   - Paragraph 2: the gap: static contamination checks cannot say how much of the progress curve is recall.
   - Paragraph 3: our idea in one bolded sentence.
   - Contribution bullets (3, each ending with a result pointer, e.g. "(Section 5, Table 2)").
   - A teaser figure on page 1 or 2 (Figure 1, below).
4. **Related work (half a page).** Benchmark contamination; memorization in Pythia (Biderman et al.;
   Lesci et al.); IRT for model evaluation; emergence and metric choice (Schaeffer et al.).
5. **Setting and data (1 page).** Pythia suite and checkpoints; the Pile and training order; benchmarks
   with the above-chance check (table); exposure detection pipeline and tier precision.
6. **Method (1.5 pages).** The event study (with the effect equation and the pre-trend test); the
   exposure IRT model (equation, priors, fitting order); the exposure share of progress. A method figure.
7. **Results (2.5 pages).** In this order: does exposure cause a jump (event study); how big and how
   dose-dependent (IRT); how much of progress (share); does it depend on ability; does it transfer to
   paraphrases. Each subsection leads with its answer in one sentence, then the evidence.
8. **Robustness (half a page).** Likely tier on each side; deduplicated models; t0 strata; seeds or splits.
9. **Discussion and limitations (half a page).** What it means for reading benchmark curves; every
   limitation named, including missed exposure biasing toward zero and any hypothesis that failed.
10. **Conclusion (short).** The finding in two sentences and what others should do differently.
11. **Appendix.** Configs and settings, hand-validation protocol, full per-benchmark tables, the
    method checks on simulated data, and the pre-registration log.

## 5. Figures and tables

- **Figure 1 (teaser):** one benchmark's accuracy over training, split into seen and unseen items,
  readable without any model.
- **Figure 2 (main):** the event study: change in log-probability for exposed vs matched clean items,
  before and after t0, with 95% bands. Readable without trusting a fitted model.
- **Figure 3:** observed accuracy next to the exposure-removed counterfactual, gap shaded.
- **Figure 4:** method diagram (find in the Pile, map to step, score, compare).
- **Table 1:** benchmarks, items, exposure tiers, tier precision.
- **Table 2:** main estimates with intervals, against the naive baseline and standard IRT.
- **Table 3:** robustness.
- Every figure: axis labels with units, a one-line takeaway caption that states the finding, colour-blind
  safe palette, direct labels, rebuilt by one command from `data/`.

## 6. Style rules

- Lead every section and paragraph with its point; evidence follows.
- One idea per sentence; prefer plain words. Define every term at first use.
- Numbers: give the effect, its interval and the sample size together ("+0.42 nats, 95% CI 0.31 to 0.55,
  n = 412 items").
- Name the theme and connect to it in the abstract, introduction and conclusion.
- Report failed hypotheses as findings, in the same detail as successful ones.
- No claim without a pointer to the table or figure that supports it.
- Cite Lesci et al. (2024) wherever the event-study design is introduced.

## 7. Pre-submission checklist

- [ ] First sentence names the stakes and the theme; a real number in the first paragraph
- [ ] One bolded thesis sentence; 3 contribution bullets with section pointers
- [ ] Headline number with interval in the abstract
- [ ] Every estimate has an interval; main results checked with a second seed or split
- [ ] Main claim compared against a naive baseline and standard IRT
- [ ] Pre-trend test reported for every causal reading
- [ ] Tier precision from hand validation reported
- [ ] Every pre-registration deviation listed
- [ ] Limitations named, including failed hypotheses and the bias toward zero
- [ ] Figures 1 to 4 and Tables 1 to 3 rebuilt from `data/` by one command each
- [ ] Code, configs and data links in the paper; references verified
- [ ] No banned overclaiming words; every claim points to evidence
- [ ] Read by someone outside the team: can they state the finding after the abstract?
