# Seen or Solved?

**Decomposing language model progress into memorization and generalization.**
Project X 2026 (UofT AI), Research stream. Theme: *What is Progress?*

Benchmark scores mix two things: recalling items seen in training and getting more capable.
This project measures how much of Pythia's benchmark progress over training is explained by
exposure to benchmark items. It uses two tools:

1. **Exposure detection.** Find which benchmark items appear in the Pile and the training step
   where Pythia first saw each one.
2. **Exposure IRT plus an event study.** An Item Response Theory model with a term for how much
   easier an item gets after exposure, checked by a difference-in-differences event study around
   each exposure step.

The closest prior work is Lesci et al. (2024), *Causal Estimation of Memorisation Profiles* (ACL),
which runs difference-in-differences on Pythia for training sequences. We study benchmark items,
their effect on measured progress, and add the exposure IRT model.

- Full plan, hypotheses and conventions: [`AGENTS.md`](AGENTS.md) (read this first).
- Proposal: [`docs/proposal.md`](docs/proposal.md).
- Changes to hypotheses or thresholds: [`docs/prereg_changes.md`](docs/prereg_changes.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # analysis + tests (CPU only)
pip install -e ".[dev,eval,exposure]"   # add checkpoint evaluation and exposure detection
pytest
```

## Common commands

```bash
# Phase 0 smoke test: one Pythia checkpoint on one benchmark
python -m evals.smoke --model EleutherAI/pythia-410m --revision step143000 --task arc_easy --limit 50

# Compare configs/models.yaml checkpoints against what Hugging Face actually hosts
python -m evals.list_checkpoints --config configs/models.yaml

# Figures (read only from data/)
python -m figures.fig_event_study --config configs/event_study.yaml
python -m figures.fig_counterfactual --config configs/irt.yaml
```

## Layout

```
configs/      YAML configs for every run
data/         raw/, exposure/, evals/, results/  (gitignored bulk data)
src/          exposure/, evals/, irt/, event_study/, figures/, common/
tests/        unit tests, including IRT parameter recovery on simulated data
docs/         proposal, step-mapping notes, pre-registration change log
notebooks/    exploration only
```

## License

MIT, see [`LICENSE`](LICENSE).
