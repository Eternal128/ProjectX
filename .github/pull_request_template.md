## What changed and why

## Checklist

- [ ] Driven by a config in `configs/`; outputs record config hash and git commit
- [ ] Tests added or updated (`pytest` passes)
- [ ] No result looked at before analysis code and thresholds were fixed
- [ ] Any change to hypotheses, thresholds or filters is logged in `docs/prereg_changes.md`
- [ ] IRT uses `y` only; event study uses `logp` only
- [ ] Changes to `src/irt/` or `src/event_study/`: reviewed by the statistics lead
- [ ] Compute estimate stays within the 50–100 GPU-hour budget (flag if not)
