# Project website

Static site for *Seen or Solved?* in `site/`: plain HTML, CSS and JavaScript, no build step.
Everything in `site/` is public. Hosted on Vercel: https://projectx-psi-five.vercel.app/

- `site/index.html`: the study, with interactive charts.
- `site/proposal.html`: the full proposal, rendered from `docs/proposal.md`.
- `site/data/*.json`: the data behind every chart, written by scripts in `src/` (below).

Design: light theme, Clash Grotesk (loaded from Fontshare), greys with a mint accent; every chart is a
working SVG fragment with hover values and a table for screen readers. Chart colours are the dataviz
reference palette's first three slots, validated colour-blind safe for all pairs.

## What the charts show

| Data file | Written by | What it is |
| --- | --- | --- |
| `benchmarks.json` | `python -m evals.pythia_published` | **Real.** EleutherAI's published zero-shot accuracy for the six study models at 27 checkpoints (Pythia repo, pinned commit, Apache-2.0), plus the above-chance check. Not our own per-item evaluation. |
| `methods.json` | `python -m figures.site_methods` | **Simulated.** Our event-study and IRT code run on simulated data with planted effects, to show the estimators recover them. Not results about Pythia. |
| `results.json` | `python -m figures.site_data` | Study results. Every block stays "pending" until `data/results/` has outputs. Run only after the analysis is frozen. |

After editing `docs/proposal.md`, regenerate the page with `python -m figures.site_proposal`
(needs `pip install -e ".[site]"`).

## Preview locally

```bash
python -m http.server -d site 8000   # then open http://localhost:8000
```

Charts read `site/data/*.json` with `fetch`, so open the site through a server, not straight from disk.

## Deploy (Vercel)

`vercel.json` at the repo root serves `site/` as a static site with no install or build step. Pushes to
`main` deploy production; other branches and pull requests get preview deployments. In the Vercel
project settings leave Root Directory empty and the framework preset on "Other".
`npx vercel build` builds the same output locally into `.vercel/output/`.
