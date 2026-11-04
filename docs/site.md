# Project website

Static site for *Seen or Solved?* in `site/`: no build step, plain HTML, CSS and JavaScript.
Everything in `site/` is public. It is hosted on Vercel: https://projectx-psi-five.vercel.app/

## Preview locally

```bash
python -m http.server -d site 8000   # then open http://localhost:8000
```

Opening `index.html` straight from disk also works, but results then always show as pending
(browsers block reading `data/results.json` from `file://`).

## Deploy (Vercel)

The repository is connected to a Vercel project. `vercel.json` at the repo root tells Vercel this is a
static site served from `site/`, with no install or build step (so Vercel does not treat
`pyproject.toml` as a Python app).

- Every push to `main` deploys production at the URL above.
- Every other branch and every pull request gets its own preview deployment; Vercel posts the link
  on the pull request.
- In the Vercel project settings, leave **Root Directory** empty (the repository root) and the
  framework preset as **Other**; `vercel.json` overrides the build settings.

To check a change before pushing: `npx vercel build` builds the same output into `.vercel/output/`.

## Results

Every result on the page is **pending** until `site/data/results.json` says otherwise. That file is
written from `data/` and `configs/` only:

```bash
python -m figures.fig_event_study --config configs/event_study.yaml   # also writes an .svg
python -m figures.fig_counterfactual --config configs/irt.yaml
python -m figures.site_data                                           # writes site/data/results.json
```

Run `figures.site_data` only after the analysis code and thresholds are frozen (AGENTS.md): it
publishes whatever is in `data/results/`. The timing explorer reads the checkpoint list and timing
filter from the same file, so it follows `configs/models.yaml` and `configs/exposure.yaml`.

## Pictures

All pictures are flat drawings made for the site by `src/figures/site_plates.py`
(`python src/figures/site_plates.py` redraws them into `site/assets/img/`). Archival pictures would be
stronger and could not be fetched when the site was built (the build environment's network blocked
the open image collections). Good candidates, all to be checked for licence and credited in the footer:

- the Aegeus kylix (Antikensammlung Berlin), the only ancient image of the Pythia at work, for the hero;
- John Collier, *Priestess of Delphi* (1891, public domain), for the tall intro picture;
- photographs of the Temple of Apollo and the Delphic tripod for the hero and intro;
- an archival photograph of a paper archive or library stack for the "When Pythia read it" picture.

## Sample content

The example questions on the benchmark cards are written for this page in each benchmark's format;
they are not items from the benchmarks. Swap in real items (with the dataset licence noted) if wanted.
