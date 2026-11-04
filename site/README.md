# Project website

Static site for *Seen or Solved?*: no build step, plain HTML, CSS and JavaScript.
Deployed to GitHub Pages by `.github/workflows/site.yml` on every push to `main` that touches `site/`.

## Preview locally

```bash
python -m http.server -d site 8000   # then open http://localhost:8000
```

Opening `index.html` straight from disk also works, but results then always show as pending
(browsers block reading `data/results.json` from `file://`).

## Deploy

One-time: on GitHub, **Settings → Pages → Source: GitHub Actions**. After that, merging to `main`
publishes the site. The workflow can also be run by hand from the Actions tab.

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

All pictures are flat drawings made for the site by `site/tools/plates.py`
(`python site/tools/plates.py` redraws them into `site/assets/img/`). Archival pictures would be
stronger and could not be fetched when the site was built (the build environment's network blocked
the open image collections). Good candidates, all to be checked for licence and credited in the footer:

- the Aegeus kylix (Antikensammlung Berlin), the only ancient image of the Pythia at work, for the hero;
- John Collier, *Priestess of Delphi* (1891, public domain), for the tall intro picture;
- photographs of the Temple of Apollo and the Delphic tripod for the hero and intro;
- an archival photograph of a paper archive or library stack for the "When Pythia read it" picture.

## Sample content

The example questions on the benchmark cards are written for this page in each benchmark's format;
they are not items from the benchmarks. Swap in real items (with the dataset licence noted) if wanted.
