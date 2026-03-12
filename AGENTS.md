# FLOPBench Repo Guide

## Purpose

This repo backs the `FLOPBench/FLOPBench.github.io` GitHub Pages site.
It contains:

- A full source site in `FLOPBench/`
- A generated anonymous site in `FLOPBench_anonymous/`
- A Pages workflow in `.github/workflows/`

The site is a research-project website for gpuFLOPBench. It combines:

- Static project framing and paper-oriented copy
- Generated benchmark summaries and kernel tables
- A filterable roofline atlas
- A kernel/source explorer backed by generated data artifacts

## Top-Level Layout

- `FLOPBench/`
  - Source of truth for the website
  - Contains HTML, CSS, JS, vendored source data, build scripts, and generated site data
- `FLOPBench_anonymous/`
  - Generated publishable anonymous bundle
  - This is what GitHub Pages deploys
  - Do not treat this directory as hand-authored source; regenerate it from `FLOPBench/`
- `.github/workflows/deploy-pages.yml`
  - Deploys the anonymous site to `https://flopbench.github.io/`

## Important Subdirectories

Inside `FLOPBench/`:

- `assets/`
  - Frontend JS/CSS and static runtime assets
- `data/`
  - Generated site payloads
  - Includes `site-data.js`, `site-metadata.json`, summary CSV/JSON files, and `explorer-programs/*.json`
- `downloads/`
  - Downloadable raw artifacts exposed on the site
  - Includes `gpuFLOPBench.json.gz`
- `scripts/`
  - Build scripts
- `source-data/`
  - Vendored source inputs used to regenerate site data

## Anonymous vs Non-Anonymous

`FLOPBench/` is the full site. It may contain:

- Citation metadata
- Team metadata and team images
- Full project framing

`FLOPBench_anonymous/` is the scrubbed variant. The anonymous build:

- Removes the Citation and Team sections from HTML
- Removes `paper` and `team` metadata from `site-data.js` and `site-metadata.json`
- Removes citation/team rendering code from the frontend bundle
- Excludes `assets/team/`

Design rule:

- Make content changes in `FLOPBench/`
- Regenerate `FLOPBench_anonymous/`
- Commit both when the anonymous output changes

## Data Source

The site’s benchmark source of truth is the vendored compressed dataset:

- `FLOPBench/source-data/gpuFLOPBench.json.gz`

This vendored file came from the updated gpuFLOPBench dataset originally located at:

- `/gpuFLOPBench-updated/dataset-creation/gpuFLOPBench.json`

The repo stores the compressed `.json.gz` variant because the raw JSON is too large for a normal Git commit.

Additional vendored metadata comes from:

- `FLOPBench/source-data/HeCBench/benchmarks.yaml`
- `FLOPBench/source-data/HeCBench/README.md`

## Generated Outputs

`FLOPBench/scripts/build_site_data.py` reads the vendored dataset and generates:

- `FLOPBench/data/site-data.js`
- `FLOPBench/data/site-metadata.json`
- `FLOPBench/data/kernel-performance.{json,csv}`
- `FLOPBench/data/source-performance.{json,csv}`
- `FLOPBench/data/explorer-programs/*.json`
- `FLOPBench/downloads/gpuFLOPBench.json.gz`

`explorer-programs/*.json` are sharded per-program payloads used by the Source Explorer for SASS/IMIX lookup without loading the full dataset in the browser.

## Build Workflow

If you changed frontend copy, layout, or JS/CSS only:

```bash
python /FLOPBench/FLOPBench/scripts/build_anonymous_site.py
```

If you changed the source dataset, data builder, or anything that affects generated site payloads:

```bash
python /FLOPBench/FLOPBench/scripts/build_site_data.py
python /FLOPBench/FLOPBench/scripts/build_anonymous_site.py
```

Useful verification commands:

```bash
node --check /FLOPBench/FLOPBench/assets/index.js
node --check /FLOPBench/FLOPBench_anonymous/assets/index.js
git -C /FLOPBench status --short
```

## Deployment

GitHub Pages is driven from this repo on `main`.

Deployment flow:

1. Make edits under `FLOPBench/`
2. Regenerate `FLOPBench_anonymous/`
3. Commit both source and generated anonymous changes
4. Push `main`
5. The Pages workflow publishes the anonymous site

Typical commands:

```bash
git -C /FLOPBench add AGENTS.md FLOPBench FLOPBench_anonymous
git -C /FLOPBench commit -m "Describe change"
git -C /FLOPBench push origin main
gh run list --repo FLOPBench/FLOPBench.github.io --limit 5
gh run watch <run-id> --repo FLOPBench/FLOPBench.github.io --exit-status
```

## Editing Rules

- Prefer editing `FLOPBench/`, not `FLOPBench_anonymous/`
- Treat `FLOPBench_anonymous/` as generated output
- If the anonymous bundle differs after regeneration, commit it too
- If the site data changes, rebuild before regenerating the anonymous bundle
- Keep anonymous-safe content free of names or identifying metadata unless it is intentionally scrubbed later
