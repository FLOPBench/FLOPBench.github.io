# FLOPBench GitHub Pages

This repository is intended to be published as the GitHub Pages site for the `FLOPBench` account at:

`https://flopbench.github.io/`

Repository layout:

- `FLOPBench/`: full site bundle and source assets
- `FLOPBench_anonymous/`: anonymous bundle that is published to GitHub Pages
- `scripts/`: website-owned extraction scripts for distilling paper-subset LLM data from the already-loaded PostgreSQL databases

GitHub Pages deployment is handled by the workflow in `.github/workflows/deploy-pages.yml`, which uploads only `FLOPBench_anonymous/`.

The reviewer-facing site focuses on the 254 sampled paper kernels. To refresh generated data from the live experiment database and vendored dataset:

```bash
python scripts/export_paper_llm_data.py
python FLOPBench/scripts/build_site_data.py
python FLOPBench/scripts/build_anonymous_site.py
```

`scripts/export_paper_llm_data.py` reads `gpuflops_db` through Python/`psycopg`; it does not restore or mutate dump files.
