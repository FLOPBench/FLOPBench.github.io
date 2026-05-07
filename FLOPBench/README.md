# gpuFLOPBench site

Static single-page benchmark site.

## Rebuild site data

```bash
python3 scripts/export_paper_llm_data.py
python3 FLOPBench/scripts/build_site_data.py
python3 FLOPBench/scripts/build_anonymous_site.py
```

Run the first command from the repository root. It reads the already-loaded PostgreSQL `gpuflops_db` through `psycopg` and writes the 254-kernel paper subset plus LLM prompt/response shards used by the site.

## Local preview

```bash
cd FLOPBench_anonymous
python3 -m http.server 4177
```

Open `http://127.0.0.1:4177/`.

## Files

- `index.html`: single-page benchmark presentation with charts, explorer, and downloads
- `data/`: generated JSON/CSV for the site, including LLM prediction index and per-program response shards
- `source-data/`: vendored build inputs, including `gpuFLOPBench.json.gz` and the generated `paper-kernel-subset.json`
- `assets/`: shared styles and page scripts
- `scripts/build_site_data.py`: data builder from the vendored gpuFLOPBench dataset
