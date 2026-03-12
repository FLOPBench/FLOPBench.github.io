# gpuFLOPBench site

Static single-page benchmark site.

## Rebuild site data

```bash
python3 FLOPBench/scripts/build_site_data.py
python3 FLOPBench/scripts/build_anonymous_site.py
```

## Local preview

```bash
cd FLOPBench_anonymous
python3 -m http.server 4177
```

Open `http://127.0.0.1:4177/`.

## Files

- `index.html`: single-page benchmark presentation with charts, explorer, and downloads
- `data/`: generated JSON/CSV for the site
- `source-data/`: vendored build inputs, including `gpuFLOPBench.json.gz`
- `assets/`: shared styles and page scripts
- `scripts/build_site_data.py`: data builder from the vendored gpuFLOPBench dataset
