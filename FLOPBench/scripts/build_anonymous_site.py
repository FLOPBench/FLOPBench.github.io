#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = DOCS_DIR.parent
OUT_DIR = REPO_DIR / "docs-anon"
ANON_REPO_URL = "https://github.com/FLOPBench/FLOPBench.github.io"
SITE_DATA_PREFIX = "window.GPU_FLOWBENCH_DATA = "


def copy_runtime_tree() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCS_DIR / ".nojekyll", OUT_DIR / ".nojekyll")
    shutil.copytree(DOCS_DIR / "assets", OUT_DIR / "assets", ignore=shutil.ignore_patterns("team"))
    shutil.copytree(DOCS_DIR / "data", OUT_DIR / "data")
    shutil.copytree(DOCS_DIR / "downloads", OUT_DIR / "downloads")


def scrub_html() -> None:
    html = (DOCS_DIR / "index.html").read_text()
    html = html.replace('          <a href="#citation">Citation</a>\n', "")
    html = html.replace('          <a href="#team">Team</a>\n', "")

    for section_id in ("citation", "team"):
        html = re.sub(
            rf"\n\s*<section class=\"section\" id=\"{section_id}\">.*?</section>",
            "",
            html,
            flags=re.DOTALL,
        )

    html = html.replace(
        "https://github.com/FLOPBench/FLOPBench.github.io",
        ANON_REPO_URL,
    )
    (OUT_DIR / "index.html").write_text(html)


def scrub_site_data() -> None:
    site_data_path = OUT_DIR / "data" / "site-data.js"
    text = site_data_path.read_text().strip()
    if not text.startswith(SITE_DATA_PREFIX) or not text.endswith(";"):
        raise ValueError("Unexpected site-data.js format.")

    payload = json.loads(text[len(SITE_DATA_PREFIX):-1])
    payload.get("meta", {}).pop("paper", None)
    payload.get("meta", {}).pop("team", None)
    site_data_path.write_text(f"{SITE_DATA_PREFIX}{json.dumps(payload, separators=(',', ':'))};")

    metadata_path = OUT_DIR / "data" / "site-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("paper", None)
    metadata.pop("team", None)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def scrub_index_script() -> None:
    script_path = OUT_DIR / "assets" / "index.js"
    script = script_path.read_text()

    patterns = [
        r"\n  const citationMetaNode = document\.getElementById\(\"citationMeta\"\);",
        r"\n  const citationBibtexNode = document\.getElementById\(\"citationBibtex\"\);",
        r"\n  const teamGridNode = document\.getElementById\(\"teamGrid\"\);",
        r"\n  function renderCitation\(\) \{.*?\n  \}",
        r"\n  function renderTeam\(team\) \{.*?\n  \}",
        r"\n    renderCitation\(\);",
        r"\n    renderTeam\(meta\.team\);",
    ]

    for pattern in patterns:
        script = re.sub(pattern, "", script, flags=re.DOTALL)

    script_path.write_text(script)


def main() -> None:
    copy_runtime_tree()
    scrub_html()
    scrub_site_data()
    scrub_index_script()
    print(f"Wrote anonymous site bundle to {OUT_DIR}")


if __name__ == "__main__":
    main()
