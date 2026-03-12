#!/usr/bin/env python3
"""Build site datasets from the vendored gpuFLOPBench benchmark JSON."""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import yaml


SITE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SITE_DIR.parent
DATA_DIR = SITE_DIR / "data"
DOWNLOADS_DIR = SITE_DIR / "downloads"
SOURCE_DATA_DIR = SITE_DIR / "source-data"
VENDORED_DATASET_GZ = SOURCE_DATA_DIR / "gpuFLOPBench.json.gz"
BENCHMARKS_YAML = SOURCE_DATA_DIR / "HeCBench" / "benchmarks.yaml"
HECBENCH_README = SOURCE_DATA_DIR / "HeCBench" / "README.md"

DEVICE_FULL_NAME_MAP = {
    "3080": "NVIDIA GeForce RTX 3080",
    "A10": "NVIDIA A10",
    "A100": "NVIDIA A100-SXM4-40GB",
    "H100": "NVIDIA H100 80GB HBM3",
}

GPU_SPECS = {
    "3080": {
        "label": "RTX 3080 (PCIe)",
        "architecture": "Ampere (GA102)",
        "compute_capability": "8.6",
        "memory_bandwidth_gbps": 760,
        "peak_tflops": {"fp16": 30.55, "fp32": 30.55, "fp64": 0.477},
    },
    "A10": {
        "label": "A10 (PCIe)",
        "architecture": "Ampere (GA102)",
        "compute_capability": "8.6",
        "memory_bandwidth_gbps": 600,
        "peak_tflops": {"fp16": 15.62, "fp32": 15.62, "fp64": 0.244},
    },
    "A100": {
        "label": "A100 (SXM4)",
        "architecture": "Ampere (GA100)",
        "compute_capability": "8.0",
        "memory_bandwidth_gbps": 1555,
        "peak_tflops": {"fp16": 77.97, "fp32": 19.49, "fp64": 9.75},
    },
    "H100": {
        "label": "H100 (SXM5)",
        "architecture": "Hopper (GH100)",
        "compute_capability": "9.0",
        "memory_bandwidth_gbps": 3360,
        "peak_tflops": {"fp16": 133.82, "fp32": 66.91, "fp64": 33.45},
    },
}

PAPER_BIBTEX = """@inproceedings{boletCanLargeLanguage2025a,
  title = {Can {{Large Language Models Predict Parallel Code Performance}}?},
  booktitle = {Proceedings of the 34th {{International Symposium}} on {{High-Performance Parallel}} and {{Distributed Computing}}},
  author = {Bolet, Gregory and Georgakoudis, Giorgis and Menon, Harshitha and Parasyris, Konstantinos and Hasabnis, Niranjan and Estes, Hayden and Cameron, Kirk and Oren, Gal},
  date = {2025-09-09},
  series = {{{HPDC}} '25},
  pages = {1--6},
  publisher = {Association for Computing Machinery},
  location = {New York, NY, USA},
  doi = {10.1145/3731545.3743645},
  url = {https://dl.acm.org/doi/10.1145/3731545.3743645},
  urldate = {2026-02-21},
  isbn = {979-8-4007-1869-4},
  keywords = {llms4PerfPrediction}
}"""

TEAM_MEMBERS = [
    {
        "name": "Gregory Bolet",
        "affiliation": "Virginia Tech",
        "profile_url": "https://people.cs.vt.edu/gbolet/",
        "image_path": "./assets/team/gregory-bolet.jpg",
    },
    {
        "name": "Giorgis Georgakoudis",
        "affiliation": "Lawrence Livermore National Laboratory",
        "profile_url": "https://people.llnl.gov/georgakoudis1",
        "image_path": "./assets/team/giorgis-georgakoudis.png",
    },
    {
        "name": "Harshitha Menon",
        "affiliation": "Lawrence Livermore National Laboratory",
        "profile_url": "https://www.harshithamenon.com/",
        "image_path": "./assets/team/harshitha-menon.jpg",
    },
    {
        "name": "Konstantinos Parasyris",
        "affiliation": "Lawrence Livermore National Laboratory",
        "profile_url": "https://www.ashes-hpc.org/2025/program.html",
        "image_path": "./assets/team/konstantinos-parasyris.png",
    },
    {
        "name": "Niranjan Hasabnis",
        "affiliation": "Code Metal",
        "profile_url": "https://www.codemetal.ai/about",
        "image_path": "./assets/team/niranjan-hasabnis.jpg",
    },
    {
        "name": "Hayden Estes",
        "affiliation": "Virginia Tech",
        "profile_url": "https://www.linkedin.com/in/haydenvestes/",
        "image_path": "./assets/team/hayden-estes.svg",
    },
    {
        "name": "Kirk Cameron",
        "affiliation": "Virginia Tech",
        "profile_url": "https://website.cs.vt.edu/people/faculty/kirk-cameron.html",
        "image_path": "./assets/team/kirk-cameron.jpg",
    },
    {
        "name": "Gal Oren",
        "affiliation": "Stanford University",
        "profile_url": "https://profiles.stanford.edu/galoren",
        "image_path": "./assets/team/gal-oren.jpg",
    },
]


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "unknown"


def maybe_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def finite_median(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None and math.isfinite(value)]
    return median(usable) if usable else None


def normalize_omp_symbol(symbol: str) -> str:
    match = re.search(r"([A-Za-z0-9]+_l\d+)$", symbol or "")
    return match.group(1) if match else (symbol or "unknown")


def simplify_demangled(name: str) -> str:
    simple = (name or "").strip()
    if "(" in simple:
        simple = simple.split("(", 1)[0].strip()
    if " " in simple:
        simple = simple.split()[-1]
    if "<" in simple:
        simple = simple.split("<", 1)[0]
    if "::" in simple:
        simple = simple.split("::")[-1]
    return simple or "unknown"


def normalize_display_kernel(model_type: str, kernel_symbol: str, demangled_name: str) -> str:
    if model_type == "omp":
        return normalize_omp_symbol(kernel_symbol)
    return simplify_demangled(demangled_name or kernel_symbol)


def dominant_precision(hp_flop: float, sp_flop: float, dp_flop: float) -> str:
    counts = {"fp64": dp_flop, "fp32": sp_flop, "fp16": hp_flop}
    if max(counts.values()) <= 0:
        return "int-only"
    return max(counts.items(), key=lambda item: item[1])[0]


def benchmark_name_from_source(source: str) -> str:
    return re.sub(r"-(cuda|hip|omp|sycl)$", "", source)


def model_type_from_source(source: str) -> str:
    return source.rsplit("-", 1)[-1]


def parse_hecbench_categories() -> tuple[dict[str, str], list[str]]:
    text = HECBENCH_README.read_text()
    category_map: dict[str, str] = {}
    category_order: list[str] = []
    in_section = False
    current_category: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# Benchmark categories"):
            in_section = True
            continue
        if in_section and line.startswith("# Run a benchmark"):
            break
        if not in_section:
            continue
        if line.startswith("### "):
            current_category = line.replace("### ", "", 1).strip()
            category_order.append(current_category)
            continue
        if current_category and line.startswith("    "):
            for benchmark in [item.strip() for item in line.strip().split(",")]:
                if benchmark:
                    category_map[benchmark] = current_category

    return category_map, category_order


def load_metadata() -> dict:
    with BENCHMARKS_YAML.open() as handle:
        return yaml.safe_load(handle)


def category_for_benchmark(benchmark: str, metadata: dict, readme_categories: dict[str, str]) -> str:
    if benchmark in readme_categories:
        return readme_categories[benchmark]
    details = metadata.get(benchmark, {})
    return (details.get("categories") or ["uncategorized"])[0]


def load_dataset() -> dict:
    with gzip.open(VENDORED_DATASET_GZ, "rt") as handle:
        return json.load(handle)


def build_inventory(metadata: dict, profiled_sources: set[str], category_map: dict[str, str]) -> dict:
    available_by_model: dict[str, int] = {}
    available_by_category: dict[str, int] = {}

    for benchmark, details in metadata.items():
        for model in details.get("models", []):
            available_by_model[model] = available_by_model.get(model, 0) + 1
        category = category_for_benchmark(benchmark, metadata, category_map)
        available_by_category[category] = available_by_category.get(category, 0) + 1

    profiled_benchmarks = {benchmark_name_from_source(source) for source in profiled_sources}
    categorized_profiled = {name for name in profiled_benchmarks if name in category_map}

    return {
        "totals": {
            "benchmarks_yaml": len(metadata),
            "profiled_benchmarks": len(profiled_benchmarks),
            "profiled_sources": len(profiled_sources),
            "categorized_profiled_benchmarks": len(categorized_profiled),
        },
        "models_available": [
            {"model": model, "count": count}
            for model, count in sorted(available_by_model.items())
        ],
        "categories_available": [
            {"category": name, "count": count}
            for name, count in sorted(
                available_by_category.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "category_source": "source-data/HeCBench/README.md",
    }


def build_audit(category_map: dict[str, str]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": "source-data/gpuFLOPBench.json.gz",
        "category_source": "source-data/HeCBench/README.md",
        "categorized_benchmarks": len(category_map),
    }


def dense_rank_desc(rows: list[dict], group_keys: tuple[str, ...], metric_key: str) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: row[metric_key], reverse=True)
        rank = 0
        last_value: float | None = None
        for row in group_rows:
            value = row[metric_key]
            if last_value is None or value != last_value:
                rank += 1
                last_value = value
            row["coverage_rank"] = float(rank)


def build_perf_data(dataset: dict, metadata: dict, category_map: dict[str, str]) -> tuple[list[dict], list[dict]]:
    kernel_rows: list[dict] = []
    source_accumulator: dict[tuple[str, str, str, str, str], dict] = defaultdict(
        lambda: {
            "kernels": set(),
            "peak_performance_tflops": [],
            "median_performance_tflops": [],
            "median_arithmetic_intensity": [],
            "median_xtime_ns": [],
        }
    )

    for source, payload in dataset.items():
        benchmark = benchmark_name_from_source(source)
        model_type = model_type_from_source(source)
        category = category_for_benchmark(benchmark, metadata, category_map)
        exe_args = payload.get("exeArgs", "")

        for kernel_symbol, kernel_payload in payload.get("kernels", {}).items():
            kernel_name = normalize_display_kernel(
                model_type,
                kernel_symbol,
                str(kernel_payload.get("demangledName", "")),
            )
            block_size = kernel_payload.get("blockSz")
            grid_size = kernel_payload.get("gridSz")

            for device, metrics in kernel_payload.get("metrics", {}).items():
                hp_flop = maybe_float(metrics.get("HP_FLOP")) or 0.0
                sp_flop = maybe_float(metrics.get("SP_FLOP")) or 0.0
                dp_flop = maybe_float(metrics.get("DP_FLOP")) or 0.0
                bytes_read = maybe_float(metrics.get("bytesRead")) or 0.0
                bytes_written = maybe_float(metrics.get("bytesWritten")) or 0.0
                bytes_total = bytes_read + bytes_written
                xtime_ns = maybe_float(metrics.get("xtime_ns"))
                float_flops = hp_flop + sp_flop + dp_flop
                performance_tflops = (
                    float_flops / (xtime_ns * 1_000.0)
                    if xtime_ns is not None and xtime_ns > 0
                    else None
                )
                arithmetic_intensity = (
                    float_flops / bytes_total if bytes_total > 0 else None
                )

                kernel_row = {
                    "kernel_row_id": slug(
                        f"{device}-{model_type}-{source}-{kernel_name}-{block_size}-{grid_size}-{exe_args}"
                    ),
                    "source": source,
                    "benchmark": benchmark,
                    "category": category,
                    "model_type": model_type,
                    "device": device,
                    "kernel": kernel_name,
                    "block_size": block_size,
                    "grid_size": grid_size,
                    "exe_args": exe_args,
                    "SP_FLOP": round(sp_flop, 2),
                    "DP_FLOP": round(dp_flop, 2),
                    "HP_FLOP": round(hp_flop, 2),
                    "int_ops": 0.0,
                    "float_flops": round(float_flops, 2),
                    "bytes_total": round(bytes_total, 2),
                    "xtime_ns": round(xtime_ns, 2) if xtime_ns is not None else None,
                    "arithmetic_intensity": round(arithmetic_intensity, 6)
                    if arithmetic_intensity is not None
                    else None,
                    "performance_tflops": round(performance_tflops, 6)
                    if performance_tflops is not None
                    else None,
                    "dominant_precision": dominant_precision(hp_flop, sp_flop, dp_flop),
                }
                kernel_rows.append(kernel_row)

                source_key = (source, benchmark, category, model_type, device)
                source_accumulator[source_key]["kernels"].add(kernel_name)
                source_accumulator[source_key]["peak_performance_tflops"].append(
                    performance_tflops if performance_tflops is not None else 0.0
                )
                source_accumulator[source_key]["median_performance_tflops"].append(
                    performance_tflops if performance_tflops is not None else 0.0
                )
                source_accumulator[source_key]["median_arithmetic_intensity"].append(
                    arithmetic_intensity if arithmetic_intensity is not None else 0.0
                )
                source_accumulator[source_key]["median_xtime_ns"].append(xtime_ns)

    source_rows: list[dict] = []
    for (source, benchmark, category, model_type, device), values in source_accumulator.items():
        source_rows.append(
            {
                "source": source,
                "benchmark": benchmark,
                "category": category,
                "model_type": model_type,
                "device": device,
                "kernel_count": len(values["kernels"]),
                "peak_performance_tflops": round(max(values["peak_performance_tflops"]), 6),
                "median_performance_tflops": round(
                    finite_median(values["median_performance_tflops"]) or 0.0,
                    6,
                ),
                "median_arithmetic_intensity": round(
                    finite_median(values["median_arithmetic_intensity"]) or 0.0,
                    6,
                ),
                "median_xtime_ns": round(
                    finite_median(values["median_xtime_ns"]) or 0.0,
                    6,
                ),
                "coverage_rank": 0.0,
            }
        )

    dense_rank_desc(source_rows, ("device", "model_type"), "peak_performance_tflops")
    return kernel_rows, source_rows


def build_device_summary(kernel_rows: list[dict], source_rows: list[dict]) -> list[dict]:
    device_summary: list[dict] = []
    for device in sorted({row["device"] for row in kernel_rows}):
        kernel_subset = [row for row in kernel_rows if row["device"] == device]
        source_subset = [row for row in source_rows if row["device"] == device]
        specs = GPU_SPECS.get(device, {})
        models = Counter(row["model_type"] for row in source_subset)
        positive_perf = [
            row["performance_tflops"]
            for row in kernel_subset
            if row["performance_tflops"] is not None and row["performance_tflops"] > 0
        ]
        positive_ai = [
            row["arithmetic_intensity"]
            for row in kernel_subset
            if row["arithmetic_intensity"] is not None and row["arithmetic_intensity"] > 0
        ]

        device_summary.append(
            {
                "device": device,
                "label": specs.get("label", device),
                "full_name": DEVICE_FULL_NAME_MAP.get(device, device),
                "architecture": specs.get("architecture", "unknown"),
                "compute_capability": specs.get("compute_capability", "unknown"),
                "memory_bandwidth_gbps": specs.get("memory_bandwidth_gbps"),
                "peak_fp16_tflops": specs.get("peak_tflops", {}).get("fp16"),
                "peak_fp32_tflops": specs.get("peak_tflops", {}).get("fp32"),
                "peak_fp64_tflops": specs.get("peak_tflops", {}).get("fp64"),
                "rows": len(kernel_subset),
                "benchmarks": len({row["benchmark"] for row in source_subset}),
                "sources": len({row["source"] for row in source_subset}),
                "kernels": len({row["kernel"] for row in kernel_subset}),
                "models": [
                    {"model": model, "sources": count}
                    for model, count in sorted(models.items(), key=lambda item: item[0])
                ],
                "median_performance_tflops": finite_median(positive_perf),
                "p95_performance_tflops": (
                    round(sorted(positive_perf)[max(0, math.ceil(len(positive_perf) * 0.95) - 1)], 6)
                    if positive_perf
                    else None
                ),
                "median_arithmetic_intensity": finite_median(positive_ai),
                "median_xtime_ns": finite_median([row["xtime_ns"] for row in kernel_subset]),
            }
        )
    return device_summary


def build_model_matrix(metadata: dict, source_rows: list[dict]) -> list[dict]:
    available_by_model: dict[str, int] = {}
    for details in metadata.values():
        for model in details.get("models", []):
            available_by_model[model] = available_by_model.get(model, 0) + 1

    profiled_by_model = Counter(row["model_type"] for row in source_rows)
    result = []
    for model, available in sorted(available_by_model.items()):
        profiled = profiled_by_model.get(model, 0)
        result.append(
            {
                "model": model,
                "available": available,
                "profiled": profiled,
                "profiled_ratio": round(profiled / available, 4) if available else 0.0,
            }
        )
    return result


def build_category_profiled(source_rows: list[dict], category_order: list[str]) -> list[dict]:
    counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in source_rows:
        counts[(row["category"], row["model_type"])].add(row["source"])

    totals: dict[str, int] = defaultdict(int)
    for (category, _model_type), sources in counts.items():
        totals[category] += len(sources)

    order_index = {name: index for index, name in enumerate(category_order)}
    rows = [
        {
            "category": category,
            "model_type": model_type,
            "profiled_sources": len(sources),
            "_sort_total": totals[category],
            "_sort_order": order_index.get(category, len(order_index) + 1),
        }
        for (category, model_type), sources in counts.items()
    ]
    rows.sort(
        key=lambda row: (
            -row["_sort_total"],
            row["_sort_order"],
            row["category"],
            row["model_type"],
        )
    )
    for row in rows:
        row.pop("_sort_total")
        row.pop("_sort_order")
    return rows


def build_top_lists(source_rows: list[dict]) -> dict:
    perf_positive = sorted(
        [row for row in source_rows if row["peak_performance_tflops"] > 0],
        key=lambda row: row["peak_performance_tflops"],
        reverse=True,
    )
    ai_positive = sorted(
        [row for row in source_rows if row["median_arithmetic_intensity"] > 0],
        key=lambda row: row["median_arithmetic_intensity"],
        reverse=True,
    )
    fields = [
        "source",
        "benchmark",
        "category",
        "model_type",
        "device",
        "kernel_count",
        "peak_performance_tflops",
        "median_arithmetic_intensity",
        "median_xtime_ns",
    ]
    return {
        "performance_sources": [{field: row[field] for field in fields} for row in perf_positive[:20]],
        "ai_dense_sources": [{field: row[field] for field in fields} for row in ai_positive[:20]],
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_js(path: Path, variable_name: str, payload: dict | list) -> None:
    compact = json.dumps(payload, separators=(",", ":"))
    path.write_text(f"window.{variable_name} = {compact};\n")


def reset_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not VENDORED_DATASET_GZ.exists():
        raise FileNotFoundError(f"Vendored dataset is missing: {VENDORED_DATASET_GZ}")
    if not BENCHMARKS_YAML.exists():
        raise FileNotFoundError(f"Vendored metadata is missing: {BENCHMARKS_YAML}")
    if not HECBENCH_README.exists():
        raise FileNotFoundError(f"Vendored README is missing: {HECBENCH_README}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reset_directory(DOWNLOADS_DIR)

    metadata = load_metadata()
    category_map, category_order = parse_hecbench_categories()
    dataset = load_dataset()
    kernel_rows, source_rows = build_perf_data(dataset, metadata, category_map)

    inventory = build_inventory(metadata, set(dataset.keys()), category_map)
    audit = build_audit(category_map)
    device_summary = build_device_summary(kernel_rows, source_rows)
    model_matrix = build_model_matrix(metadata, source_rows)
    category_profiled = build_category_profiled(source_rows, category_order)
    top_lists = build_top_lists(source_rows)

    hero = {
        "headline_metrics": [
            {"label": "benchmark entries", "value": inventory["totals"]["benchmarks_yaml"]},
            {"label": "GPUs covered", "value": len(device_summary)},
            {"label": "profiled binaries", "value": inventory["totals"]["profiled_sources"]},
            {"label": "kernel-device rows", "value": len(kernel_rows)},
        ],
        "subhead": "gpuFLOPBench is a multi-GPU benchmark atlas for floating-point rooflines, source-level coverage, and exact kernel exploration across the profiled corpus.",
    }

    write_json(DATA_DIR / "kernel-performance.json", kernel_rows)
    write_json(DATA_DIR / "source-performance.json", source_rows)

    kernel_csv_lines = [
        "kernel_row_id,source,benchmark,category,model_type,device,kernel,block_size,grid_size,exe_args,SP_FLOP,DP_FLOP,HP_FLOP,int_ops,float_flops,bytes_total,xtime_ns,arithmetic_intensity,performance_tflops,dominant_precision"
    ]
    for row in kernel_rows:
        kernel_csv_lines.append(
            ",".join(
                json.dumps(row[field])
                for field in [
                    "kernel_row_id",
                    "source",
                    "benchmark",
                    "category",
                    "model_type",
                    "device",
                    "kernel",
                    "block_size",
                    "grid_size",
                    "exe_args",
                    "SP_FLOP",
                    "DP_FLOP",
                    "HP_FLOP",
                    "int_ops",
                    "float_flops",
                    "bytes_total",
                    "xtime_ns",
                    "arithmetic_intensity",
                    "performance_tflops",
                    "dominant_precision",
                ]
            )
        )
    (DATA_DIR / "kernel-performance.csv").write_text("\n".join(kernel_csv_lines) + "\n")

    source_csv_lines = [
        "source,benchmark,category,model_type,device,kernel_count,peak_performance_tflops,median_performance_tflops,median_arithmetic_intensity,median_xtime_ns,coverage_rank"
    ]
    for row in source_rows:
        source_csv_lines.append(
            ",".join(
                json.dumps(row[field])
                for field in [
                    "source",
                    "benchmark",
                    "category",
                    "model_type",
                    "device",
                    "kernel_count",
                    "peak_performance_tflops",
                    "median_performance_tflops",
                    "median_arithmetic_intensity",
                    "median_xtime_ns",
                    "coverage_rank",
                ]
            )
        )
    (DATA_DIR / "source-performance.csv").write_text("\n".join(source_csv_lines) + "\n")

    raw_download = DOWNLOADS_DIR / "gpuFLOPBench.json.gz"
    shutil.copy2(VENDORED_DATASET_GZ, raw_download)

    metadata_payload = {
        "hero": hero,
        "inventory": inventory,
        "audit": audit,
        "device_summary": device_summary,
        "model_matrix": model_matrix,
        "category_profiled": category_profiled,
        "top_lists": top_lists,
        "roofline_specs": [
            {
                "device": device,
                "label": specs["label"],
                "architecture": specs["architecture"],
                "compute_capability": specs["compute_capability"],
                "memory_bandwidth_gbps": specs["memory_bandwidth_gbps"],
                "peak_fp16_tflops": specs["peak_tflops"]["fp16"],
                "peak_fp32_tflops": specs["peak_tflops"]["fp32"],
                "peak_fp64_tflops": specs["peak_tflops"]["fp64"],
            }
            for device, specs in GPU_SPECS.items()
        ],
        "paper": {
            "title": "Can Large Language Models Predict Parallel Code Performance?",
            "venue": "HPDC '25",
            "doi_url": "https://dl.acm.org/doi/10.1145/3731545.3743645",
            "pdf_url": "https://arxiv.org/pdf/2505.03988",
            "pdf_label": "arXiv PDF",
            "bibtex": PAPER_BIBTEX,
        },
        "team": TEAM_MEMBERS,
        "downloads": [
            {
                "label": "Compact source summary CSV",
                "path": "data/source-performance.csv",
                "href": "./data/source-performance.csv",
                "size_bytes": (DATA_DIR / "source-performance.csv").stat().st_size,
            },
            {
                "label": "Compact kernel summary CSV",
                "path": "data/kernel-performance.csv",
                "href": "./data/kernel-performance.csv",
                "size_bytes": (DATA_DIR / "kernel-performance.csv").stat().st_size,
            },
            {
                "label": "Source summary JSON",
                "path": "data/source-performance.json",
                "href": "./data/source-performance.json",
                "size_bytes": (DATA_DIR / "source-performance.json").stat().st_size,
            },
            {
                "label": "Kernel summary JSON",
                "path": "data/kernel-performance.json",
                "href": "./data/kernel-performance.json",
                "size_bytes": (DATA_DIR / "kernel-performance.json").stat().st_size,
            },
            {
                "label": "Structured benchmark dataset JSON.gz",
                "path": "downloads/gpuFLOPBench.json.gz",
                "href": "./downloads/gpuFLOPBench.json.gz",
                "size_bytes": raw_download.stat().st_size,
            },
        ],
    }

    write_json(DATA_DIR / "site-metadata.json", metadata_payload)
    write_js(
        DATA_DIR / "site-data.js",
        "GPU_FLOWBENCH_DATA",
        {
            "meta": metadata_payload,
            "kernelRows": kernel_rows,
            "sourceRows": source_rows,
        },
    )

    print(f"Wrote site datasets to {DATA_DIR}")


if __name__ == "__main__":
    main()
