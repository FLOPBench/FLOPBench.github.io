#!/usr/bin/env python3
"""Export paper-subset LLM prediction data for the FLOPBench website.

This script reads the already-loaded PostgreSQL experiment database. It never
restores, dumps, wipes, deletes, or otherwise mutates database state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


WEBSITE_ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = WEBSITE_ROOT / "FLOPBench"
DATA_DIR = SITE_DIR / "data"
LLM_RESULTS_DIR = DATA_DIR / "llm-results"
SOURCE_DATA_DIR = SITE_DIR / "source-data"
PAPER_SUBSET_PATH = SOURCE_DATA_DIR / "paper-kernel-subset.json"
PAPER_LLM_INDEX_PATH = DATA_DIR / "paper-llm-index.json"
UPDATED_REPO_ROOT = Path(os.environ.get("GPUFLOPBENCH_UPDATED_ROOT", "/gpuFLOPBench-updated"))
DEFAULT_GPUFLOPS_DB_URI = os.environ.get(
    "GPUFLOPBENCH_DB_URI",
    "postgresql://postgres:postgres@localhost:5432/gpuflops_db",
)

AI_PRECISIONS = ("fp16", "fp32", "fp64")
PAPER_MODELS = {"GPT 5.4", "GPT OSS", "Opus 4.6"}
GPU_ORDER = ("3080", "A10", "A100", "H100")
PROMPT_ORDER = ((False, False), (True, False))
SASS_LABELS = {False: "Source-Only", True: "Source+SASS"}
ZERO_CLASS = "zero"
NEGATIVE_CLASS = "bandwidth-bound"
POSITIVE_CLASS = "compute-bound"

GPU_ROOFLINE_TABLE = {
    "3080": {
        "memory_bandwidth_gb_per_s": 760.0,
        "peak_tflops": {"fp16": 30.55, "fp32": 30.55, "fp64": 0.477},
    },
    "A10": {
        "memory_bandwidth_gb_per_s": 600.0,
        "peak_tflops": {"fp16": 15.62, "fp32": 15.62, "fp64": 0.244},
    },
    "A100": {
        "memory_bandwidth_gb_per_s": 1555.0,
        "peak_tflops": {"fp16": 77.97, "fp32": 19.49, "fp64": 9.75},
    },
    "H100": {
        "memory_bandwidth_gb_per_s": 3360.0,
        "peak_tflops": {"fp16": 133.82, "fp32": 66.91, "fp64": 33.45},
    },
}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_updated_helpers() -> tuple[Any, Any, Any]:
    direct_dir = UPDATED_REPO_ROOT / "experiments" / "direct-prompting"
    db_manager = load_module("website_direct_db_manager", direct_dir / "db_manager.py")
    result_viz = load_module("website_result_viz_helper", direct_dir / "result_viz_helper.py")
    prompts = load_module("website_direct_prompts", direct_dir / "prompts.py")
    return db_manager, result_viz, prompts


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def safe_divide(numerator: Any, denominator: Any) -> float | None:
    top = finite_float(numerator)
    bottom = finite_float(denominator)
    if top is None or bottom is None or bottom == 0.0:
        return None
    return top / bottom


def pct_diff(predicted: Any, expected: Any) -> float | None:
    predicted_float = finite_float(predicted)
    expected_float = finite_float(expected)
    if predicted_float is None or expected_float is None:
        return None
    if expected_float == 0.0:
        return 0.0 if predicted_float == 0.0 else None
    return (predicted_float - expected_float) / expected_float * 100.0


def balance_point(gpu: str, precision: str) -> float | None:
    spec = GPU_ROOFLINE_TABLE.get(gpu)
    if not spec:
        return None
    return spec["peak_tflops"][precision] * 1000.0 / spec["memory_bandwidth_gb_per_s"]


def classify_ai(ai_value: Any, balance: Any, *, include_zero: bool = True) -> str | None:
    ai_float = finite_float(ai_value)
    balance_float = finite_float(balance)
    if ai_float is None:
        return None
    if include_zero and ai_float == 0.0:
        return ZERO_CLASS
    if balance_float is None:
        return None
    return NEGATIVE_CLASS if ai_float <= balance_float else POSITIVE_CLASS


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "unknown"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def prompt_type(use_sass: bool) -> str:
    return SASS_LABELS[bool(use_sass)]


def fetch_completed_checkpoints(db_uri: str, include_dry_run: bool) -> dict[str, dict[str, Any]]:
    _db_manager, result_viz, _prompts = load_updated_helpers()
    parser = result_viz.CheckpointDBParser(db_uri)
    attempt_tracker = result_viz.QueryAttemptTracker(db_uri)
    try:
        checkpoints = parser.fetch_all_checkpoints()
        tail_result = parser.fetch_tail_checkpoints_by_thread(
            checkpoints=checkpoints,
            tolerate_errors=True,
        )
        tail_checkpoints = tail_result["tails"]
        result_viz._print_invalid_thread_warnings(tail_result["invalid_threads"])
        for thread_id, checkpoint in tail_checkpoints.items():
            if not include_dry_run and result_viz._is_dry_run_thread(thread_id):
                continue
            channel_values = checkpoint["checkpoint"].get("channel_values", {})
            if "total_tokens" not in channel_values:
                continue
            parser.hydrate_checkpoint_channels(
                checkpoint,
                [
                    "prediction",
                    "raw_response",
                    "metrics_diff",
                    "metrics_pct_diff",
                    "metrics_explanations",
                    "source_code_files",
                    "compile_commands",
                    "gpu_roofline_specs",
                    "sass_dict",
                    "imix_dict",
                ],
            )
        attempts = attempt_tracker.fetch_all_attempts()
    finally:
        parser.close()
        attempt_tracker.close()

    completed = result_viz._completed_checkpoint_by_thread(tail_checkpoints)
    return {
        thread_id: checkpoint
        for thread_id, checkpoint in completed.items()
        if include_dry_run or not result_viz._is_dry_run_thread(thread_id)
    }


def build_samples_dataframe(
    completed_checkpoints: dict[str, dict[str, Any]],
    include_dry_run: bool,
) -> pd.DataFrame:
    _db_manager, result_viz, _prompts = load_updated_helpers()
    records = result_viz._extract_completed_records(completed_checkpoints, include_dry_run)
    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        return dataframe
    dataframe["use_sass"] = dataframe["use_sass"].fillna(False).astype(bool)
    dataframe["use_imix"] = dataframe["use_imix"].fillna(False).astype(bool)
    return dataframe


def filter_paper_rows(samples_df: pd.DataFrame, only_shared_samples: bool) -> pd.DataFrame:
    if samples_df.empty:
        return samples_df

    filtered = samples_df[
        (samples_df["status"] == "completed")
        & (samples_df["model_name"].isin(PAPER_MODELS))
        & (samples_df["use_imix"] == False)
        & (
            samples_df.apply(
                lambda row: (bool(row["use_sass"]), bool(row["use_imix"])) in PROMPT_ORDER,
                axis=1,
            )
        )
    ].copy()

    identity_columns = ["program_name", "kernel_mangled_name"]
    if only_shared_samples:
        anchor = filtered[
            (filtered["model_name"] == "GPT 5.4")
            & (filtered["use_sass"] == True)
            & (filtered["use_imix"] == False)
        ].copy()
        keys = (
            anchor[identity_columns + ["gpu"]]
            .drop_duplicates()
            .groupby(identity_columns, dropna=False)
            .size()
            .reset_index(name="gpu_count")
        )
        keys = keys[keys["gpu_count"] == len(GPU_ORDER)][identity_columns]
        filtered = filtered.merge(keys, on=identity_columns, how="inner")

    return filtered.sort_values(["program_name", "kernel_mangled_name", "gpu", "model_name", "use_sass"])


def enrich_prediction_rows(samples_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in samples_df.iterrows():
        expected_total_bytes = (finite_float(row.get("expected_read_bytes")) or 0.0) + (
            finite_float(row.get("expected_write_bytes")) or 0.0
        )
        predicted_read = (finite_float(row.get("expected_read_bytes")) or 0.0) + (
            finite_float(row.get("metrics_diff_read_bytes")) or 0.0
        )
        predicted_write = (finite_float(row.get("expected_write_bytes")) or 0.0) + (
            finite_float(row.get("metrics_diff_write_bytes")) or 0.0
        )
        predicted_total_bytes = predicted_read + predicted_write

        for precision in AI_PRECISIONS:
            expected_flops = finite_float(row.get(f"expected_{precision}"))
            predicted_flops = (expected_flops or 0.0) + (
                finite_float(row.get(f"metrics_diff_{precision}")) or 0.0
            )
            expected_ai = safe_divide(expected_flops, expected_total_bytes)
            predicted_ai = safe_divide(predicted_flops, predicted_total_bytes)
            raw_diff = (
                predicted_ai - expected_ai
                if predicted_ai is not None and expected_ai is not None
                else None
            )
            percent = pct_diff(predicted_ai, expected_ai)
            balance = balance_point(str(row["gpu"]), precision)

            records.append(
                {
                    "sample_id": slug(
                        f"{row['program_name']}::{row['kernel_mangled_name']}::{row['gpu']}::{row['model_name']}::{prompt_type(row['use_sass'])}::{precision}"
                    ),
                    "thread_id": row["thread_id"],
                    "program_name": row["program_name"],
                    "runtime": row["runtime"],
                    "kernel_mangled_name": row["kernel_mangled_name"],
                    "kernel_demangled_name": row["kernel_demangled_name"],
                    "gpu": row["gpu"],
                    "model_name": row["model_name"],
                    "safe_model_name": row["safe_model_name"],
                    "use_sass": bool(row["use_sass"]),
                    "prompt_type": prompt_type(bool(row["use_sass"])),
                    "precision": precision,
                    "expected_ai": expected_ai,
                    "predicted_ai": predicted_ai,
                    "ai_raw_diff": raw_diff,
                    "ai_abs_error": abs(raw_diff) if raw_diff is not None else None,
                    "ai_percent_diff": percent,
                    "ai_abs_percent_error": abs(percent) if percent is not None else None,
                    "balance_point": balance,
                    "expected_bound": classify_ai(expected_ai, balance),
                    "predicted_bound": classify_ai(predicted_ai, balance),
                    "expected_flops": expected_flops,
                    "predicted_flops": predicted_flops,
                    "expected_total_bytes": expected_total_bytes,
                    "predicted_total_bytes": predicted_total_bytes,
                    "query_time": finite_float(row.get("query_time")),
                    "cost_usd": finite_float(row.get("cost_usd")),
                    "input_tokens": finite_float(row.get("input_tokens")),
                    "output_tokens": finite_float(row.get("output_tokens")),
                    "total_tokens": finite_float(row.get("total_tokens")),
                    "result_shard": f"data/llm-results/{slug(str(row['program_name']))}.json",
                }
            )
    return records


def reconstruct_prompt(state: dict[str, Any], prompts_module: Any) -> dict[str, str | None]:
    try:
        generator = prompts_module.DirectPromptGenerator(
            program_name=state["program_name"],
            kernel_mangled_name=state["kernel_mangled_name"],
            kernel_demangled_name=state["kernel_demangled_name"],
            source_code_files=state.get("source_code_files") or {},
            gpu_roofline_specs=state.get("gpu_roofline_specs") or {},
            compile_commands=state.get("compile_commands") or [],
            exe_args=state.get("exe_args") or "",
            sass_dict=state.get("sass_dict"),
            imix_dict=None,
        )
        return {
            "system_prompt": generator.generate_system_prompt(),
            "human_prompt": generator.generate_prompt(),
        }
    except Exception as error:
        return {
            "system_prompt": None,
            "human_prompt": f"Prompt reconstruction failed: {error}",
        }


def build_shard_payloads(
    paper_df: pd.DataFrame,
    completed_checkpoints: dict[str, dict[str, Any]],
    prompts_module: Any,
) -> dict[str, dict[str, Any]]:
    shard_payloads: dict[str, dict[str, Any]] = defaultdict(lambda: {"prompts": {}, "records": []})
    selected_thread_ids = set(paper_df["thread_id"].tolist())

    for thread_id in selected_thread_ids:
        checkpoint = completed_checkpoints.get(thread_id)
        if not checkpoint:
            continue
        state = checkpoint["checkpoint"]["channel_values"]
        matching_row = paper_df[paper_df["thread_id"] == thread_id].iloc[0]
        prompt = reconstruct_prompt(state, prompts_module)
        prediction = state.get("prediction") if isinstance(state.get("prediction"), dict) else {}
        raw_response = state.get("raw_response")
        metrics_explanations = (
            state.get("metrics_explanations")
            if isinstance(state.get("metrics_explanations"), dict)
            else {}
        )
        program_name = str(state.get("program_name") or matching_row["program_name"])
        prompt_key = slug(
            f"{program_name}::{state.get('kernel_mangled_name')}::{matching_row['gpu']}::{prompt_type(bool(matching_row['use_sass']))}"
        )
        shard_payloads[program_name]["prompts"].setdefault(prompt_key, prompt)
        shard_payloads[program_name]["records"].append(
            {
                "thread_id": thread_id,
                "program_name": program_name,
                "kernel_mangled_name": state.get("kernel_mangled_name"),
                "kernel_demangled_name": state.get("kernel_demangled_name"),
                "gpu": matching_row["gpu"],
                "model_name": matching_row["model_name"],
                "use_sass": bool(matching_row["use_sass"]),
                "prompt_type": prompt_type(bool(matching_row["use_sass"])),
                "expected": {
                    "fp16": json_ready(state.get("expected_fp16")),
                    "fp32": json_ready(state.get("expected_fp32")),
                    "fp64": json_ready(state.get("expected_fp64")),
                    "read_bytes": json_ready(state.get("expected_read_bytes")),
                    "write_bytes": json_ready(state.get("expected_write_bytes")),
                    "grid_size": json_ready(state.get("expected_grid_size")),
                    "block_size": json_ready(state.get("expected_block_size")),
                },
                "prediction": json_ready(prediction),
                "raw_response": json_ready(raw_response),
                "metrics_explanations": json_ready(metrics_explanations),
                "prompt_key": prompt_key,
            }
        )

    return shard_payloads


def reset_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                reset_directory(child)
                child.rmdir()
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbUri", default=DEFAULT_GPUFLOPS_DB_URI)
    parser.add_argument("--includeDryRun", action="store_true")
    parser.add_argument(
        "--allowIncomplete",
        action="store_true",
        help="Keep completed source-only/source+SASS records even if not shared across all paper combinations.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    _db_manager, _result_viz, prompts_module = load_updated_helpers()
    completed_checkpoints = fetch_completed_checkpoints(args.dbUri, args.includeDryRun)
    samples_df = build_samples_dataframe(completed_checkpoints, args.includeDryRun)
    paper_df = filter_paper_rows(samples_df, only_shared_samples=not args.allowIncomplete)
    if paper_df.empty:
        raise RuntimeError("No completed paper-subset source-only/source+SASS rows were found.")

    paper_kernels = (
        paper_df[
            [
                "program_name",
                "runtime",
                "kernel_mangled_name",
                "kernel_demangled_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(["program_name", "kernel_mangled_name"])
        .to_dict(orient="records")
    )
    prediction_rows = enrich_prediction_rows(paper_df)
    shard_payloads = build_shard_payloads(paper_df, completed_checkpoints, prompts_module)

    reset_directory(LLM_RESULTS_DIR)
    for program_name, payload in sorted(shard_payloads.items()):
        write_json(LLM_RESULTS_DIR / f"{slug(program_name)}.json", payload)

    model_counts = (
        paper_df.groupby(["model_name", "use_sass"], dropna=False)
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    write_json(
        PAPER_SUBSET_PATH,
        {
            "source": "gpuflops_db",
            "selection": "completed paper models, source-only and source+SASS, no IMIX, shared across paper combinations",
            "kernel_count": len(paper_kernels),
            "kernels": paper_kernels,
        },
    )
    write_json(
        PAPER_LLM_INDEX_PATH,
        {
            "kernel_count": len(paper_kernels),
            "sample_count": int(paper_df.shape[0]),
            "prediction_row_count": len(prediction_rows),
            "models": sorted(paper_df["model_name"].dropna().unique().tolist()),
            "gpus": [gpu for gpu in GPU_ORDER if gpu in set(paper_df["gpu"].dropna())],
            "prompt_types": ["Source-Only", "Source+SASS"],
            "model_prompt_counts": json_ready(model_counts),
            "predictionRows": prediction_rows,
            "resultShards": {
                program_name: f"data/llm-results/{slug(program_name)}.json"
                for program_name in sorted(shard_payloads)
            },
        },
    )
    print(
        f"Exported {len(paper_kernels)} kernels, {int(paper_df.shape[0])} samples, "
        f"and {len(prediction_rows)} prediction rows."
    )


if __name__ == "__main__":
    main()
