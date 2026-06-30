#!/usr/bin/env python3
"""Emit aind-data-schema v2 DataProcess JSON files ("*_data_process.json").

Builds DataProcess documents as plain dicts (stdlib-only) and writes one file
per processing step. The upload capsule validates them and aggregates them into
the top-level processing.json.

Usage:
    from aind_process_record import make_data_process, write_data_process
    dp = make_data_process(
        process_type="Image atlas alignment",
        name="Image atlas alignment - 25 um",
        start=start_dt, end=end_dt,
        code_url="https://codeocean.allenneuraldynamics.org/capsule/6898460/tree",
        code_name="aind-exaspim-ccf-registration", code_version="0.0.1",
        parameters={"resolution_um": 25},
        experimenters=["Peter Grotz"], output_path="ccf_alignment/",
    )
    write_data_process(dp, "/results/ccf_alignment")
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

PIPELINE_NAME = "exaspim-data-processing"


def _iso(dt) -> str | None:
    """Normalize a datetime (or ISO string) to tz-aware ISO-8601 'Z' form."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        s = dt.isoformat()
        if s.endswith("+00:00"):
            return s.replace("+00:00", "Z")
        return s if ("+" in s or s.endswith("Z")) else s + "Z"
    return str(dt)


def make_data_process(
    *,
    process_type: str,
    name: str,
    start,
    end=None,
    code_url: str,
    code_name: str | None = None,
    code_version: str | None = None,
    run_script: str = "/code/run",
    language: str = "Python",
    parameters: dict | None = None,
    experimenters: list[str] | None = None,
    stage: str = "Processing",
    output_path: str | None = None,
    output_parameters: dict | None = None,
    notes: str | None = None,
    pipeline_name: str | None = PIPELINE_NAME,
) -> dict:
    """Build one v2 DataProcess document as a plain dict."""
    return {
        "object_type": "Data process",
        "process_type": process_type,
        "name": name,
        "stage": stage,
        "code": {
            "object_type": "Code",
            "url": code_url,
            "name": code_name,
            "version": code_version,
            "run_script": run_script,
            "language": language,
            "parameters": parameters or {},
        },
        "experimenters": experimenters or [],
        "pipeline_name": pipeline_name,
        "start_date_time": _iso(start),
        "end_date_time": _iso(end),
        "output_path": output_path,
        "output_parameters": output_parameters,
        "notes": notes,
    }


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "process"


def write_data_process(data_process: dict, dest_dir, filename: str | None = None) -> str:
    """Write one DataProcess to <dest_dir>/<name>_data_process.json."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / (filename or f"{_slug(data_process.get('name'))}_data_process.json")
    out.write_text(json.dumps(data_process, indent=3) + "\n")
    return str(out)


def write_data_processes(data_processes, dest_dir) -> list[str]:
    """Write a list of DataProcess docs, one *_data_process.json file each."""
    return [write_data_process(dp, dest_dir) for dp in data_processes]
