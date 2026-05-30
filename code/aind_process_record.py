#!/usr/bin/env python3
"""
Lightweight, dependency-free "process record" emitter for PRODUCER capsules.

Centralized-metadata architecture (PLAN.md §10.8 / §13.3): producer capsules do
NOT build aind-data-schema objects. They drop a small `process_record.json` into
their results subfolder; the UPLOAD capsule (the only one with the v2 stack)
converts records -> validated v2 Processing.

This module is STDLIB-ONLY on purpose, so it runs unchanged in every producer
env — incl. the registration capsule's Python 3.9 (no aind-data-schema, no
Python bump, no registration-repro risk).

Vendoring: copy this file into each producer capsule's `code/`. Keep one
canonical copy here (`_capsules/_shared/`) as the source of truth.

Usage (in a producer's run / main):
    from aind_process_record import make_record, write_records
    rec = make_record(
        process_type="Image atlas alignment",
        name="Image atlas alignment - 25 um",
        start=start_dt, end=end_dt,            # datetimes or ISO strings
        code_url="https://github.com/AllenNeuralDynamics/aind-exaspim-ccf-registration.git",
        code_name="aind-exaspim-ccf-registration",
        code_version=os.environ.get("CODE_VERSION", "0.0.0"),
        parameters={"resolution_um": 25, "moving_mask": "registration_metadata/..._mask_25um.nii.gz"},
        experimenters=["Di Wang"],
        output_path="ccf_alignment/",          # asset-root-relative
        notes="sample -> template -> CCF",
    )
    write_records([rec], "/results/ccf_alignment")
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

PIPELINE_NAME = "aind-exaSPIM-ccf-registration"


def _iso(dt) -> str | None:
    """Normalize a datetime (or ISO string) to tz-aware ISO-8601 'Z' form."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        s = dt.isoformat()
        return s.replace("+00:00", "Z") if s.endswith("+00:00") else (s + "Z" if "+" not in s and "Z" not in s else s)
    return str(dt)


def make_record(
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
    """Build one schema-agnostic process record (plain dict). No validation here;
    the upload capsule validates when it constructs the v2 DataProcess."""
    return {
        "process_type": process_type,
        "name": name,
        "stage": stage,
        "start_date_time": _iso(start),
        "end_date_time": _iso(end),
        "experimenters": experimenters or [],
        "pipeline_name": pipeline_name,
        "code": {
            "url": code_url,
            "name": code_name,
            "version": code_version,
            "run_script": run_script,
            "language": language,
        },
        "parameters": parameters or {},
        "output_path": output_path,
        "output_parameters": output_parameters,
        "notes": notes,
    }


def write_records(records, dest_dir, filename: str = "process_record.json") -> str:
    """Write a list of records (or a single record) to <dest_dir>/<filename>."""
    if isinstance(records, dict):
        records = [records]
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / filename
    out.write_text(json.dumps(records, indent=3) + "\n")
    return str(out)
