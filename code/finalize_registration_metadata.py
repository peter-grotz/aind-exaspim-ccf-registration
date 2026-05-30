#!/usr/bin/env python3
"""Append the CCF-annotation process record after the annotation step runs.

Centralized-metadata architecture: main.py writes ccf_alignment/process_record.json
(a list of lightweight process records). The annotation step
(run_register_ccf_annotation_10um.sh) runs AFTER main.py, so this script appends
the annotation record and enriches the 25um alignment record's outputs once the
files exist. The upload capsule converts records -> validated v2 processing.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from aind_process_record import make_record

RESULTS_DIR = Path("../results")
CCF_ALIGNMENT_DIR = RESULTS_DIR / "ccf_alignment"
RECORD_PATH = CCF_ALIGNMENT_DIR / "process_record.json"
CODE_URL = "https://github.com/AllenNeuralDynamics/aind-exaspim-ccf-registration.git"
CODE_NAME = "aind-exaspim-ccf-registration"
VERSION = "0.0.1"
ANNOTATION_NAME = "CCF annotation to sample space"


def main() -> None:
    records = json.loads(RECORD_PATH.read_text()) if RECORD_PATH.exists() else []
    subject_id = _find_subject_id()

    _enrich_alignment_outputs(records, subject_id)
    records = [r for r in records if r.get("name") != ANNOTATION_NAME]
    records.append(_annotation_record(subject_id))

    RECORD_PATH.write_text(json.dumps(records, indent=3) + "\n")
    print(f"Appended annotation record to: {RECORD_PATH}")


def _enrich_alignment_outputs(records: list, subject_id: str) -> None:
    outputs = {
        "transforms": _existing_paths(
            [
                CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_0GenericAffine.mat",
                CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_1Warp.nii.gz",
                CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz",
            ]
        ),
        "aligned_image": _existing_path(CCF_ALIGNMENT_DIR / "ccf_aligned.zarr"),
    }
    outputs = {k: v for k, v in outputs.items() if v}
    for r in records:
        if r.get("process_type") == "Image atlas alignment" and str(r.get("name", "")).endswith("25 um"):
            op = r.get("output_parameters") or {}
            op["standard_outputs"] = outputs
            r["output_parameters"] = op


def _annotation_record(subject_id: str) -> dict:
    annotation_dir = CCF_ALIGNMENT_DIR / "ccf_anno_to_sample"
    start_date_time, end_date_time = _output_time_bounds(annotation_dir)
    standard_outputs = {
        "annotation_image": _existing_path(annotation_dir / "ccf_anno_in_sample_space.nii.gz"),
        "annotation_zarr": _existing_path(annotation_dir / "ccf_anno_in_sample_space.zarr"),
    }
    standard_outputs = {k: v for k, v in standard_outputs.items() if v}
    return make_record(
        process_type="Image atlas alignment",
        name=ANNOTATION_NAME,
        start=start_date_time,
        end=end_date_time,
        code_url=CODE_URL,
        code_name=CODE_NAME,
        code_version=VERSION,
        run_script="/code/run_register_ccf_annotation_10um.sh",
        experimenters=["Peter Grotz"],
        parameters={
            "subject_id": subject_id,
            "ccf_annotation_path": "../data/allen_mouse_ccf/annotation/ccf_2017/annotation_10.nii.gz",
            "ccf_template_path": "../data/allen_mouse_ccf/average_template/average_template_10.nii.gz",
            "exaspim_template_path": "../data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz",
            "level": 2,
        },
        output_path=_asset_path(annotation_dir),
        output_parameters={"standard_outputs": standard_outputs} if standard_outputs else None,
        notes="Transforms CCF annotation into sample space and writes zarr/precomputed label volumes.",
    )


def _find_subject_id() -> str:
    transforms = sorted(CCF_ALIGNMENT_DIR.glob("*_to_exaSPIM_SyN_0GenericAffine.mat"))
    if transforms:
        return transforms[0].name.split("_to_exaSPIM_", 1)[0]

    manifest = next(Path("../data").glob("exaspim_manifest*.json"), None)
    if manifest:
        payload = json.loads(manifest.read_text())
        for key in ("subject_id", "name"):
            value = payload.get(key)
            if not value:
                continue
            import re
            match = re.search(r"(\d{6})", str(value))
            if match:
                return match.group(1)

    raise RuntimeError("Could not determine subject id for registration metadata")


def _output_time_bounds(path: Path) -> Tuple[str, str]:
    marker_files = [
        path / "register_process.log",
        path / "dask_report.html",
        path / "ccf_anno_in_sample_space.nii.gz",
        path / "sample.nii.gz",
    ]
    timestamps = [c.stat().st_mtime for c in marker_files if c.exists()]
    if not timestamps:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return now, now
    return (
        datetime.fromtimestamp(min(timestamps), timezone.utc).isoformat().replace("+00:00", "Z"),
        datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _existing_paths(paths: List[Path]) -> List[str]:
    return [_asset_path(p) for p in paths if p.exists()]


def _existing_path(path: Path) -> Optional[str]:
    return _asset_path(path) if path.exists() else None


def _asset_path(path: Path) -> str:
    path = path.resolve()
    results_dir = RESULTS_DIR.resolve()
    try:
        return str(path.relative_to(results_dir))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
