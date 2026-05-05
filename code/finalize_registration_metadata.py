#!/usr/bin/env python3
"""Finalize ccf_alignment processing metadata after all capsule outputs exist."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


RESULTS_DIR = Path("../results")
CCF_ALIGNMENT_DIR = RESULTS_DIR / "ccf_alignment"
PROCESSING_PATH = CCF_ALIGNMENT_DIR / "processing.json"
PIPELINE_NAME = "aind-exaspim-ccf-registration"
PIPELINE_URL = "https://codeocean.allenneuraldynamics.org/capsule/1087961/tree"


def main() -> None:
    if not PROCESSING_PATH.exists():
        raise FileNotFoundError(f"Missing registration metadata: {PROCESSING_PATH}")

    processing = json.loads(PROCESSING_PATH.read_text())
    subject_id = _find_subject_id()

    _add_alignment_outputs(processing, subject_id)
    _append_annotation_process(processing, subject_id)

    PROCESSING_PATH.write_text(json.dumps(processing, indent=3) + "\n")
    print(f"Finalized registration processing metadata: {PROCESSING_PATH}")


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
            match = re.search(r"(\d{6})", str(value))
            if match:
                return match.group(1)

    raise RuntimeError("Could not determine subject id for registration metadata")


def _add_alignment_outputs(processing: dict, subject_id: str) -> None:
    outputs = {
        "transforms": _existing_paths(
            [
                CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_0GenericAffine.mat",
                CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_1Warp.nii.gz",
                CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz",
            ]
        ),
        "aligned_image": _existing_path(CCF_ALIGNMENT_DIR / "ccf_aligned.zarr"),
        "registration_metadata": _existing_path(CCF_ALIGNMENT_DIR / "registration_metadata"),
    }
    outputs = {key: value for key, value in outputs.items() if value}

    for process in processing.get("data_processes", []):
        if process.get("process_type") != "Image atlas alignment":
            continue
        if process.get("output_path") == "ccf_alignment/" or process.get("name", "").endswith("25 um"):
            output_parameters = process.setdefault("output_parameters", {})
            output_parameters["standard_outputs"] = outputs


def _append_annotation_process(processing: dict, subject_id: str) -> None:
    process_name = "CCF annotation to sample space"
    data_processes = processing.setdefault("data_processes", [])
    data_processes[:] = [
        process for process in data_processes
        if process.get("name") != process_name
    ]

    annotation_dir = CCF_ALIGNMENT_DIR / "ccf_anno_to_sample"
    start_date_time, end_date_time = _output_time_bounds(annotation_dir)
    data_processes.append(
        {
            "object_type": "Data process",
            "process_type": "Image atlas alignment",
            "name": process_name,
            "stage": "Processing",
            "code": {
                "object_type": "Code",
                "url": PIPELINE_URL,
                "name": PIPELINE_NAME,
                "version": "0.0.1",
                "container": None,
                "run_script": "code/run_register_ccf_annotation_10um.sh",
                "language": "Python",
                "language_version": None,
                "input_data": None,
                "parameters": {
                    "subject_id": subject_id,
                    "ccf_annotation_path": "../data/allen_mouse_ccf/annotation/ccf_2017/annotation_10.nii.gz",
                    "ccf_template_path": "../data/allen_mouse_ccf/average_template/average_template_10.nii.gz",
                    "exaspim_template_path": "../data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz",
                    "level": 2,
                    "seg_path": _asset_path(annotation_dir),
                    "ccf_to_template_transforms": [
                        "/data/reg_exaspim_template_to_ccf_25um_v1.4/0GenericAffine.mat",
                        "/data/reg_exaspim_template_to_ccf_25um_v1.4/1InverseWarp.nii.gz",
                    ],
                    "template_to_sample_transforms": [
                        _asset_path(CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_0GenericAffine.mat"),
                        _asset_path(CCF_ALIGNMENT_DIR / f"{subject_id}_to_exaSPIM_SyN_1InverseWarp.nii.gz"),
                    ],
                },
                "core_dependency": None,
            },
            "experimenters": ["Peter Grotz"],
            "pipeline_name": PIPELINE_NAME,
            "start_date_time": start_date_time,
            "end_date_time": end_date_time,
            "output_path": _asset_path(annotation_dir),
            "output_parameters": {
                "standard_outputs": {
                    "annotation_image": _existing_path(annotation_dir / "ccf_anno_in_sample_space.nii.gz"),
                    "annotation_zarr": _existing_path(annotation_dir / "ccf_anno_in_sample_space.zarr"),
                    "precomputed_annotation": _existing_path(annotation_dir / "ccf_annotation_precomputed"),
                    "figures": _existing_path(annotation_dir / "figures"),
                    "sample_image": _existing_path(annotation_dir / "sample.nii.gz"),
                },
                "logs": _existing_paths(
                    [
                        annotation_dir / "register_process.log",
                        annotation_dir / "dask_report.html",
                    ]
                ),
            },
            "notes": "Transforms CCF annotation into sample space and writes zarr/precomputed label volumes.",
            "resources": None,
        }
    )

    graph = processing.setdefault("dependency_graph", {})
    graph[process_name] = _annotation_dependencies(processing)


def _annotation_dependencies(processing: dict) -> list:
    dependencies = [
        process.get("name")
        for process in processing.get("data_processes", [])
        if process.get("process_type") == "Image atlas alignment"
        and process.get("name") != "CCF annotation to sample space"
    ]
    return [dependency for dependency in dependencies if dependency]


def _output_time_bounds(path: Path) -> Tuple[str, str]:
    marker_files = [
        path / "register_process.log",
        path / "dask_report.html",
        path / "ccf_anno_in_sample_space.nii.gz",
        path / "sample.nii.gz",
    ]
    timestamps = [
        candidate.stat().st_mtime
        for candidate in marker_files
        if candidate.exists()
    ]
    if not timestamps:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return now, now
    return (
        datetime.fromtimestamp(min(timestamps), timezone.utc).isoformat().replace("+00:00", "Z"),
        datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _existing_paths(paths: List[Path]) -> List[str]:
    return [_asset_path(path) for path in paths if path.exists()]


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
