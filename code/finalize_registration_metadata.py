#!/usr/bin/env python3
"""Write a *_data_process.json describing the CCF-annotation step from its outputs.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from aind_process_record import make_data_process, write_data_process

RESULTS_DIR = Path("../results")
CCF_ALIGNMENT_DIR = RESULTS_DIR / "ccf_alignment"
# Code Ocean capsule URL recorded in the DataProcess.
CODE_URL = "https://codeocean.allenneuraldynamics.org/capsule/6898460/tree"
CODE_NAME = "aind-exaspim-ccf-registration"
VERSION = "0.0.1"


def main() -> None:
    subject_id = _find_subject_id()
    annotation_dir = CCF_ALIGNMENT_DIR / "ccf_anno_to_sample"
    start_date_time, end_date_time = _output_time_bounds(annotation_dir)

    standard_outputs = {
        "annotation_image": _existing_path(annotation_dir / "ccf_anno_in_sample_space.nii.gz"),
        "annotation_zarr": _existing_path(annotation_dir / "ccf_anno_in_sample_space.zarr"),
    }
    standard_outputs = {k: v for k, v in standard_outputs.items() if v}

    dp = make_data_process(
        process_type="Image atlas alignment",
        name="CCF annotation to sample space",
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
    out = write_data_process(dp, str(CCF_ALIGNMENT_DIR))
    print(f"Wrote annotation data_process.json: {out}")


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
