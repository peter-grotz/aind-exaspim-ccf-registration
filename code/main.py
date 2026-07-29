"""Registration of exaSPIM data to the Allen Mouse Brain CCF, via the exaSPIM template."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional
import argparse

import ants
import numpy as np
import zarr
import dask.array as da
from numcodecs import blosc
import glob
import s3fs
from urllib.parse import urlparse
import shutil

from pathlib import Path
from aind_exaspim_ccf_reg.utils import (
    create_logger, 
    read_json_as_dict, 
    prepare_config_sample, 
    create_folder,
    extract_dataset_id
)
from aind_exaspim_ccf_reg.configs import PathLike, RegSchema
from aind_exaspim_ccf_reg.preprocess import perc_normalization, check_orientation
from aind_exaspim_ccf_reg.plots import plot_reg, plot_antsimgs
from aind_process_record import make_data_process, write_data_processes
from aind_exaspim_ccf_reg.register import RegistrationPipeline
from argschema import ArgSchemaParser

__version__ = "0.0.1"
# Code Ocean capsule holding the running code.
code_url = "https://codeocean.allenneuraldynamics.org/capsule/6898460/tree"


def load_zarr(
    image_path: PathLike, 
    logger: logging.Logger
) -> np.ndarray:
    """Load an OME-Zarr image as a squeezed numpy array."""
    image = zarr.open(image_path, mode="r")
    image = np.squeeze(np.squeeze(np.array(image), axis=0), axis=0)
    logger.info("----"*10)
    logger.info(f"Loading OMEZarr image from path: {image_path}")
    logger.info(f"image shape: {image.shape}")
    logger.info("----"*10)
        
    return image


def upload_alignment_data(
    s3_path: str,
    folder_to_upload: PathLike,
) -> str:
    """Recursively upload a results folder to an s3:// path.

    Parameters
    ----------
    s3_path: str
        Destination, e.g. s3://{bucket_path}/{new_dataset_name}
    folder_to_upload: PathLike
        Results folder path in Code Ocean
    """

    print(f"upload files to path {s3_path}")

    fs = s3fs.S3FileSystem()
    url = urlparse(s3_path)
    print(f"url: {url}")

    if url.scheme != "s3":
        raise NotImplementedError("Only s3 output_uri is supported, not {url.scheme}")
    
    print(f"uploading {folder_to_upload}")
    fs.put(
        folder_to_upload, url.netloc + url.path.rstrip("/") + "/", recursive=True, maxdepth=10
    ) 


def get_root_s3_prefix(s3_uri, levels_up=1):
    """Return the s3:// prefix `levels_up` path segments below the bucket root."""
    scheme, bucket_and_key = s3_uri.split('://', 1)
    bucket, *key_parts = bucket_and_key.split('/')

    base_key = '/'.join(key_parts[:levels_up])
    return f's3://{bucket}/{base_key}/'

def main() -> None:
    """Run the CCF registration pipeline: load config and image, register at
    25 um, optionally apply transforms at 10 um, and write processing metadata."""
    DATA_FOLDER = os.path.abspath("../data")
    RESULTS_FOLDER = os.path.abspath("../results")
    CCF_FOLDER = os.path.abspath(f"{DATA_FOLDER}/allen_mouse_ccf")
    start_time = datetime.now(timezone.utc)

    processing_manifest_file = os.path.abspath(glob.glob(f"{DATA_FOLDER}/*.json")[0])
    try:
        with open(processing_manifest_file, 'r') as f:
            dataset_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: {processing_manifest_file} not found.")
        return

    shutil.copy(processing_manifest_file, f"{RESULTS_FOLDER}/exaspim_manifest.json")

    print(f"processing_manifest_file: {processing_manifest_file}")

    dataset_path = str(dataset_config["zarr_multiscale"]["input_uri"])
    fused_path = dataset_path  # read the fused image + mask from the input asset
    level = 3
    resolution = 25

    # dataset_path = str(dataset_config["pipeline_processing"]["registration"]["alignment_channel_path"])
    # level = int(str(dataset_config["pipeline_processing"]["registration"]["level"]))
    # resolution = int(str(dataset_config["pipeline_processing"]["registration"]["resolution"]))
    
    dataset_id = extract_dataset_id(dataset_path)
    print("Dataset ID:", dataset_id)
    
    outprefix_reg = f"{RESULTS_FOLDER}/ccf_alignment/"
    create_folder(dest_dir=outprefix_reg, verbose=True)

    outprefix = f"{RESULTS_FOLDER}/ccf_alignment/registration_metadata/"
    create_folder(dest_dir=outprefix, verbose=True)
    logger = create_logger(output_log_path=outprefix)
    logger.info(f"Processing dataset: {dataset_id}, level={level}, resolution={resolution}um - Output dir: {outprefix} - Config: {dataset_config}")
    logger.info(dataset_path)

    exaspim_to_ccf_transform_path = [
        os.path.abspath("/data/reg_exaspim_template_to_ccf_25um_v1.4/1Warp.nii.gz"),
        os.path.abspath("/data/reg_exaspim_template_to_ccf_25um_v1.4/0GenericAffine.mat")
    ]
    
    logger.info(f"Starting processing for sample {dataset_id}")
    acquisition_output = prepare_config_sample(
        dataset_path=dataset_path,
        logger=logger,
        dataset_id=dataset_id,
        acquisition_output=outprefix,
    )
    logger.info(f"acquisition_output: {acquisition_output}")
    
    example_input = {
        "dataset_path": fused_path,
        "level": level,
        "resolution": resolution,
        "dataset_id": dataset_id,
        "acquisition_output": acquisition_output, 
        "bucket_path": "aind-open-data",
        "outprefix_reg": outprefix_reg,
        "outprefix": outprefix,
        "exaspim_to_ccf_transform_path": exaspim_to_ccf_transform_path,
        "reg_param_25um": {
            "sample_scale": [i/1000.0 for i in [20.25, 20.25, 27]],
            "exaspim_template_path": "../data/exaSPIM_template_25um/exaspim_template_7sujects_nomask_25um_round6.nii.gz",
            "ccf_path": '../data/allen_mouse_ccf/average_template/average_template_25.nii.gz',
            "affine_reg_iterations": [200, 100, 25, 3],
            "syn_reg_iterations": [200, 100, 25, 3],
        },
        "reg_param_10um": {
            "sample_scale": [i/1000.0 for i in [10.125, 10.125, 13.5]],
            "exaspim_template_path": "../data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz",
            "ccf_path": '../data/allen_mouse_ccf/average_template/average_template_10.nii.gz',
            "affine_reg_iterations": [300, 200, 50, 0],
            "syn_reg_iterations": [400, 200, 40, 0],
        }
    }

    # ants_exaspim = ants.image_read(os.path.abspath("/data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz")) # 10um

    records = []

    # Create ArgSchemaParser from example_input
    parser = ArgSchemaParser(schema_type=RegSchema, input_data=example_input)
    pipeline = RegistrationPipeline(logger)

    # Load image and register at 25 um, recorded as one process (load folded in).
    start_date_time = datetime.now(timezone.utc)
    image_path = f"{fused_path.rstrip('/')}/{level}"  # rstrip keeps the level a separate path segment
    image = load_zarr(image_path, logger)

    brain_to_exaspim_transform_path = pipeline.register(
        parser=parser,
        zarr_image=image,
    )
    end_date_time = datetime.now(timezone.utc)

    records.append(
        make_data_process(
            process_type="Image atlas alignment",
            name="Image atlas alignment - 25 um",
            start=start_date_time,
            end=end_date_time,
            code_url=code_url,
            code_name="aind-exaspim-ccf-registration",
            code_version=__version__,
            parameters={
                "input_uri": dataset_path,
                "level": level,
                "resolution_um": resolution,
                "dataset_id": dataset_id,
                # use_fused_mask = whether the mask was requested (toggle);
                # mask_applied = whether the mask was actually used (the fused-mask
                # nii exists only when the mask was multiplied in).
                "use_fused_mask": os.environ.get("USE_FUSED_MASK", "true").strip().lower()
                                  not in ("0", "false", "no", "off", "none", "skip", "n", "f", ""),
                "mask_applied": os.path.exists(f"{outprefix}{dataset_id}_fused_mask.nii.gz"),
                "sample_to_template_registration": example_input["reg_param_25um"],
                "template_to_ccf_transforms": exaspim_to_ccf_transform_path,
            },
            experimenters=["Peter Grotz"],
            output_path="ccf_alignment/",
            notes="Import + template-based registration (25 um): sample -> template -> CCF",
        )
    )

    # QC overlay: warp the fused brain mask into template space and overlay it
    # on the template mask. Written under ccf_alignment/mask_qc/ (not uploaded
    # to S3). Skipped when no fused mask is present or on error.
    try:
        fused_mask_nii = f"{outprefix}{dataset_id}_fused_mask.nii.gz"
        template_mask_nii = "../data/exaSPIM_template_mask_25um_otsu.nii.gz"
        if os.path.exists(fused_mask_nii) and os.path.exists(template_mask_nii):
            from aind_exaspim_ccf_reg.plots import plot_mask_overlay
            tmpl_mask = ants.image_read(template_mask_nii)
            fused_mask = ants.image_read(fused_mask_nii)
            # nearestNeighbor keeps the warped mask binary.
            warped_mask = ants.apply_transforms(
                fixed=tmpl_mask,
                moving=fused_mask,
                transformlist=brain_to_exaspim_transform_path,
                interpolator="nearestNeighbor",
            )
            qc_dir = f"{outprefix_reg}mask_qc/"
            create_folder(dest_dir=qc_dir, verbose=False)
            qc_png = f"{qc_dir}{dataset_id}_fused_mask_vs_template_mask.png"
            dice = plot_mask_overlay(
                warped_mask,
                tmpl_mask,
                figpath=qc_png,
                title=f"{dataset_id}: fused brain mask (warped to template) vs exaSPIM template mask",
                label_a="fused mask -> template",
                label_b="template mask",
            )
            logger.info(f"Wrote mask QC overlay (results only, not uploaded): {qc_png} (Dice={dice})")
        else:
            logger.info("Skipping mask QC overlay: no fused mask present for this run.")
    except Exception as exc:
        logger.info(f"Mask QC overlay failed (non-fatal): {exc}")

    if level in [3, 6]:
        level = {3: 2, 6: 5}.get(level, level)    
        resolution = 10

        # Load image and apply transforms at 10 um, recorded as one process.
        start_date_time = datetime.now(timezone.utc)
        image_path = f"{fused_path.rstrip('/')}/{level}"  # rstrip keeps the level a separate path segment
        image = load_zarr(image_path, logger)

        output_path = pipeline.apply_transforms_to_10um(
            parser=parser,
            zarr_image=image,
            brain_to_exaspim_transform_path=brain_to_exaspim_transform_path,
            dataset_id=f"{dataset_id}_10um",
        )
        end_date_time = datetime.now(timezone.utc)
        records.append(
            make_data_process(
                process_type="Image atlas alignment",
                name="Image atlas alignment - 10 um",
                start=start_date_time,
                end=end_date_time,
                code_url=code_url,
                code_name="aind-exaspim-ccf-registration",
                code_version=__version__,
                parameters={
                    "input_uri": dataset_path,
                    "level": level,
                    "resolution_um": resolution,
                    "dataset_id": f"{dataset_id}_10um",
                    "sample_to_template_registration": example_input["reg_param_10um"],
                    "sample_to_template_transforms": brain_to_exaspim_transform_path,
                    "template_to_ccf_transforms": exaspim_to_ccf_transform_path,
                },
                experimenters=["Peter Grotz"],
                output_path="ccf_alignment/",
                notes=f"Import + 10um sample->CCF registration using transforms: {brain_to_exaspim_transform_path} and {exaspim_to_ccf_transform_path}",
            )
        )

    paths = write_data_processes(records, outprefix_reg)
    logger.info(f"Wrote {len(paths)} data_process.json files (upload capsule builds processing.json)")

    end_time = datetime.now(timezone.utc)
    logger.info(f"Finish all steps, execution time: {end_time - start_time} s")

    # NOTE: the SmartSheet "CCF Registered" write is deliberately NOT done here.
    # main.py is only the first step of code/run -- the annotation inversion, metadata
    # finalization and mesh export all run after it. Marking the sheet here would flag
    # a sample registered while those later steps can still fail. code/run invokes
    # `main.py --smartsheet-only` as its final step instead.


    # s3_reg_path = get_root_s3_prefix(dataset_path)
    # print(f"Upload reg to {s3_reg_path}")
    # print(f"folder_to_upload: {outprefix_reg}")
    
    # upload_alignment_data(
    #     s3_reg_path,
    #     outprefix_reg,
    # )


def mark_smartsheet() -> None:
    """Mark the sample CCF Registered. Run as the LAST step of code/run, once the
    annotation inversion, metadata finalization and mesh export have all completed.

    The subject id is recovered from the acquisition_<id>.json main() wrote, the same
    way the shell steps recover it, so all of them agree without re-deriving it from
    the asset name. Never raises: a SmartSheet problem must not fail a good run.
    """
    token = os.environ.get("SMARTSHEET_TOKEN")
    if not token:
        print("SMARTSHEET_TOKEN not set; skipping SmartSheet update.")
        return
    meta_dir = Path("/results/ccf_alignment/registration_metadata")
    found = sorted(meta_dir.glob("acquisition_*.json"))
    dataset_id = found[0].stem[len("acquisition_"):] if found else None
    if not dataset_id:
        print(f"no acquisition_<id>.json under {meta_dir}; skipping SmartSheet update.")
        return
    try:
        update_smartsheet(dataset_id, token)
        print(f"SmartSheet: marked CCF Registered for {dataset_id}")
    except Exception as e:
        print(f"SmartSheet update failed (non-fatal): {e}")


def update_smartsheet(brain_id, access_token):
    from aind_exaspim_dataset_utils.smartsheet_util import SmartSheetClient

    class PaginatedSmartSheetClient(SmartSheetClient):
        # list_sheets() returns only the first 100 sheets by default;
        # include_all=True returns every sheet so the name lookup is reliable.
        def find_sheet_id(self):
            response = self.client.Sheets.list_sheets(include_all=True)
            for sheet in response.data:
                if sheet.name == self.sheet_name:
                    return sheet.id
            raise Exception(f"Sheet Not Found - sheet_name={self.sheet_name}")

    client = PaginatedSmartSheetClient(access_token, "ExM Dataset Summary")
    column_map = {col.title: col.id for col in client.sheet.columns}
    row = client.client.models.Row()
    row.id = client.find_row_id(brain_id)
    row.cells.append({"column_id": column_map.get("CCF Registered"), "value": True, "strict": False})
    row.cells.append({"column_id": column_map.get("Affine Registration Date"),
                      "value": datetime.today().strftime("%m/%d/%Y"), "strict": False})
    client.client.Sheets.update_rows(client.sheet_id, [row])


if __name__ == "__main__":
    # --smartsheet-only marks the sheet and exits; code/run calls it last so the
    # sample is only flagged registered once every step has succeeded.
    if "--smartsheet-only" in sys.argv:
        mark_smartsheet()
    else:
        main()

    # parser = argparse.ArgumentParser(description='CCF registration')
    # parser.add_argument('dataset_path', help='S3 path to the ccf channel fusion')
    # parser.add_argument('level', default=3, help='level of input data')
    # parser.add_argument('resolution', default=25, help='resolution')
    # args = parser.parse_args()

    # dataset_path=args.dataset_path
    # level=args.level
    # resolution=args.resolution
    # print(f"dataset_path: {dataset_path}")
    # print(f"level: {level}")
    # print(f"resolution: {resolution}")

    # main(dataset_path, level, resolution)

    # dataset_path = str(dataset_config["pipeline_processing"]["registration"]["alignment_channel_path"])
    # level = int(str(dataset_config["pipeline_processing"]["registration"]["level"]))
    # resolution = int(str(dataset_config["pipeline_processing"]["registration"]["resolution"]))
    