"""
Main script for exaspim-to-template-to-CCF registration pipeline.

This module contains the main functions for performing CCF (Common Coordinate Framework)
registration of exaSPIM data to the Allen Mouse Brain Atlas.
"""

import json
import logging
import os
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
code_url = "https://github.com/AllenNeuralDynamics/aind-exaspim-ccf-registration.git"


def _apply_output_prefix(path: str) -> str:
    """Test mode: when OUTPUT_PREFIX is set, read the fused image + mask from
    where the fusion capsule wrote them ({OUTPUT_PREFIX}/<asset>/fusion/...),
    not the production asset. Acquisition metadata is still read from the
    original (aind-open-data) path. No effect when OUTPUT_PREFIX is unset."""
    prefix = os.environ.get("OUTPUT_PREFIX")
    if not prefix or "/fusion/" not in path:
        return path
    in_base, rest = path.split("/fusion/", 1)
    asset = in_base.rstrip("/").split("/")[-1]
    return f"{prefix.rstrip('/')}/{asset}/fusion/{rest}"

def load_zarr(
    image_path: PathLike, 
    logger: logging.Logger
) -> np.ndarray:
    """
    Load Zarr image.
    """
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
    """
    generate output meta data, processing.json
    Copies results to the destination bucket to make it available
    to scientists as soon as possible.

    Parameters
    ----------
    s3_path: str
        New dataset name where the data will
        be copied following the aind conventions
        e.g., s3://{bucket_path}/{new_dataset_name}

    folder_to_upload: PathLike
        Results folder path in Code Ocean

    Returns
    -------
    Tuple[str, str]
        The first position is the path where the dataset
        was moved. e.g., s3://{bucket_path}/{new_dataset_name}
        It includes the "s3://" prefix. 
        e.g., s3://{bucket_path}/{new_dataset_name}/{output_prediction}
    """

    #------------------------------------#
    # upload alignment results to s3
    #------------------------------------#
    # s3_path = f"s3://{bucket_path}/{new_dataset_name}"
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
    # Remove 's3://' and split path
    scheme, bucket_and_key = s3_uri.split('://', 1)
    bucket, *key_parts = bucket_and_key.split('/')
  
    # Go `levels_up` directories up from the current file path
    base_key = '/'.join(key_parts[:levels_up])
    return f's3://{bucket}/{base_key}/'

def main() -> None:
    """
    Main function to run the CCF registration pipeline.
    
    This function orchestrates the entire registration process:
    1. Loads configuration from processing manifest
    2. Sets up output directories and logging
    3. Performs registration at the specified resolution level
    4. Optionally applies transforms to 10um resolution
    5. Generates processing metadata
    """
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
    # Where to READ the fused image + mask (test mode redirects to OUTPUT_PREFIX,
    # where the fusion capsule wrote them; production = the asset path itself).
    fused_path = _apply_output_prefix(dataset_path)
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
        "bucket_path": "aind-scratch-data",
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

    #-----------------------------------------------#
    # Load OMEZarr image 
    #-----------------------------------------------#

    start_date_time = datetime.now(timezone.utc)
    image_path = f"{fused_path}{level}"
    image = load_zarr(image_path, logger)
    end_date_time = datetime.now(timezone.utc)

    records.append(
        make_data_process(
            process_type="Image importing",
            name="Image importing - 25 um",
            start=start_date_time,
            end=end_date_time,
            code_url=code_url,
            code_name="aind-exaspim-ccf-registration",
            code_version=__version__,
            parameters={
                "input_uri": dataset_path,
                "level": level,
                "resolution_um": resolution,
            },
            experimenters=["Peter Grotz"],
            notes="Importing fused data for alignment",
        )
    )

    #-----------------------------------------------#
    # registration
    #-----------------------------------------------#
    start_date_time = datetime.now(timezone.utc)

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
                "level": level,
                "resolution_um": resolution,
                "dataset_id": dataset_id,
                "sample_to_template_registration": example_input["reg_param_25um"],
                "template_to_ccf_transforms": exaspim_to_ccf_transform_path,
            },
            experimenters=["Peter Grotz"],
            output_path="ccf_alignment/",
            notes="Template based registration: sample -> template -> CCF",
        )
    )

    if level in [3, 6]:
        level = {3: 2, 6: 5}.get(level, level)    
        resolution = 10

        #-----------------------------------------------#
        # load zarr
        #-----------------------------------------------#
        start_date_time = datetime.now(timezone.utc)
        image_path = f"{fused_path}{level}"
        image = load_zarr(image_path, logger)    
        end_date_time = datetime.now(timezone.utc)
        records.append(
            make_data_process(
                process_type="Image importing",
                name="Image importing - 10 um",
                start=start_date_time,
                end=end_date_time,
                code_url=code_url,
                code_name="aind-exaspim-ccf-registration",
                code_version=__version__,
                parameters={
                    "input_uri": dataset_path,
                    "level": level,
                    "resolution_um": resolution,
                },
                experimenters=["Peter Grotz"],
                notes="Importing fused data for alignment",
            )
        )

        #-----------------------------------------------#
        # apply transforms to 10um
        #-----------------------------------------------#    
        start_date_time = datetime.now(timezone.utc)
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
                    "level": level,
                    "resolution_um": resolution,
                    "dataset_id": f"{dataset_id}_10um",
                    "sample_to_template_registration": example_input["reg_param_10um"],
                    "sample_to_template_transforms": brain_to_exaspim_transform_path,
                    "template_to_ccf_transforms": exaspim_to_ccf_transform_path,
                },
                experimenters=["Peter Grotz"],
                output_path="ccf_alignment/",
                notes=f"Registering 10um sample to CCF using transforms: {brain_to_exaspim_transform_path} and {exaspim_to_ccf_transform_path}",
            )
        )

    paths = write_data_processes(records, outprefix_reg)
    logger.info(f"Wrote {len(paths)} data_process.json files (upload capsule builds processing.json)")

    end_time = datetime.now(timezone.utc)    
    logger.info(f"Finish all steps, execution time: {end_time - start_time} s")


    # s3_reg_path = get_root_s3_prefix(dataset_path)
    # print(f"Upload reg to {s3_reg_path}")
    # print(f"folder_to_upload: {outprefix_reg}")
    
    # upload_alignment_data(
    #     s3_reg_path,
    #     outprefix_reg,
    # )


if __name__ == "__main__":
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
    