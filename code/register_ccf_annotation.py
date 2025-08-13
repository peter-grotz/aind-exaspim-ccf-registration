#!/usr/bin/env python3
"""
Register CCF annotation to sample space and create segmentation mask.

This script takes CCF annotation and registers it through template space to sample space,
applies reverse orientation, and creates a multiscale segmentation mask.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import ants
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from dask.distributed import Client, LocalCluster
from urllib.parse import urlparse

# Import local modules
from aind_exaspim_ccf_reg.preprocess import get_adjustments, adjust_array
from aind_exaspim_ccf_reg.utils import create_logger


def adjust_array_reverse(arr: np.ndarray, swaps: List[Tuple[int, int]], flips: List[int]) -> np.ndarray:
    """
    Reverse orientation adjustment by applying flips first, then swaps.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to adjust
    swaps : List[Tuple[int, int]]
        List of axis swaps to reverse
    flips : List[int]
        List of axes to flip
        
    Returns
    -------
    np.ndarray
        Reoriented array
    """
    if flips:
        arr = np.flip(arr, axis=flips)
    if swaps:
        in_axis, out_axis = zip(*swaps)
        arr = np.moveaxis(arr, in_axis, out_axis)
    return arr


def show_overlay(base_img, overlay_img, title, slice_idx=None, alpha=0.3, save_path=None):
    """Show overlay visualization in three orthogonal directions."""
    if slice_idx is None:
        # Use middle slices for each direction
        sagittal_idx = base_img.shape[0] // 2
        coronal_idx = base_img.shape[1] // 2
        axial_idx = base_img.shape[2] // 2
    else:
        sagittal_idx = coronal_idx = axial_idx = slice_idx
    
    # Get slices for each direction
    base_sagittal = base_img.numpy()[sagittal_idx, :, :]
    overlay_sagittal = overlay_img.numpy()[sagittal_idx, :, :]
    
    base_coronal = base_img.numpy()[:, coronal_idx, :]
    overlay_coronal = overlay_img.numpy()[:, coronal_idx, :]
    
    base_axial = base_img.numpy()[:, :, axial_idx]
    overlay_axial = overlay_img.numpy()[:, :, axial_idx]
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sagittal view (XZ plane)
    axes[0].imshow(base_sagittal, cmap='gray')
    axes[0].imshow(np.ma.masked_where(overlay_sagittal == 0, overlay_sagittal), 
                   cmap='bwr', alpha=alpha)
    axes[0].set_title(f'{title} - Sagittal')
    axes[0].axis('off')
    
    # Coronal view (YZ plane)
    axes[1].imshow(base_coronal, cmap='gray')
    axes[1].imshow(np.ma.masked_where(overlay_coronal == 0, overlay_coronal), 
                   cmap='bwr', alpha=alpha)
    axes[1].set_title(f'{title} - Coronal')
    axes[1].axis('off')
    
    # Axial view (XY plane)
    axes[2].imshow(base_axial, cmap='gray')
    axes[2].imshow(np.ma.masked_where(overlay_axial == 0, overlay_axial), 
                   cmap='bwr', alpha=alpha)
    axes[2].set_title(f'{title} - Axial')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure if save_path is provided, otherwise show
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Register CCF annotation to sample space')
    
    # Input paths
    parser.add_argument('--ccf_annotation_path', required=True,
                       help='Path to CCF annotation file')
    parser.add_argument('--ccf_template_path', required=True,
                       help='Path to CCF template file')
    parser.add_argument('--exaspim_template_path', required=True,
                       help='Path to exaSPIM template file')
    parser.add_argument('--resampled_image_path', required=True,
                       help='Path to resampled image file')
    parser.add_argument('--sample_image_path', required=True,
                       help='Path to sample image file')
    
    # Transform paths
    parser.add_argument('--ccf_to_template_transforms', nargs='+', required=True,
                       help='List of transform files from CCF to template space')
    parser.add_argument('--template_to_sample_transforms', nargs='+', required=True,
                       help='List of transform files from template to sample space')
    
    # Acquisition metadata
    parser.add_argument('--acquisition_path', required=True,
                       help='Path to acquisition metadata JSON file')
    
    # Dataset information
    parser.add_argument('--dataset_path', required=True,
                       help='S3 path to dataset')
    parser.add_argument('--level', type=int, default=3,
                       help='Zarr level to use (default: 3)')
    
    # Output settings
    parser.add_argument('--seg_path', default='/results/',
                       help='Path for segmentation output (default: /results/)')
    parser.add_argument('--bucket_path', default='aind-scratch-data',
                       help='S3 bucket for upload (default: aind-scratch-data)')
    parser.add_argument('--new_dataset_name', required=True,
                       help='Name for the new dataset in S3')
    
    # Visualization
    parser.add_argument('--show_visualizations', action='store_true',
                       help='Show overlay visualizations')
    
    args = parser.parse_args()
    logger = create_logger(output_log_path=args.seg_path)
    
    logger.info("Loading images...")
    # Load images
    ccf_annotation = ants.image_read(args.ccf_annotation_path)
    ccf_template = ants.image_read(args.ccf_template_path)
    exaspim_template = ants.image_read(args.exaspim_template_path)
    resampled_image = ants.image_read(args.resampled_image_path)
    sample_image = ants.image_read(args.sample_image_path)
    
    logger.info("Image shapes:")
    logger.info(f"CCF annotation: {ccf_annotation}")
    logger.info(f"CCF template: {ccf_template}")
    logger.info(f"exaSPIM template: {exaspim_template}")
    logger.info(f"Resampled image: {resampled_image}")
    logger.info(f"Sample image: {sample_image}")
    
    logger.info("Applying transforms...")
    logger.info(f"Applying ccf_to_template_transforms: {args.ccf_to_template_transforms}")
    if "/" == args.dataset_path[-1]:
        dataset_path = args.dataset_path[:-1]
    logger.info(f"dataset_path: {dataset_path}")
    # Apply transforms: CCF annotation to template space
    annotation_in_template = ants.apply_transforms(
        fixed=exaspim_template,
        moving=ccf_annotation,
        transformlist=args.ccf_to_template_transforms,
        interpolator='nearestNeighbor',
        whichtoinvert=[True, False]
    )
    
    logger.info(f"Applying template_to_sample_transforms: {args.template_to_sample_transforms}")
    # Apply transforms: Template to sample space
    annotation_in_resampled_image = ants.apply_transforms(
        fixed=resampled_image,
        moving=annotation_in_template,
        transformlist=args.template_to_sample_transforms,
        interpolator='nearestNeighbor',
        whichtoinvert=[True, False]
    )
    
    logger.info("Resampling to sample space...")
    # Resample to sample image space
    annotation_in_sample = ants.resample_image_to_target(
        image=annotation_in_resampled_image,
        target=sample_image,
        interp_type='nearestNeighbor'
    )
    
    if args.show_visualizations:
        logger.info("Saving visualizations...")
        # Create output directory for figures
        fig_dir = os.path.join(args.seg_path, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        
        show_overlay(ccf_template, ccf_annotation, "CCF Annotation on CCF", 
                    save_path=os.path.join(fig_dir, "ccf_annotation_on_ccf.png"))
        show_overlay(exaspim_template, annotation_in_template, "CCF Annotation on exaSPIM Template", 
                    save_path=os.path.join(fig_dir, "ccf_annotation_on_exaspim_template.png"))
        show_overlay(resampled_image, annotation_in_resampled_image, "CCF Annotation on resampled image", 
                    save_path=os.path.join(fig_dir, "ccf_annotation_on_resampled_image.png"))
        show_overlay(sample_image, annotation_in_sample, "CCF Annotation on sample image", 
                    save_path=os.path.join(fig_dir, "ccf_annotation_on_sample_image.png"))
    
    logger.info("Applying reverse orientation...")
    # Load metadata and get swaps/flips
    with open(args.acquisition_path, "r") as f:
        metadata = json.load(f)
        if "tile_000000_ch_" in metadata["tiles"][0]["file_name"]:
            ccf_directions = {
                0: "Anterior_to_posterior",
                1: "Superior_to_inferior",
                2: "Left_to_right",
            }
        else:
            ccf_directions = {
                0: "Posterior_to_anterior",
                1: "Inferior_to_superior",
                2: "Left_to_right",
            }
    
    swaps, flips = get_adjustments(metadata['axes'], ccf_directions)
    logger.info(f"Original swaps: {swaps}, flips: {flips}")
    
    # Invert swaps and flips
    inv_swaps = [(b, a) for (a, b) in reversed(swaps)]
    inv_flips = flips
    logger.info(f"Inverse swaps: {inv_swaps}, flips: {inv_flips}")
    
    # Apply reverse orientation
    anno_np = annotation_in_sample.numpy()
    anno_np = adjust_array_reverse(anno_np, inv_swaps, inv_flips)
    annotation_in_sample_reoriented = ants.from_numpy(anno_np.astype(np.uint8))
    ants.image_write(annotation_in_sample_reoriented, f"{args.seg_path}ccf_anno_in_sample_space.nii.gz")
    
    logger.info("Loading original sample data...")
    # Load original sample data
    image_path = f"{dataset_path}/{args.level}"
    logger.info(f"Loading from: {image_path}")
    
    try:
        image = zarr.open(image_path, mode="r")
        image = np.squeeze(np.squeeze(np.array(image), axis=0), axis=0)
        logger.info(f"Original image shape: {image.shape}")
        logger.info(f"Annotation shape: {annotation_in_sample_reoriented.shape}")
        ants_image = ants.from_numpy(image.astype(np.uint8))
        ants.image_write(ants_image, f"{args.seg_path}sample.nii.gz")

        if args.show_visualizations:
            show_overlay(ants_image, 
                        annotation_in_sample_reoriented, 
                        "CCF Annotation on original sample image",
                        save_path=os.path.join(fig_dir, "ccf_anno_in_sample_space.png"))
    except Exception as e:
        logger.info(f"Warning: Could not load original sample data: {e}")
    
    logger.info("Creating segmentation mask...")
    # Import upscale_mask modules (assuming they exist)
    try:
        from upscale_mask import utils, upscale_mask
        
        # Get image metadata
        image_metadata = utils.load_json(data_path=dataset_path, keyname=".zattrs")
        scale = str(args.level)
        image_metadata = utils.parse_zarr_metadata(metadata=image_metadata, multiscale=scale)
        
        # Calculate resolution
        current_res = (
            image_metadata["axes"]["z"]["scale"],
            image_metadata["axes"]["y"]["scale"],
            image_metadata["axes"]["x"]["scale"],
        )
        logger.info(f"Current resolution: {current_res}")
        
        # Target resolution
#         logger.info(np.array(current_res) / (2 ** (int(scale) + 1)))
#         # target_res = tuple(np.array(current_res) / (2 ** (int(scale) + 1)).tolist())
        
        target_res = (
            np.array(current_res) / ( 2 ** ( int(scale) + 1) )
        ).tolist()
        target_res = tuple(target_res)
        logger.info(f"Target resolution: {target_res}")
        
        upscale_factors_zyx = (
            (current_res[0] / target_res[0]) / 2,
            (current_res[1] / target_res[1]) / 2,
            (current_res[2] / target_res[2]) / 2,
        )
        
        # Create segmentation mask
        voxel_size, n_lvls = upscale_mask.upscale_mask(
            dataset_path=dataset_path,
            mask_data=annotation_in_sample_reoriented,
            upscale_factors_zyx=upscale_factors_zyx,
            output_folder=args.seg_path,
            filename="ccf_anno_in_sample_space.zarr",
            dest_multiscale="0",
        )
        
        logger.info(f"Creating {n_lvls} levels in the pyramid.")
        
        # Write multiscales
        cluster = LocalCluster()
        client = Client(cluster)
        
        upscale_mask.write_multiscales(
            path_to_data=f"{args.seg_path}/ccf_anno_in_sample_space.zarr",
            chunk_size=[128, 128, 128],
            scale_factor=[2, 2, 2],
            target_size_mb=1024,
            n_lvls=n_lvls - 1,
            root_group=None,
            voxel_size=voxel_size,
        )
        
        client.close()
        cluster.close()

        # logger.info("Uploading to S3...")
        # # Upload to S3
        # s3_path = f"s3://{args.bucket_path}/{args.new_dataset_name}"
        # logger.info(f"Uploading to: {s3_path}")

        # fs = s3fs.S3FileSystem()
        # url = urlparse(s3_path)
        
        # if url.scheme != "s3":
        #     raise NotImplementedError(f"Only s3 output_uri is supported, not {url.scheme}")
        
        # file_to_be_upload = f"{args.seg_path}/ccf_anno_in_sample_space.zarr"
        # logger.info(f"Uploading: {file_to_be_upload}")
        
        # fs.put(
        #     file_to_be_upload, 
        #     url.netloc + url.path.rstrip("/") + "/", 
        #     recursive=True, 
        #     maxdepth=10
        # )
        
        # logger.info("Upload complete!")
        
    except ImportError as e:
        logger.info(f"Warning: Could not import upscale_mask modules: {e}")
        logger.info("Skipping segmentation mask creation and S3 upload.")
    except Exception as e:
        logger.info(f"Error during segmentation mask creation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 