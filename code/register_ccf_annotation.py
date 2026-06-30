#!/usr/bin/env python3
"""
Register CCF annotation to sample space and create a multiscale segmentation mask.

Transforms the CCF annotation through template space into sample space, applies the
reverse orientation adjustment, and writes zarr and precomputed label volumes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from typing import List, Tuple

import ants
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from dask.distributed import Client, LocalCluster
from urllib.parse import urlparse

from aind_exaspim_ccf_reg.preprocess import get_adjustments, adjust_array
from aind_exaspim_ccf_reg.utils import create_logger, create_folder, read_json_as_dict
from aind_exaspim_ccf_reg.register import ZarrWriter

from upscale_mask import utils, upscale_mask

from cloudvolume import CloudVolume
import scipy.ndimage as ndi
import dask.array as da
from numcodecs import blosc
blosc.use_threads = False


class create_precomputed:
    def __init__(self, ng_params):
        self.regions = ng_params["regions"]
        self.scaling = ng_params["scale_params"]
        self.save_path = ng_params["save_path"]

    def save_json(self, fpath: str, info: dict):
        """
        Write an info JSON for the precomputed format.

        Parameters
        ----------
        fpath: str
            directory where the info file is written
        info: dict
            data to write
        """

        path = f"{fpath}/info"

        with open(path, "w") as fp:
            json.dump(info, fp, indent=2)

        return

    def create_segmentation_info(self):
        """Write the segment_properties info file mapping CCFv3 region IDs to labels."""

        json_data = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(k) for k in self.regions.keys()],
                "properties": [
                    {
                        "id": "label",
                        "type": "label",
                        "values": [str(v) for k, v in self.regions.items()],
                    }
                ],
            },
        }

        fpath = f"{self.save_path}/segment_properties"
        self.save_json(fpath, json_data)

        return

    def build_scales(self):
        """
        Build the per-scale entries for the segmentation precomputed info file.

        Return
        ------
        scales: dict
            resolution scales of the precomputed pyramid
        """

        scales = []
        for s in range(self.scaling["num_scales"]):
            scale = {
                "chunk_sizes": [self.scaling["chunk_size"]],
                "encoding": self.scaling["encoding"],
                "compressed_segmentation_block_size": self.scaling[
                    "compressed_block"
                ],
                "key": "_".join(
                    [
                        str(int(r * f**s))
                        for r, f in zip(
                            self.scaling["res"], self.scaling["factors"]
                        )
                    ]
                ),
                "resolution": [
                    int(r * f**s)
                    for r, f in zip(
                        self.scaling["res"], self.scaling["factors"]
                    )
                ],
                "size": [
                    int(d // f**s)
                    for d, f in zip(
                        self.scaling["dims"], self.scaling["factors"]
                    )
                ],
            }
            scales.append(scale)

        return scales

    def build_precomputed_info(self):
        """
        Build and write the top-level segmentation precomputed info file.

        Returns
        -------
        info: dict
            the info dictionary written to disk
        """
        info = {
            "type": "segmentation",
            "segment_properties": "segment_properties",
            "data_type": "uint32",
            "num_channels": 1,
            "scales": self.build_scales(),
        }

        self.save_json(self.save_path, info)

        return info

    def volume_info(self, scale: int, shape: tuple):
        """
        Build the CloudVolume info for a single scale level.

        Returns
        -------
        info: Cloudvolume Object
            scaling information for one pyramid level
        """

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type="segmentation",
            data_type="uint32",
            encoding=self.scaling["encoding"],
            resolution=[
                int(r * f**scale)
                for r, f in zip(self.scaling["res"], self.scaling["factors"])
            ],
            voxel_offset=[0, 0, 0],
            chunk_size=self.scaling["chunk_size"],
            volume_size=[dim for dim in shape],
        )

        return info

    def create_segment_precomputed(self, img: np.array):
        """
        Build the segmentation precomputed pyramid and write its files.

        Parameters
        ----------
        img: np.array
            label image to convert to precomputed format
        """

        for scale in range(self.scaling["num_scales"]):
            if scale == 0:
                curr_img = img
            else:
                factor = [1 / 2**scale for d in img.shape]
                curr_img = ndi.zoom(img, tuple(factor), order=0)

            info = self.volume_info(scale, curr_img.shape)
            vol = CloudVolume(
                f"file://{self.save_path}", info=info, compress=False
            )
            vol[:, :, :] = curr_img.astype("uint32")

        return

    def cleanup_seg_files(self):
        files = glob(f"{self.save_path}/**/*.br", recursive=True)

        for file in files:
            new_file = file[:-3]
            os.rename(file, new_file)

        return

def create_uint32_reference_image(reference_image: ants.ANTsImage) -> ants.ANTsImage:
    """Return an empty uint32 image matching the reference image geometry."""
    label_reference = ants.from_numpy(
        np.zeros(reference_image.shape, dtype=np.uint32)
    )
    label_reference.set_spacing(reference_image.spacing)
    label_reference.set_origin(reference_image.origin)
    label_reference.set_direction(reference_image.direction)
    return label_reference


def apply_label_transforms(
    fixed_reference: ants.ANTsImage,
    moving_labels: ants.ANTsImage,
    transformlist: List[str],
    whichtoinvert: List[bool],
) -> ants.ANTsImage:
    """Apply ANTs transforms to a label image using GenericLabel interpolation at double precision."""
    return ants.apply_transforms(
        fixed=fixed_reference,
        moving=moving_labels,
        transformlist=transformlist,
        interpolator="genericLabel",
        whichtoinvert=whichtoinvert,
        singleprecision=False,
    )


def resample_label_image_to_target_with_ants(
    moving_labels: ants.ANTsImage,
    target_reference: ants.ANTsImage,
    work_dir: str,
) -> ants.ANTsImage:
    """Resample a label image onto a target grid via the antsApplyTransforms CLI (GenericLabel)."""
    ants_apply_transforms = shutil.which("antsApplyTransforms")
    if ants_apply_transforms is None:
        raise RuntimeError(
            "antsApplyTransforms was not found on PATH. "
            "Install the ANTs CLI package in the Docker image."
        )

    with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir:
        moving_path = os.path.join(tmpdir, "moving_labels.nii.gz")
        reference_path = os.path.join(tmpdir, "target_reference.nii.gz")
        output_path = os.path.join(tmpdir, "resampled_labels.nii.gz")

        ants.image_write(moving_labels, moving_path)
        ants.image_write(target_reference, reference_path)

        cmd = [
            ants_apply_transforms,
            "-d", str(moving_labels.dimension),
            "-i", moving_path,
            "-r", reference_path,
            "-o", output_path,
            "-n", "GenericLabel",
            "-t", "identity",
            "--float", "0",
        ]
        subprocess.run(cmd, check=True)

        return ants.image_read(output_path, pixeltype="unsigned int")


def get_estimated_downsample(
    voxel_resolution: List[float],
    registration_res: Tuple[float] = (16.0, 14.4, 14.4),
) -> int:
    """
    Return the multiscale level whose resolution is closest to registration_res.

    Parameters
    ----------
    voxel_resolution: List[float]
        original image resolution (multiscale level "0")
    registration_res: Tuple[float]
        resolution used for registration. Default: (16.0, 14.4, 14.4)
    """

    downsample_versions = []
    for idx in range(len(voxel_resolution)):
        downsample_versions.append(
            registration_res[idx] // float(voxel_resolution[idx])
        )

    downsample_res = int(min(downsample_versions))
    return round(np.log2(downsample_res))



def adjust_array_reverse(arr: np.ndarray, swaps: List[Tuple[int, int]], flips: List[int]) -> np.ndarray:
    """
    Reverse an orientation adjustment by applying flips first, then swaps.

    Parameters
    ----------
    arr : np.ndarray
        array to adjust
    swaps : List[Tuple[int, int]]
        axis swaps to apply
    flips : List[int]
        axes to flip

    Returns
    -------
    np.ndarray
        reoriented array
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
        # middle slice in each direction
        sagittal_idx = base_img.shape[0] // 2
        coronal_idx = base_img.shape[1] // 2
        axial_idx = base_img.shape[2] // 2
    else:
        sagittal_idx = coronal_idx = axial_idx = slice_idx

    # extract slices for each direction
    base_sagittal = base_img.numpy()[sagittal_idx, :, :]
    overlay_sagittal = overlay_img.numpy()[sagittal_idx, :, :]
    
    base_coronal = base_img.numpy()[:, coronal_idx, :]
    overlay_coronal = overlay_img.numpy()[:, coronal_idx, :]
    
    base_axial = base_img.numpy()[:, :, axial_idx]
    overlay_axial = overlay_img.numpy()[:, :, axial_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # sagittal view (XZ plane)
    axes[0].imshow(base_sagittal, cmap='gray')
    axes[0].imshow(np.ma.masked_where(overlay_sagittal == 0, overlay_sagittal), 
                   cmap='bwr', alpha=alpha)
    axes[0].set_title(f'{title} - Sagittal')
    axes[0].axis('off')
    
    # coronal view (YZ plane)
    axes[1].imshow(base_coronal, cmap='gray')
    axes[1].imshow(np.ma.masked_where(overlay_coronal == 0, overlay_coronal), 
                   cmap='bwr', alpha=alpha)
    axes[1].set_title(f'{title} - Coronal')
    axes[1].axis('off')
    
    # axial view (XY plane)
    axes[2].imshow(base_axial, cmap='gray')
    axes[2].imshow(np.ma.masked_where(overlay_axial == 0, overlay_axial), 
                   cmap='bwr', alpha=alpha)
    axes[2].set_title(f'{title} - Axial')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # save to save_path if given, otherwise display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Register CCF annotation to sample space')
    
    # input paths
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
    
    # transform paths
    parser.add_argument('--ccf_to_template_transforms', nargs='+', required=True,
                       help='List of transform files from CCF to template space')
    parser.add_argument('--template_to_sample_transforms', nargs='+', required=True,
                       help='List of transform files from template to sample space')
    
    # acquisition metadata
    parser.add_argument('--acquisition_path', required=True,
                       help='Path to acquisition metadata JSON file')

    # dataset information
    parser.add_argument('--dataset_path', required=True,
                       help='S3 path to dataset')
    parser.add_argument('--level', type=int, default=3,
                       help='Zarr level to use (default: 3)')
    
    # output settings
    parser.add_argument('--seg_path', default='/results/',
                       help='Path for segmentation output (default: /results/)')
    parser.add_argument('--new_dataset_name', required=True,
                       help='Name for the new dataset in S3')
    
    # visualization
    parser.add_argument('--show_visualizations', action='store_true',
                       help='Show overlay visualizations')
    
    args = parser.parse_args()
    logger = create_logger(output_log_path=args.seg_path)
    
    if "/" == args.dataset_path[-1]:
        dataset_path = args.dataset_path[:-1]
    logger.info(f"dataset_path: {dataset_path}")
    
    image_metadata = utils.load_json(data_path=dataset_path, keyname=".zattrs")
    print(f"image_metadata: {image_metadata}")
    
    #---------------------------------------------#

    # load images
    logger.info("Loading images...")

    ccf_annotation = ants.image_read(args.ccf_annotation_path, pixeltype='unsigned int')
    ccf_template = ants.image_read(args.ccf_template_path)
    exaspim_template = ants.image_read(args.exaspim_template_path)
    resampled_image = ants.image_read(args.resampled_image_path)
    sample_image = ants.image_read(args.sample_image_path)

    exaspim_template.set_spacing(ccf_template.spacing)
    exaspim_template.set_origin(ccf_template.origin)
    exaspim_template.set_direction(ccf_template.direction)
    
    logger.info("Image shapes:")
    logger.info(f"CCF annotation: {ccf_annotation}")
    logger.info(f"CCF template: {ccf_template}")
    logger.info(f"exaSPIM template: {exaspim_template}")
    logger.info(f"Resampled image: {resampled_image}")
    logger.info(f"Sample image: {sample_image}")

    exaspim_label_reference = create_uint32_reference_image(exaspim_template)
    resampled_label_reference = create_uint32_reference_image(resampled_image)
    sample_label_reference = create_uint32_reference_image(sample_image)

    #---------------------------------------------#
            
    logger.info("Applying transforms...")
    logger.info(f"Applying ccf_to_template_transforms: {args.ccf_to_template_transforms}")
    
    # CCF annotation -> exaSPIM template space
    annotation_in_template = apply_label_transforms(
        fixed_reference=exaspim_label_reference,
        moving_labels=ccf_annotation,
        transformlist=args.ccf_to_template_transforms,
        whichtoinvert=[True, False],
    )        
    
    logger.info(f"Applying template_to_sample_transforms: {args.template_to_sample_transforms}")
    annotation_in_resampled_image = apply_label_transforms(
        fixed_reference=resampled_label_reference,
        moving_labels=annotation_in_template,
        transformlist=args.template_to_sample_transforms,
        whichtoinvert=[True, False],
    )    
    
    logger.info("Resampling to sample space...")
    # resample onto the sample image grid
    annotation_in_sample = resample_label_image_to_target_with_ants(
        moving_labels=annotation_in_resampled_image,
        target_reference=sample_label_reference,
        work_dir=args.seg_path,
    )
    
    #---------------------------------------------#
    if args.show_visualizations:
        logger.info("Saving visualizations...")
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
    
    #--------------------------------------
    # reorient to the original image direction
    #---------------------------------------
    logger.info("Applying reverse orientation...")
    # read acquisition metadata to derive axis swaps/flips
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
    
    # invert the swaps and flips
    inv_swaps = [(b, a) for (a, b) in reversed(swaps)]
    inv_flips = flips
    logger.info(f"Inverse swaps: {inv_swaps}, flips: {inv_flips}")

    anno_np = annotation_in_sample.numpy()
    anno_np = adjust_array_reverse(anno_np, inv_swaps, inv_flips)
    annotation_in_sample_reoriented = ants.from_numpy(anno_np.astype(np.uint32))
    logger.info(f"CCF annotation in the original brain space: {annotation_in_sample_reoriented}")
    ants.image_write(annotation_in_sample_reoriented, f"{args.seg_path}ccf_anno_in_sample_space.nii.gz")
    
    # annotation_in_sample_reoriented = ants.image_read(f"{args.seg_path}ccf_anno_in_sample_space.nii.gz")
    # print(f"annotation_in_sample_reoriented: {annotation_in_sample_reoriented}")

    aligned_image = annotation_in_sample_reoriented.numpy()
    
    #--------------------------------------
    # load original zarr image
    #--------------------------------------
    logger.info("Loading original sample data...")
    image_path = f"{dataset_path}/{args.level}"
    logger.info(f"Loading from: {image_path}")
    
    try:
        image = zarr.open(image_path, mode="r")
        image = np.squeeze(np.squeeze(np.array(image), axis=0), axis=0)
        logger.info(f"Original image shape: {image.shape}")
        logger.info(f"Annotation shape: {annotation_in_sample_reoriented.shape}")
        ants_image = ants.from_numpy(image.astype(np.float32))
        logger.info(f"original brain image: {ants_image}")
        
        ants.image_write(ants_image, f"{args.seg_path}sample.nii.gz")

        if args.show_visualizations:
            show_overlay(ants_image, 
                        annotation_in_sample_reoriented, 
                        "CCF Annotation on original sample image",
                        save_path=os.path.join(fig_dir, "ccf_anno_in_sample_space.png"))
    except Exception as e:
        logger.info(f"Warning: Could not load original sample data: {e}")


    aligned_image_dask = da.from_array(aligned_image)

    logger.info(f"Before changing orientation: {aligned_image_dask.shape}, DR: {aligned_image.min()}, {aligned_image.max()}")

    aligned_image_dask = da.moveaxis(aligned_image_dask, [0, 1, 2], [2, 1, 0])
    logger.info(f"After changing orientation: {aligned_image_dask.shape}, DR: {aligned_image.min()}, {aligned_image.max()}, {aligned_image_dask.dtype}, {aligned_image.dtype}")

    params = {
        "OMEZarr_params": {
            "clevel": 1,
            "compressor": "zstd",
            "chunks": (64, 64, 64),
        },
        "metadata_folder": args.seg_path
    }

    opts = {
        "compressor": blosc.Blosc(
            cname=params["OMEZarr_params"]["compressor"],
            clevel=params["OMEZarr_params"]["clevel"],
            shuffle=blosc.SHUFFLE,
        )
    }
    image_name = "ccf_anno_in_sample_space.zarr"

    zarr_writer = ZarrWriter(logger)
    
    zarr_writer.write_zarr(
        img_array=aligned_image_dask,
        physical_pixel_sizes=(10, 10, 10),
        output_path=args.seg_path,
        image_name=image_name,
        opts=opts,
        params=params
    )
    #-------------------------------------#
   
    regions = read_json_as_dict(
        "./aind_exaspim_ccf_reg/ccf_files/annotation_map.json"
    )
    precompute_path = os.path.join(args.seg_path, "ccf_annotation_precomputed")
    create_folder(precompute_path)
    create_folder(f"{precompute_path}/segment_properties")
   
    # ---------------------------
    
    acquisition_res = image_metadata["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"][2:]
    logger.info(f"Image was acquired at resolution (um): {acquisition_res}")
    reg_scale = get_estimated_downsample(acquisition_res)
    logger.info(f"Image is being downsampled by a factor: {reg_scale}")
    reg_res = [(float(res) * 2**reg_scale) / 1000 for res in acquisition_res]
    logger.info(f"Registration resolution (mm): {reg_res}")
    spacing = tuple(reg_res)
    
    
    # -----------------------------
    ng_params = {
            "save_path": precompute_path,
            "regions": regions,
            "scale_params": {
                "encoding": "compresso",
                "compressed_block": [16, 16, 16],
                "chunk_size": [32, 32, 32],
                "factors": [2, 2, 2],
                "num_scales": 3,
            }
        }
    
     # precomputed expects XYZ order, not ZYX
    aligned_image_out = np.swapaxes(aligned_image, 0, 2)

    visual_spacing = tuple(
        [s * 10**6 for s in spacing[::-1]]
    )
    ng_params["scale_params"]["res"] = visual_spacing
    ng_params["scale_params"]["dims"] = [
        dim for dim in aligned_image.shape
    ]
    logger.info("-----"*10)
    logger.info(f"ng_params: {ng_params}")
    logger.info("-----"*10)

    seg = create_precomputed(ng_params)
    seg.create_segmentation_info()
    seg.build_precomputed_info()
    seg.create_segment_precomputed(aligned_image)

    # ----------------------------------------
    # build the multiscale segmentation mask
    #----------------------------------------

    logger.info("Creating segmentation mask...")
    try:

        # read image metadata for the given level
        image_metadata = utils.load_json(data_path=dataset_path, keyname=".zattrs")
        scale = str(args.level)
        image_metadata = utils.parse_zarr_metadata(metadata=image_metadata, multiscale=scale)

        # current resolution (zyx)
        current_res = (
            image_metadata["axes"]["z"]["scale"],
            image_metadata["axes"]["y"]["scale"],
            image_metadata["axes"]["x"]["scale"],
        )
        logger.info(f"Current resolution: {current_res}")
        
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
        
        # write level-0 mask and report pyramid depth
        voxel_size, n_lvls = upscale_mask.upscale_mask(
            dataset_path=dataset_path,
            mask_data=aligned_image,
            upscale_factors_zyx=upscale_factors_zyx,
            output_folder=args.seg_path,
            filename="ccf_anno_in_sample_space.zarr",
            dest_multiscale="0",
        )
        
        logger.info(f"Creating {n_lvls} levels in the pyramid.")
        
        # write the remaining pyramid levels
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

    except ImportError as e:
        logger.info(f"Warning: Could not import upscale_mask modules: {e}")
        logger.info("Skipping segmentation mask creation and S3 upload.")
    except Exception as e:
        logger.info(f"Error during segmentation mask creation: {e}")
        sys.exit(1)

    
if __name__ == "__main__":
    main() 