"""
Register an exaspim data to the Allen Institute's CCF atlas via the exaspim template

Pipeline:
(1) check orientation and run preprocessing on the given image.
(2) register the preprocessed brain image to the exaspim template using ANTs rigid and SyN registration.
(3) register the deformed image from (2) to the CCF Allen Atlas by applying template-to-CCF transforms
(4) register CCF annotation to brain space
"""
import gc
import logging
import multiprocessing
import os
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import ants
import dask
import dask.array as da
import numpy as np
import tifffile
import xarray_multiscale
import zarr
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeZarrWriter
from aind_data_schema.core.processing import DataProcess, ProcessName
from argschema import ArgSchemaParser
from dask.distributed import Client, LocalCluster, performance_report
from distributed import wait
from numcodecs import blosc
from skimage import io

from .__init__ import __version__

blosc.use_threads = False

from aind_exaspim_ccf_reg.configs import VMAX, VMIN, ArrayLike, PathLike, RegSchema
from aind_exaspim_ccf_reg.plots import visualize_reg, plot_antsimgs, plot_reg
from aind_exaspim_ccf_reg.preprocess import perc_normalization, check_orientation
from aind_exaspim_ccf_reg.utils import create_folder

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TemplateLoader:
    """Loading CCF and exaSPIM templates."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize TemplateLoader.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance for output messages
        """
        self.logger = logger
    
    def load_templates(self, reg_params: dict, outprefix: str) -> Tuple[ants.ANTsImage, ants.ANTsImage]:
        """
        Load and prepare CCF and exaSPIM templates.
        
        Parameters
        ----------
        reg_params : dict
            Registration parameters containing template paths and scaling
        outprefix : str
            Output prefix for saving template images
            
        Returns
        -------
        Tuple[ants.ANTsImage, ants.ANTsImage]
            Tuple of (ccf_template, exaspim_template)
        """
        scale = reg_params['sample_scale']
        ants_exaspim = ants.image_read(reg_params['exaspim_template_path'])
        ccf = ants.image_read(reg_params['ccf_path'])
        
        # Normalize CCF
        ccf = perc_normalization(ccf)
        self.logger.info(f"Loaded ccf: {ccf}")
        plot_antsimgs(ccf, 
                      f"{outprefix}/ccf_template",
                      title="ccf_template", 
                      vmin=0, vmax=1.5)

        # ants.image_write(ccf, f"{outprefix}load_ccf.nii.gz")
        
        # set the physical information of exaSPIM template to CCF
        ants_exaspim.set_spacing(ccf.spacing)
        ants_exaspim.set_origin(ccf.origin)
        ants_exaspim.set_direction(ccf.direction)
        self.logger.info(f"Loaded exaspim template: {ants_exaspim}")
        plot_antsimgs(ants_exaspim, 
                      f"{outprefix}/exaspim_template",
                      title="exaspim_template", 
                      vmin=0, vmax=1.5)

        # #------------------------
        # if round(ccf.spacing[0], 3) == 0.025:
        #     # TODO, create a mask for exaspim_template
        #     mask_path = "/data/exaSPIM_template_mask_25um_otsu.nii.gz"
        #     self.logger.info(f"Load exaSPIM template mask from {mask_path}")
        #     mask_25um = ants.image_read(mask_path)
        #     ants_exaspim = mask_25um * ants_exaspim
            
        #     plot_antsimgs(ants_exaspim, 
        #               f"{outprefix}/exaspim_template_masked",
        #               title="exaspim_template_masked", 
        #               vmin=0, vmax=1.5)

        # #------------------------
 
        return ccf, ants_exaspim


class ImagePreprocessor:
    """Handles image preprocessing including orientation checking and normalization."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize ImagePreprocessor.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance for output messages
        """
        self.logger = logger
    
    def preprocess_image(
        self, 
        acquisition_path: str, 
        zarr_image: np.ndarray, 
        scale: List[float],
        ants_exaspim: ants.ANTsImage,
        dataset_id: str,
        outprefix: str
    ) -> ants.ANTsImage:
        """
        Preprocess the input image including orientation checking and normalization.
        
        Parameters
        ----------
        acquisition_path : str
            Path to acquisition data
        zarr_image : np.ndarray
            Input image array
        scale : List[float]
            Scale factors for the image
        ants_exaspim : ants.ANTsImage
            exaSPIM template for alignment
        dataset_id : str
            Dataset identifier
        outprefix : str
            Output prefix for saving intermediate images
            
        Returns
        -------
        ants.ANTsImage
            Preprocessed ANTs image
        """
        # Check orientation
        ants_img = check_orientation(acquisition_path, zarr_image, self.logger)
        
        ants_img.set_spacing(scale)
        ants_img.set_direction(ants_exaspim.direction)
        ants_img.set_origin(ants_exaspim.origin)
        
        self.logger.info(f"Loaded OMEZarr dataset as antsimg: {ants_img}")

        # Intensity normalization
        self.logger.info("Start intensity normalization")
        start_time = datetime.now()
        ants_img = perc_normalization(ants_img)
        end_time = datetime.now()
        self.logger.info(
            f"Intensity normalization completed, execution time: {end_time - start_time} s -- image {ants_img}"
        )
        # #------------------------
        # # TODO, hardcode for 721332
        # mask_path = "/data/721332_whole_brain_mask_25um.nii.gz"
        # self.logger.info(f"Load brain mask from {mask_path}")
        # mask_25um = ants.image_read(mask_path)
        # ants_img = mask_25um * ants_img
        # #------------------------
        
        figpath = f"{outprefix}{dataset_id}_loaded_zarr_img"
        plot_antsimgs(ants_img, figpath, title=f"{dataset_id}_loaded_zarr_img", vmin=0, vmax=1.5)
        ants.image_write(ants_img, f"{outprefix}{dataset_id}_loaded_zarr_img.nii.gz")
        
        # Resample to isotropic resolution
        self.logger.info(f"Resample OMEZarr image to the resolution of ants_exaspim")
        self.logger.info(f"ants_exaspim: {ants_exaspim}")
        self.logger.info(f"ants_img: {ants_img}")
        
        ants_img = ants.resample_image(ants_img, ants_exaspim.spacing)
        self.logger.info(f"Resampled OMEZarr dataset: {ants_img}")

        figpath = f"{outprefix}{dataset_id}_resampled_zarr_img"
        plot_antsimgs(ants_img, figpath, title=f"{dataset_id}_resampled_zarr_img")        
        ants.image_write(ants_img, f"{outprefix}{dataset_id}_resampled_zarr_img.nii.gz")

        return ants_img


class RegistrationProcessor:
    """Handles ANTs registration operations."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize RegistrationProcessor.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance for output messages
        """
        self.logger = logger
    
    def perform_affine_registration(
        self,
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        outprefix: str,
        dataset_id: str,
        affine_reg_iterations: List[int]
    ) -> Tuple[ants.ANTsImage, List[str], List[str]]:
        """
        Perform affine registration between moving and fixed images.
        
        Parameters
        ----------
        fixed : ants.ANTsImage
            Fixed (target) image
        moving : ants.ANTsImage
            Moving (source) image
        outprefix : str
            Output prefix for transform files
        dataset_id : str
            Dataset identifier
        affine_reg_iterations : List[int]
            Number of iterations for affine registration
            
        Returns
        -------
        Tuple[ants.ANTsImage, List[str], List[str]]
            Tuple of (registered_image, forward_transforms, inverse_transforms)
        """
        start_time = datetime.now()
        registration_params = {
            "fixed": fixed,
            "moving": moving, 
            "type_of_transform": "TRSAA", 
            "outprefix": f"{outprefix}/{dataset_id}_to_exaSPIM_affine_", 
            "mask_all_stages": True,
            "grad_step": 0.25,
            "reg_iterations": affine_reg_iterations,
            "aff_metric": "mattes"
        }
        
        self.logger.info(f"Computing Alignment with parameters: {registration_params}")
        reg = ants.registration(**registration_params)
        end_time = datetime.now()
        self.logger.info(f"Affine alignment completed, execution time: {end_time - start_time} s -- image {reg}")
        
        ants_reg = reg["warpedmovout"]
        transform_dataset_to_atlas = reg["fwdtransforms"]
        transform_atlas_to_dataset = reg["invtransforms"]
        
        self.logger.info("Alignment Complete")
        self.logger.info(f"Transform to go from dataset to atlas: {transform_dataset_to_atlas}")
        self.logger.info(f"Transform to go from atlas to dataset: {transform_atlas_to_dataset}")

        task_name = f"{dataset_id}_to_exaspim_affine"
        visualize_reg(moving, fixed, ants_reg, task_name, outprefix)
        ants.image_write(ants_reg, f"{outprefix}{dataset_id}_to_exaspim_moved_affine.nii.gz") 
        
        return ants_reg, transform_dataset_to_atlas, transform_atlas_to_dataset
    
    def perform_syn_registration(
        self,
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        outprefix: str,
        outprefix_reg: str,
        dataset_id: str,
        syn_reg_iterations: List[int]
    ) -> Tuple[ants.ANTsImage, List[str], List[str]]:
        """
        Perform SyN registration between moving and fixed images.
        
        Parameters
        ----------
        fixed : ants.ANTsImage
            Fixed (target) image
        moving : ants.ANTsImage
            Moving (source) image
        outprefix : str
            Output prefix for visualization
        outprefix_reg : str
            Output prefix for transform files
        dataset_id : str
            Dataset identifier
        syn_reg_iterations : List[int]
            Number of iterations for SyN registration
            
        Returns
        -------
        Tuple[ants.ANTsImage, List[str], List[str]]
            Tuple of (registered_image, forward_transforms, inverse_transforms)
        """
        self.logger.info('Starting Deformable Registration')
        start_time = datetime.now()
        
        self.logger.info(f"ants_exaspim: {fixed}")
        self.logger.info(f"ants_img: {moving}")

        registration_params = {
            "fixed": fixed,
            "moving": moving, 
            "type_of_transform": "SyNOnly",
            "syn_metric": "CC", 
            "syn_sampling": 2,    
            "reg_iterations": syn_reg_iterations,
            "initial_transform": [f'{outprefix}/{dataset_id}_to_exaSPIM_affine_0GenericAffine.mat'],
            "outprefix": f"{outprefix_reg}/{dataset_id}_to_exaSPIM_SyN_",
        }

        reg = ants.registration(**registration_params)
        end_time = datetime.now()
        self.logger.info(f"Alignment Completed, execution time: {end_time - start_time} s -- image {reg}")
        
        ants_reg = reg["warpedmovout"]
        transform_dataset_to_atlas = reg["fwdtransforms"]
        transform_atlas_to_dataset = reg["invtransforms"]
        
        task_name = f"{dataset_id}_to_exaspim_syn"
        visualize_reg(moving, fixed, ants_reg, task_name, outprefix)
        
        self.logger.info(f"Saving aligned image: {outprefix}{dataset_id}_to_exaspim_moved_syn.nii.gz")
        ants.image_write(ants_reg, f"{outprefix}{dataset_id}_to_exaspim_moved_syn.nii.gz") 
        self.logger.info("Done saving")
        
        return ants_reg, transform_dataset_to_atlas, transform_atlas_to_dataset
    
    def apply_transforms(
        self,
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        transformlist: List[str],
        task_name: str,
        outprefix: str
    ) -> ants.ANTsImage:
        """
        Apply transforms to moving image.
        
        Parameters
        ----------
        fixed : ants.ANTsImage
            Fixed (target) image
        moving : ants.ANTsImage
            Moving (source) image
        transformlist : List[str]
            List of transform files to apply
        task_name : str
            Name for the task (used in visualization and output)
        outprefix : str
            Output prefix for saving results
            
        Returns
        -------
        ants.ANTsImage
            Transformed image
        """
        start_time = datetime.now()
        ants_reg = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=transformlist,
        )
        end_time = datetime.now()

        self.logger.info(f"Register to {task_name}, execution time: {end_time - start_time} s -- image {ants_reg}")
        
        visualize_reg(moving, fixed, ants_reg, task_name, outprefix)
        ants.image_write(ants_reg, f"{outprefix}{task_name}_moved.nii.gz") 
        
        return ants_reg


class RegistrationPipeline:
    """Main registration pipeline that orchestrates the entire registration process."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize RegistrationPipeline.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance for output messages
        """
        self.logger = logger
        self.template_loader = TemplateLoader(logger)
        self.image_preprocessor = ImagePreprocessor(logger)
        self.registration_processor = RegistrationProcessor(logger)
        self.zarr_writer = ZarrWriter(logger)
    
    def _get_registration_parameters(self, inputs: dict, level: int) -> dict:
        """
        Get registration parameters based on the processing level.
        
        Parameters
        ----------
        inputs : dict
            Input configuration dictionary
        level : int
            Processing level (2, 3, 5, or 6)
            
        Returns
        -------
        dict
            Registration parameters for the specified level
            
        Raises
        ------
        ValueError
            If level is not supported
        """
        if level in [6, 3]:
            return inputs['reg_param_25um']
        elif level in [5, 2]:
            return inputs['reg_param_10um']
        else:
            raise ValueError(f"Unsupported level: {level}")
    
    def register(
        self,
        parser: ArgSchemaParser,
        zarr_image: np.ndarray,
    ) -> List[str]:
        """
        Perform registration of zarr image to CCF template using ArgSchemaParser for config.
        
        Parameters
        ----------
        parser : ArgSchemaParser
            Parser containing RegSchema configuration
        zarr_image : np.ndarray
            Input image array
            
        Returns
        -------
        List[str]
            List of transform paths for brain to exaSPIM template registration
        """
        self.logger.info(f"Running preprocessing and initial alignment with ants version: {ants.__version__}")
        inputs = parser.args

        level = inputs['level']
        self.logger.info(f"level: {level}")
        acquisition_path = inputs['acquisition_output']
        outprefix = inputs['outprefix']
        outprefix_reg = inputs['outprefix_reg']
        dataset_id = inputs['dataset_id']
        
        # Get registration parameters based on level
        reg_params = self._get_registration_parameters(inputs, level)
        scale = reg_params['sample_scale']
        affine_reg_iterations = reg_params['affine_reg_iterations']
        syn_reg_iterations = reg_params['syn_reg_iterations']
        
        # Load templates
        ccf, ants_exaspim = self.template_loader.load_templates(reg_params, outprefix)
        
        # Preprocess image
        ants_img = self.image_preprocessor.preprocess_image(
            acquisition_path, zarr_image, scale, ants_exaspim, dataset_id, outprefix
        )
        ants_img_original = ants_img

        # Perform affine registration
        ants_reg, _, _ = self.registration_processor.perform_affine_registration(
            ants_exaspim, ants_img, outprefix, dataset_id, affine_reg_iterations
        )

        # Perform SyN registration
        ants_reg, _, _ = self.registration_processor.perform_syn_registration(
            ants_exaspim, ants_img, outprefix, outprefix_reg, dataset_id, syn_reg_iterations
        )
        
        ants_brain_exaspim = ants_reg
        
        # Apply registration: ants_img --> exaSPIM
        brain_to_exaspim_transform_warp_path = os.path.abspath(
            f"{outprefix_reg}/{dataset_id}_to_exaSPIM_SyN_1Warp.nii.gz"
        )
        brain_to_exaspim_transform_affine_path = os.path.abspath(
            f"{outprefix_reg}/{dataset_id}_to_exaSPIM_SyN_0GenericAffine.mat"
        )

        brain_to_exaspim_transform_path = [
            brain_to_exaspim_transform_warp_path,
            brain_to_exaspim_transform_affine_path,
        ]

        self.logger.info(f"brain_to_exaspim_transform_path: {brain_to_exaspim_transform_path}")

        ants_brain_exaspim = self.registration_processor.apply_transforms(
            ants_exaspim, ants_img_original, brain_to_exaspim_transform_path,
            f"{dataset_id}_to_exaspim", outprefix
        )
        
        task_name = f"{dataset_id}_to_exaspim"
        visualize_reg(ants_img_original, ants_exaspim, ants_brain_exaspim, task_name, outprefix)
        ants.image_write(ants_brain_exaspim, f"{outprefix}{task_name}_moved.nii.gz") 
    
        # Apply registration: ants_brain_exaspim --> ccf
        ants_brain_ccf = self.registration_processor.apply_transforms(
            ccf, ants_brain_exaspim, inputs["exaspim_to_ccf_transform_path"],
            f"{dataset_id}_to_ccf", outprefix
        )

        task_name = f"{dataset_id}_to_ccf"
        visualize_reg(ants_brain_exaspim, ccf, ants_brain_ccf, task_name, outprefix)
        ants.image_write(ants_brain_ccf, f"{outprefix}{task_name}_moved.nii.gz") 

        return brain_to_exaspim_transform_path

    def apply_transforms_to_10um(
        self,
        parser: ArgSchemaParser,
        zarr_image: np.ndarray,
        brain_to_exaspim_transform_path: List[str],
        dataset_id: Optional[str] = None
    ) -> str:
        """
        Apply transforms to 10um resolution data using ArgSchemaParser for config.
        
        Parameters
        ----------
        parser : ArgSchemaParser
            Parser containing RegSchema configuration
        zarr_image : np.ndarray
            Input image array
        brain_to_exaspim_transform_path : List[str]
            Transform paths for brain to exaSPIM
        dataset_id : str, optional
            Dataset identifier, defaults to parser.inputs['dataset_id']
            
        Returns
        -------
        str
            Output path of the registered image
        """
        inputs = parser.args

        acquisition_path = inputs['acquisition_output']
        exaspim_to_ccf_transform_path = inputs['exaspim_to_ccf_transform_path']
        outprefix = inputs['outprefix']
        outprefix_reg = inputs['outprefix_reg']

        if dataset_id is None:
            dataset_id = inputs['dataset_id']
        
        self.logger.info(f"Running preprocessing and initial alignment with ants version: {ants.__version__}")

        # Load ccf and exaspim templates using 10um parameters
        reg_params = inputs['reg_param_10um']
        scale = reg_params['sample_scale']
        
        # Load templates
        ccf, ants_exaspim = self.template_loader.load_templates(reg_params, outprefix)

        # Preprocess image (without intensity normalization for 10um)
        ants_img = check_orientation(acquisition_path, zarr_image, self.logger)
        
        ants_img.set_spacing(scale)
        ants_img.set_direction(ants_exaspim.direction)
        ants_img.set_origin(ants_exaspim.origin)
        
        figpath = f"{outprefix}{dataset_id}_loaded_zarr_img"
        plot_antsimgs(perc_normalization(ants_img), 
                      figpath, 
                      title=f"{dataset_id}_loaded_zarr_img")
        ants.image_write(ants_img, f"{outprefix}{dataset_id}_loaded_zarr_img.nii.gz")
        self.logger.info(f"Loaded OMEZarr dataset as antsimg: {ants_img}")

        # Resample to 10um isotropic resolution
        self.logger.info(f"Resample OMEZarr image to the resolution of ants_exaspim")
        self.logger.info(f"ants_exaspim: {ants_exaspim}")
        self.logger.info(f"ants_img: {ants_img}")
        
        ants_img = ants.resample_image(ants_img, ants_exaspim.spacing)

        ants.image_write(ants_img, f"{outprefix}{dataset_id}_resampled_zarr_img.nii.gz")
        self.logger.info(f"Resampled OMEZarr dataset: {ants_img}")

        ants_img_perc_norm = perc_normalization(ants_img)
        
        figpath = f"{outprefix}{dataset_id}_resampled_zarr_img"
        plot_antsimgs(ants_img_perc_norm, 
                      figpath, 
                      title=f"{dataset_id}_resampled_zarr_img")

        # Apply registration: ants_img --> exaSPIM
        tp = dataset_id.replace("_10um", "")
        self.logger.info(f"brain_to_exaspim_transform_path: {brain_to_exaspim_transform_path}")

        ants_brain_exaspim = self.registration_processor.apply_transforms(
            ants_exaspim, ants_img, brain_to_exaspim_transform_path,
            f"{dataset_id}_to_exaSPIM", outprefix
        )
        ants_brain_exaspim_perc_norm = perc_normalization(ants_brain_exaspim)                
        task_name = f"{dataset_id}_to_exaSPIM"
        visualize_reg(ants_img_perc_norm, 
                     ants_exaspim,
                     ants_brain_exaspim_perc_norm, task_name, outprefix)
        ants.image_write(ants_brain_exaspim, f"{outprefix}{task_name}_moved.nii.gz") 

        # Apply registration: ants_img --> ccf
        self.logger.info(f"exaspim_to_ccf_transform_path: {exaspim_to_ccf_transform_path}")

        ants_brain_ccf = self.registration_processor.apply_transforms(
            ccf, ants_brain_exaspim, inputs["exaspim_to_ccf_transform_path"],
            f"{dataset_id}_to_ccf", outprefix
        )
        ants_brain_ccf_perc_norm = perc_normalization(ants_brain_ccf)
        task_name = f"{dataset_id}_to_ccf"
        visualize_reg(ants_brain_exaspim_perc_norm, 
                     ccf,
                     ants_brain_ccf_perc_norm, task_name, outprefix)

        output_path = f"{outprefix}/{task_name}_moved.nii.gz"
        ants.image_write(ants_brain_ccf, output_path) 

        # Write to zarr
        image_name = "ccf_aligned.zarr"
        
        aligned_image = ants_brain_ccf.numpy()
        aligned_image_dask = da.from_array(aligned_image)
        
        self.logger.info(f"Before changing orientation: {aligned_image_dask.shape}, DR: {aligned_image.min()}, {aligned_image.max()}")

        aligned_image_dask = da.moveaxis(aligned_image_dask, [0, 1, 2], [2, 1, 0])
        self.logger.info(f"After changing orientation: {aligned_image_dask.shape}, DR: {aligned_image.min()}, {aligned_image.max()}, {aligned_image_dask.dtype}, {aligned_image.dtype}")

        params = {
            "OMEZarr_params": {
                "clevel": 1,
                "compressor": "zstd",
                "chunks": (64, 64, 64),
            },
            "metadata_folder": outprefix
        }

        opts = {
            "compressor": blosc.Blosc(
                cname=params["OMEZarr_params"]["compressor"],
                clevel=params["OMEZarr_params"]["clevel"],
                shuffle=blosc.SHUFFLE,
            )
        }

        self.zarr_writer.write_zarr(
            img_array=aligned_image_dask,
            physical_pixel_sizes=(10, 10, 10),
            output_path=outprefix_reg,
            image_name=image_name,
            opts=opts,
            params=params
        )

        return output_path

class ImageProcessor:
    """Handles image processing operations including padding, pyramid computation, and normalization."""
    
    @staticmethod
    def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
        """
        Pads a dask array to be in a 5D shape.
        
        Parameters
        ----------
        arr : ArrayLike
            Dask/numpy array that contains image data.
        dim : int
            Number of dimensions that the array will be padded
            
        Returns
        -------
        ArrayLike
            Padded dask/numpy array.
            
        Raises
        ------
        ValueError
            If padding more than 5 dimensions is requested.
        """
        if dim > 5:
            raise ValueError("Padding more than 5 dimensions is not supported.")

        while arr.ndim < dim:
            arr = arr[np.newaxis, ...]
        return arr

    @staticmethod
    def compute_pyramid(
        data: dask.array.core.Array,
        n_lvls: int,
        scale_axis: Tuple[int],
        chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
    ) -> List[dask.array.core.Array]:
        """
        Computes the pyramid levels given an input full resolution image data.
        
        Parameters
        ----------
        data : dask.array.core.Array
            Dask array of the image data
        n_lvls : int
            Number of downsampling levels that will be applied to the original image
        scale_axis : Tuple[int]
            Scaling applied to each axis
        chunks : Union[str, Sequence[int], Dict[Hashable, int]]
            chunksize that will be applied to the multiscales. Default: "auto"
            
        Returns
        -------
        List[dask.array.core.Array]
            List with the downsampled image(s)
        """
        pyramid = xarray_multiscale.multiscale(
            data,
            xarray_multiscale.reducers.windowed_mean,  # func
            scale_axis,  # scale factors
            preserve_dtype=True,
            chunks=chunks,
        )[:n_lvls]

        return [arr.data for arr in pyramid]

    @staticmethod
    def get_pyramid_metadata() -> dict:
        """
        Gets pyramid metadata in OMEZarr format.
        
        Returns
        -------
        dict
            Dictionary with the downscaling OMEZarr metadata
        """
        return {
            "metadata": {
                "description": """Downscaling implementation based on the
                    windowed mean of the original array""",
                "method": "xarray_multiscale.reducers.windowed_mean",
                "version": str(xarray_multiscale.__version__),
                "args": "[false]",
                # No extra parameters were used different
                # from the orig. array and scales
                "kwargs": {},
            }
        }


class ZarrWriter:
    """Handles writing arrays to OMEZarr format."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize ZarrWriter.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance for output messages
        """
        self.logger = logger
        self._setup_dask_config()
    
    def _setup_dask_config(self) -> None:
        """Configure dask settings for optimal performance."""
        dask_folder = Path("../scratch")
        dask.config.set({
            "temporary-directory": dask_folder,
            "local_directory": dask_folder,
            "tcp-timeout": "300s",
            "array.chunk-size": "384MiB",
            "distributed.comm.timeouts": {
                "connect": "300s",
                "tcp": "300s",
            },
            "distributed.scheduler.bandwidth": 100000000,
            "distributed.worker.memory.rebalance.measure": "optimistic",
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": 0.92,
            "distributed.worker.memory.pause": 0.95,
            "distributed.worker.memory.terminate": 0.98,
        })

    def write_zarr(
        self,
        img_array: np.array,
        physical_pixel_sizes: List[int],
        output_path: PathLike,
        image_name: PathLike,
        opts: dict,
        params: dict
    ) -> None:
        """
        Writes array to the OMEZarr format.

        Parameters
        ----------
        img_array : np.array
            Array with the registered image
        physical_pixel_sizes : List[int]
            List with the physical pixel sizes. The order must be [Z, Y, X]
        output_path : PathLike
            Path where the .zarr image will be written
        image_name : PathLike
            Image name for the .zarr image
        opts : dict
            Dictionary with the storage options for the zarr image
        params : dict
            Dictionary with additional parameters including metadata_folder
        """
        physical_pixels = PhysicalPixelSizes(
            physical_pixel_sizes[0],
            physical_pixel_sizes[1],
            physical_pixel_sizes[2],
        )

        scale_axis = [2, 2, 2]
        pyramid_data = ImageProcessor.compute_pyramid(
            img_array,
            -1,
            scale_axis,
            params["OMEZarr_params"]["chunks"],
        )

        pyramid_data = [ImageProcessor.pad_array_n_d(pyramid) for pyramid in pyramid_data]
        self.logger.info(f"Pyramid {pyramid_data}")

        # Writing OMEZarr image
        n_workers = multiprocessing.cpu_count()
        threads_per_worker = 1
        # Using 1 thread since is in single machine.
        # Avoiding the use of multithreaded due to GIL
        
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit="auto",
        )
        client = Client(cluster)

        writer = OmeZarrWriter(output_path)

        dask_report_file = Path(params["metadata_folder"]).joinpath("dask_report.html")

        with performance_report(filename=dask_report_file):
            dask_jobs = writer.write_multiscale(
                pyramid=pyramid_data,
                image_name=image_name,
                chunks=pyramid_data[0].chunksize,
                physical_pixel_sizes=physical_pixels,
                channel_names=None,
                channel_colors=None,
                scale_factor=scale_axis,
                storage_options=opts,
                compute_dask=False,
                **ImageProcessor.get_pyramid_metadata(),
            )

            if len(dask_jobs):
                dask_jobs = dask.persist(*dask_jobs)
                wait(dask_jobs)

        client.close()