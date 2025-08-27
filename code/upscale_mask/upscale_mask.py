"""
Code to upsample a segmentatation mask
"""

import json
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import numcodecs
import numpy as np
import s3fs
import xarray_multiscale
import zarr
from . import utils
from dask.distributed import Client, LocalCluster
from numcodecs import Blosc
from zarr import Group, open_group

from .omezarr_metadata import _get_pyramid_metadata, write_ome_ngff_metadata
from .zarr_writer import BlockedArrayWriter
import gc


def compute_pyramid(
    data: dask.array.core.Array,
    n_lvls: int,
    scale_axis: Tuple[int],
    chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
) -> List[dask.array.core.Array]:
    """
    Computes the pyramid levels given an input full resolution image data

    Parameters
    ------------------------

    data: dask.array.core.Array
        Dask array of the image data

    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image

    scale_axis: Tuple[int]
        Scaling applied to each axis

    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"

    Returns
    ------------------------

    Tuple[List[dask.array.core.Array], Dict]:
        List with the downsampled image(s) and dictionary
        with image metadata
    """

    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=xarray_multiscale.reducers.windowed_mode_countless,  # func
        scale_factors=scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [pyramid_level.data for pyramid_level in pyramid]


def write_multiscales(
    path_to_data: Union[str, Path],
    voxel_size: List[float],
    chunk_size: List[int] = [128, 128, 128],
    scale_factor: List[int] = [2, 2, 2],
    target_size_mb: int = 2048,
    n_lvls: int = 5,
    root_group: Group = None,
):
    """
    Writes a multi-scale pyramid from an existing Zarr dataset.

    Parameters
    ----------
    path_to_data : Union[str, Path]
        Path to the base Zarr dataset (e.g., '0' level should be present).
    chunk_size : List[int], optional
        Chunk size to use for writing each pyramid level. Default is [128, 128, 128].
    scale_factor : List[int], optional
        Scaling factor per axis to downsample the data. Default is [2, 2, 2].
    target_size_mb : int, optional
        Target block size in MB for optimized writing. Default is 2048 MB.
    n_lvls : int, optional
        Number of pyramid levels to generate (excluding base). Default is 5.
    root_group : Group, optional
        Zarr group to write the pyramid to. If None, a new group will be created at `path_to_data`.
    """
    path_to_data = Path(path_to_data)
    if not path_to_data.exists():
        raise FileNotFoundError(f"Path {path_to_data} does not exist!")

    # Load the base scale (level 0)
    base_scale = da.from_zarr(path_to_data / "0")

    if root_group is None:
        # Assume top-level group creation if not provided
        root_group = open_group(path_to_data.parent, mode="a")

    if path_to_data.name in root_group:
        new_channel_group = root_group[path_to_data.name]
        print(f"Group '{path_to_data.name}' already exists. Reusing it.")
    else:
        raise ValueError("There must be a group created!")

    # Compute block shape used for optimized writing
    block_shape = list(
        BlockedArrayWriter.get_block_shape(
            arr=base_scale,
            target_size_mb=target_size_mb,
            chunks=chunk_size,
        )
    )

    # Pad block shape if fewer than 5D
    extra_axes = (1,) * (5 - len(block_shape))

    block_shape = extra_axes + tuple(block_shape)

    extra_axes_chunks = (1,) * (5 - len(chunk_size))
    chunk_size = extra_axes_chunks + tuple(chunk_size)

    multiscale_zarr_json = write_ome_ngff_metadata(
        arr_shape=base_scale.shape,
        chunk_size=chunk_size,
        image_name="cell_segmentation",
        n_lvls=n_lvls,
        scale_factors=scale_factor,
        voxel_size=voxel_size,
        origin=[0, 0, 0],
        metadata=_get_pyramid_metadata(),
    )

    with open(f"{path_to_data}/.zattrs", "w") as f:
        json.dump(multiscale_zarr_json, f)

    # Compression settings
    compressor = Blosc(cname="zstd", clevel=3, shuffle=1, blocksize=0)

    current_scale = base_scale

    for level in range(n_lvls):
        # Add missing dimensions if needed
        scale_factors_padded = ([1] * (len(current_scale.shape) - len(scale_factor))) + scale_factor

        # Compute one level of pyramid
        pyramid = compute_pyramid(
            data=current_scale,
            scale_axis=scale_factors_padded,
            chunks=chunk_size,
            n_lvls=2,  # Generate next level only
        )

        # Select the downsampled array (next level)
        current_scale = pyramid[-1]

        print(
            f"[level {level + 1}] Writing pyramid level with shape {current_scale.shape} - Block shape: {block_shape}"
        )

        # Create Zarr dataset for the level
        pyramid_group = new_channel_group.create_dataset(
            name=str(level + 1),
            shape=current_scale.shape,
            chunks=chunk_size,
            dtype=current_scale.dtype,
            compressor=compressor,
            dimension_separator="/",
            overwrite=True,
        )

        # Store data in blocks
        BlockedArrayWriter.store(current_scale, pyramid_group, block_shape)


def initialize_output_volume(
    output_params: Dict,
    output_volume_size: Tuple[int, int, int],
) -> zarr.core.Array:
    """
    Initializes the zarr directory where the
    volume will be upsampled.

    Inputs
    ------
    output_params: Dict
        Parameters to create the zarr storage.
    output_volume_size: Tuple[int]
        Output volume size for the zarr file.

    Returns
    -------
    Zarr thread-safe datastore initialized on OutputParameters.
    """

    # Local execution
    out_group = zarr.open_group(output_params["path"], mode="w")

    # Cloud execuion
    if output_params["path"].startswith("s3"):
        s3 = s3fs.S3FileSystem(
            config_kwargs={
                "max_pool_connections": 50,
                "s3": {
                    "multipart_threshold": 64
                    * 1024
                    * 1024,  # 64 MB, avoid multipart upload for small chunks
                    "max_concurrent_requests": 20,  # Increased from 10 -> 20.
                },
                "retries": {
                    "total_max_attempts": 100,
                    "mode": "adaptive",
                },
            }
        )
        store = s3fs.S3Map(root=output_params["path"], s3=s3)
        out_group = zarr.open(store=store, mode="a")

    path = "0"
    chunksize = output_params["chunksize"]
    datatype = output_params["dtype"]
    dimension_separator = output_params["dimension_separator"]
    compressor = output_params["compressor"]
    print("Using compressor: ", compressor)
    output_volume = out_group.create_dataset(
        path,
        shape=(
            1,
            1,
            output_volume_size[0],
            output_volume_size[1],
            output_volume_size[2],
        ),
        chunks=chunksize,
        dtype=datatype,
        compressor=compressor,
        dimension_separator=dimension_separator,
        overwrite=True,
        fill_value=0,
    )

    return output_volume


def upscale_zarr_with_padding(
    input_zarr,
    output_params: Dict,
    upscale_factors_zyx: Tuple[int] = (1, 4, 4),
    new_shape: Optional[Tuple] = None,
    n_workers: Optional[int] = 16,
):
    """
    Upscale a Zarr volume by specified factors in the spatial dimensions (z, y, x)
    and save to a new Zarr file. Assumes input is in tczyx format with t and c = 1.
    Adds zero padding if new_shape is provided and differs from calculated shape.

    Parameters:
    input_zarr: dask.array.Array
        Lazy mask
    output_params: Dict
        Dictionary with the parameters for the output Zarr file.
    upscale_factors_zyx: Tuple[int]
        Tuple of upscale factors for z, y, and x dimensions. Default: (1, 4, 4)
    new_shape: Optional[Tuple]
        If provided, the output will be padded to this shape. Default: None
    n_workers: Optional[int]
        Optional number of workers for the dask cluster

    """
    t, c = 1, 1
    if len(input_zarr.shape) == 5:
        _, _, z, y, x = input_zarr.shape
    else:
        z, y, x = input_zarr.shape

    # Calculate the shape of the upscaled volume
    calculated_new_shape = (
        t,
        c,
        z * upscale_factors_zyx[0],
        y * upscale_factors_zyx[1],
        x * upscale_factors_zyx[2],
    )

    if new_shape is not None:
        if len(new_shape) != 5:
            new_shape = (t, c, new_shape[0], new_shape[1], new_shape[2])
        padding = tuple(max(0, new - calc) for new, calc in zip(new_shape, calculated_new_shape))
    else:
        new_shape = calculated_new_shape
        padding = (0, 0, 0, 0, 0)

    chunk_size = output_params["chunksize"]  # (1, 1, 128, 128, 128)

    print("Getting max value")
    max_value = input_zarr.max()
    print("Maximum value:", max_value)

    print(
        f"Upscaling from size {input_zarr.shape} by {upscale_factors_zyx} to new shape {new_shape} with {output_params['chunksize']} chunk size and dtype: {output_params['dtype']} as determined by maximum value {max_value}"
    )
    print(f"Padding: {padding}")

    client = Client(LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True))

    # Initialize output volume with the new shape
    output_zarr = initialize_output_volume(output_params, new_shape[-3:])

    # Calculate the total number of chunks to process
    total_chunks = (np.ceil(z / 128) * np.ceil(y / 128) * np.ceil(x / 128)).astype(int)
    current_chunk = 1

    # Process and upscale each chunk
    for z_idx in range(0, z, 128):
        for y_idx in range(0, y, 128):
            for x_idx in range(0, x, 128):
                current_chunk += 1
                # Extract the current chunk
                if len(input_zarr.shape) == 5:  # tczyx
                    chunk = input_zarr[
                        0, 0, z_idx : z_idx + 128, y_idx : y_idx + 128, x_idx : x_idx + 128
                    ]
                elif len(input_zarr.shape) == 3:  # zyx
                    chunk = input_zarr[
                        z_idx : z_idx + 128, y_idx : y_idx + 128, x_idx : x_idx + 128
                    ]
                else:
                    print(
                        "len(input_zarr.shape) not compatible: ", len(input_zarr.shape), "exiting"
                    )
                    exit()

                # Upscale the chunk by duplicating each value to fill a block
                upscaled_chunk = np.repeat(
                    np.repeat(
                        np.repeat(chunk, upscale_factors_zyx[0], axis=0),
                        upscale_factors_zyx[1],
                        axis=1,
                    ),
                    upscale_factors_zyx[2],
                    axis=2,
                )

                # Calculate the indices for placing the upscaled chunk in the output
                z_new, y_new, x_new = (
                    z_idx * upscale_factors_zyx[0],
                    y_idx * upscale_factors_zyx[1],
                    x_idx * upscale_factors_zyx[2],
                )
                print(
                    f"Processing chunk {current_chunk}/{total_chunks} at z: {z_new}, y: {y_new}, x: {x_new}"
                    f" - Upscaled chunk shape: {upscaled_chunk.shape} - {new_shape} new shape"
                )

                # Add the upscaled chunk to the output, considering padding
                output_zarr[
                    0,
                    0,
                    z_new : min(z_new + upscaled_chunk.shape[0], new_shape[2] - padding[2]),
                    y_new : min(y_new + upscaled_chunk.shape[1], new_shape[3] - padding[3]),
                    x_new : min(x_new + upscaled_chunk.shape[2], new_shape[4] - padding[4]),
                ] = upscaled_chunk[
                    : min(upscaled_chunk.shape[0], new_shape[2] - padding[2] - z_new),
                    : min(upscaled_chunk.shape[1], new_shape[3] - padding[3] - y_new),
                    : min(upscaled_chunk.shape[2], new_shape[4] - padding[4] - x_new),
                ]

    print("Upscaling completed.")



def upscale_array_3d(data_3d: np.ndarray, upscale_factors_zyx: Tuple[int]) -> np.ndarray:
    """
    Upscale a 3D array using nearest neighbor interpolation (repeat).
    """
    upscaled = data_3d
    
    if upscale_factors_zyx[0] > 1:
        upscaled = np.repeat(upscaled, upscale_factors_zyx[0], axis=0)
    if upscale_factors_zyx[1] > 1:
        upscaled = np.repeat(upscaled, upscale_factors_zyx[1], axis=1)
    if upscale_factors_zyx[2] > 1:
        upscaled = np.repeat(upscaled, upscale_factors_zyx[2], axis=2)
    
    return upscaled

def upscale_zarr_with_padding_chunked(
    input_data: np.ndarray,
    output_params: Dict,
    upscale_factors_zyx: Tuple[int] = (1, 4, 4),
    new_shape: Optional[Tuple] = None,
    chunk_size_z: int = 32,
):
    """
    Memory-efficient upscaling using chunked processing.
    
    Parameters:
    -----------
    chunk_size_z: int
        Number of Z slices to process at once. Reduce if still running out of memory.
    """
    
    # Handle input dimensions
    if len(input_data.shape) == 5:
        t, c, z, y, x = input_data.shape
        if t != 1 or c != 1:
            raise ValueError(f"Expected t=1, c=1 for 5D input, got t={t}, c={c}")
        data_3d = input_data[0, 0]
    elif len(input_data.shape) == 3:
        z, y, x = input_data.shape
        data_3d = input_data
        t, c = 1, 1
    else:
        raise ValueError(f"Input must be 3D (z,y,x) or 5D (t,c,z,y,x), got shape {input_data.shape}")

    print(f"Input shape: {input_data.shape}")
    print(f"Processing 3D volume: {data_3d.shape}")

    # Calculate output dimensions
    upscaled_shape_3d = (
        z * upscale_factors_zyx[0],
        y * upscale_factors_zyx[1],
        x * upscale_factors_zyx[2],
    )

    # Handle padding
    if new_shape is not None:
        if len(new_shape) == 3:
            target_shape_3d = new_shape
        else:
            raise ValueError("new_shape must be 3D (z,y,x)")
        
        padding_3d = tuple(
            max(0, target - upscaled) 
            for target, upscaled in zip(target_shape_3d, upscaled_shape_3d)
        )
        
        if any(p > 0 for p in padding_3d):
            print(f"Padding will be added: {padding_3d}")
    else:
        target_shape_3d = upscaled_shape_3d
        padding_3d = (0, 0, 0)

    # Get output dtype
    max_value = data_3d.max()
    if max_value <= np.iinfo(np.uint8).max:
        output_dtype = np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        output_dtype = np.uint16
    else:
        output_dtype = np.uint32
        

    if 'dtype' in output_params:
        output_dtype = np.dtype(output_params['dtype'])

    output_dtype = np.uint32
        
    print(f"Target shape: {target_shape_3d}")
    print(f"Output dtype: {output_dtype}")
    print(f"Processing in chunks of {chunk_size_z} Z slices")

    # Create output zarr array first
    output_path = output_params.get('path', 'output.zarr')
    compression = output_params.get('compression', 'zstd')
    compression_opts = output_params.get('compression_opts', 3)
    default_chunks = (1, 1, min(128, target_shape_3d[0]), min(128, target_shape_3d[1]), min(128, target_shape_3d[2]))
    chunks = output_params.get('chunksize', default_chunks)
    
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Create zarr array with final shape
    final_shape = (1, 1) + target_shape_3d
    zarr_array = root.create_dataset(
        '0',
        shape=final_shape,
        dtype=output_dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        overwrite=True,
        fill_value=0,
        dimension_separator='/'
    )

    # Process in chunks
    z_original = data_3d.shape[0]
    num_chunks = (z_original + chunk_size_z - 1) // chunk_size_z
    
    for chunk_idx in range(num_chunks):
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
        
        # Calculate chunk boundaries
        z_start = chunk_idx * chunk_size_z
        z_end = min((chunk_idx + 1) * chunk_size_z, z_original)
        print(f"Processing chunk start: {z_start}, end: {z_end} (original Z: {z_original})")
        
        # Extract chunk
        chunk_data = data_3d[z_start:z_end, :, :]
        
        # Upscale chunk
        upscaled_chunk = upscale_array_3d(chunk_data, upscale_factors_zyx)
        
        # Calculate output Z boundaries
        out_z_start = z_start * upscale_factors_zyx[0]
        out_z_end = z_end * upscale_factors_zyx[0]
        
        # Convert dtype
        if upscaled_chunk.dtype != output_dtype:
            upscaled_chunk = upscaled_chunk.astype(output_dtype)

        coords = (
            slice(None),
            slice(None),
            slice(int(out_z_start), int(out_z_end)),
            slice(0, upscaled_chunk.shape[1]),
            slice(0, upscaled_chunk.shape[2]),
        )
        print(zarr_array.shape, upscaled_chunk.shape, coords)

        # Write to zarr (without padding first)
        zarr_array[
            coords
        ] = upscaled_chunk[np.newaxis, np.newaxis, :, :, :]
        
        # Clean up memory
        del upscaled_chunk, chunk_data
        gc.collect()
    
    print("Chunked upscaling completed successfully!")
    return target_shape_3d, output_dtype

def upscale_mask(
    dataset_path: str,
    mask_data: np.ndarray,
    output_folder: str,
    upscale_factors_zyx: tuple,
    filename: Optional[str] = "segmentation_mask.zarr",
    dest_multiscale: Optional[str] = "0",
):
    """
    Upscales a segmentation mask

    Parameters
    ----------
    dataset_path: str
        Path where the dataset that was used
        for segmentation is located.
    segmentation_mask_path: str
        Path where the segmentation mask is located
    output_folder: str
        Path where the upsampled segmentation mask will
        be stored.
    filename: str
        Filename for the upsampled segmentation mask
    dest_multiscale: Optional[str]
        Destination multiscale. This is useful to pull
        metadata. Default: "0"
    n_workers: Optional[int]
        Optional number of workers for the dask cluster. Default: 16
    """
    output_folder = Path(output_folder)

    if not output_folder.exists():
        raise FileNotFoundError(f"The output folder {output_folder} does not exist!")
    
    image_metadata = utils.load_json(data_path=dataset_path, keyname=".zattrs")
    image_lazy_data = da.from_zarr(f"{dataset_path}/0")

    image_compressor = image_metadata.get(".zarray", {}).get("compressor", None)
    if image_compressor is None:
        print(
            "Image metadata does not contain a valid compressor. Please check the dataset."
        )

    multiscales = image_metadata.get("multiscales", [])

    if not len(multiscales):
        raise ValueError(f"We need to have a multiscale pyramid dataset. Metadata: {image_metadata}")

    pyramid_scales = multiscales[0].get("datasets", [])

    if not len(pyramid_scales) > 1:
        raise ValueError(f"We need to have multiple scales. Metadata: {image_metadata}")

    image_metadata = utils.parse_zarr_metadata(metadata=image_metadata, multiscale=dest_multiscale)
    image_shape = image_lazy_data.shape

    # Getting list with Z Y X order of the resolution
    resolution_zyx = (
        image_metadata["axes"]["z"]["scale"],
        image_metadata["axes"]["y"]["scale"],
        image_metadata["axes"]["x"]["scale"],
    )

    print(
        "Image metadata: ",
        image_metadata,
        " - Resolution: ",
        resolution_zyx,
        " - Image compressor: ",
        image_compressor,
        " Image shape: ",
        image_shape,
    )
    # image_compressor = image_metadata[".zarray"]["compressor"]

    # Add this check and conversion:
    if isinstance(image_compressor, dict):
        if image_compressor["id"] == "blosc":
            image_compressor = numcodecs.Blosc(
                cname=image_compressor.get("cname", "zstd"),
                clevel=image_compressor.get("clevel", 1),
                shuffle=image_compressor.get("shuffle", 1),
                blocksize=image_compressor.get("blocksize", 0),
            )
        else:
            # Default fallback if compressor type is unknown
            image_compressor = numcodecs.Blosc(cname="zstd", clevel=3)
    elif image_compressor is None:
        image_compressor = numcodecs.Blosc(cname="zstd", clevel=3)

    # print("Image compressor: ", image_compressor)

    # image_compressor = (
    #    numcodecs.Blosc(cname="zstd", clevel=3) if image_compressor is None else image_compressor
    # )
    print("Image compressor: ", image_compressor)

    output_filepath = output_folder.joinpath(filename).as_posix()
    output_params = {
        "chunksize": [1, 1, 128, 128, 128],
        "resolution_zyx": resolution_zyx,
        # "dtype": np.uint8,
        "dtype": np.uint32,
        "path": output_filepath,
        "compressor": image_compressor,
        "dimension_separator": "/",
    }

    # Upsampling the segmentation mask
    # upscale_zarr_with_padding(
    #     input_zarr=mask_data,
    #     output_params=output_params,
    #     upscale_factors_zyx=upscale_factors_zyx,
    #     new_shape=image_lazy_data.shape,
    #     n_workers=n_workers,
    # )
    upscale_zarr_with_padding_chunked(
        input_data=mask_data,
        output_params=output_params,
        upscale_factors_zyx=upscale_factors_zyx,
        new_shape=image_lazy_data.shape[-3:],
        chunk_size_z=32,
    )

    return resolution_zyx, len(pyramid_scales)
