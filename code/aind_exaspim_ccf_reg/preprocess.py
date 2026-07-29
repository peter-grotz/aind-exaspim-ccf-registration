"""
Preprocessing functions for exaspim data.
"""

import logging
from datetime import datetime
import json
import ants
import numpy as np
import scipy.ndimage as ni
from aind_exaspim_ccf_reg.configs import VMAX, VMIN
from aind_exaspim_ccf_reg.plots import plot_antsimgs
from skimage.filters import threshold_li
from skimage.measure import label
from typing import List, Optional, Union, Tuple, Dict, Any

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def acquisition_images(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-tile image entries, from either acquisition schema.

    aind-data-schema v1 lists them at acquisition["tiles"]; v2 moved them to
    data_streams[].configurations[<Imaging config>].images ("Image spim" entries).
    Both carry "file_name", which is all the scope check below needs.
    """
    if metadata.get("tiles"):
        return metadata["tiles"]
    for stream in metadata.get("data_streams") or []:
        for config in stream.get("configurations") or []:
            if config.get("images"):
                return config["images"]
    return []


def normalize_acquisition_axes(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Axis entries as [{"dimension": int, "direction": str}], from either schema.

    v1 lists them at acquisition["axes"] with an explicit "dimension" per axis.
    v2 moved them to coordinate_system.axes and dropped "dimension" -- there the
    list position carries it, and v2 orders the axes Z, Y, X, which matches the v1
    dimension numbering (Z=0, Y=1, X=2) and the (z, y, x) array order.

    Returned in descending-dimension order (X, Y, Z). get_adjustments appends
    swaps in iteration order and adjust_array applies them sequentially, so the
    order is part of the result, not just presentation -- v2 lists the axes
    reversed relative to v1, and emitting that order directly would feed
    get_adjustments a different swap sequence for the same orientation. Every v1
    acquisition observed already lists them descending, so this is a no-op there.
    """
    axes = metadata.get("axes")
    if axes:
        normalized = [{"dimension": a["dimension"], "direction": a["direction"]} for a in axes]
    else:
        cs_axes = (metadata.get("coordinate_system") or {}).get("axes") or []
        normalized = [{"dimension": i, "direction": a["direction"]} for i, a in enumerate(cs_axes)]
    return sorted(normalized, key=lambda a: a["dimension"], reverse=True)


def check_orientation(
    acquisition_path: str,
    zarr_image: np.ndarray,
    logger: logging.Logger,
    scale: Optional[List[float]] = None,
) -> ants.ANTsImage:
    """
    Check and adjust image orientation based on acquisition metadata.
    
    Parameters
    ----------
    acquisition_path : str
        Path to the acquisition metadata JSON file
    zarr_image : np.ndarray
        Input image array
    logger : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    ants.ANTsImage
        Oriented image as ANTs image object
    """
    logger.info("Starting check orientation ......")
    
    with open(acquisition_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"acquisition schema_version: {metadata.get('schema_version')}")
    images = acquisition_images(metadata)
    if not images:
        raise ValueError(
            f"no tile/image entries in {acquisition_path} "
            f"(schema_version={metadata.get('schema_version')})")
    file_name_1st = images[0]["file_name"]
    logger.info(f"The first tile file name: {file_name_1st}")

    if "tile_000000_ch_" in file_name_1st:
        logger.info("The input is a Beta scope sample!!")
        ccf_directions = {
            0: "Anterior_to_posterior",
            1: "Superior_to_inferior",
            2: "Left_to_right",
        }
    else:
        logger.info("The input is a Alpha scope sample!!")
        ccf_directions = {
            0: "Posterior_to_anterior",
            1: "Inferior_to_superior",
            2: "Left_to_right",
        }

    logger.info(f"CCF_DIRECTIONS {ccf_directions}")
    axes = normalize_acquisition_axes(metadata)
    if not axes:
        raise ValueError(
            f"no axis entries in {acquisition_path} "
            f"(schema_version={metadata.get('schema_version')})")
    logger.info(f"normalized axes: {axes}")
    swaps, flips = get_adjustments(axes, ccf_directions)
    swaps = [s for s in swaps if s[0] != s[1]]
    logger.info(f"**swaps {swaps}, flips {flips}**")
    zarr_image = adjust_array(zarr_image, swaps, flips)
    ants_img = ants.from_numpy(zarr_image.astype(np.float32))

    if scale is not None:
        # Spacing must follow the SAME axis swaps as the image (flips do not
        # change voxel size), or the anisotropic axis pairs with the wrong
        # spacing and the moving image is squished.
        new_scale = reorder_scale(list(scale), swaps)
        logger.info(f"**scale {list(scale)} -> reordered {new_scale}**")
        ants_img.set_spacing(new_scale)

    return ants_img


def get_adjustments(
    axes: List[Dict[str, Any]], 
    orientation: Dict[int, str]
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Compute the swaps and flips needed to match axes to a reference orientation.

    Parameters
    ----------
    axes : List[Dict[str, Any]]
        Axis entries from acquisition.json, each with 'dimension' and 'direction'.
    orientation : Dict[int, str]
        Reference orientation for each axis dimension.

    Returns
    -------
    Tuple[List[Tuple[int, int]], List[int]]
        (swaps, flips): axis pairs to swap and axes to flip.
    """
    flips = []
    swaps = []
    
    for i in range(len(axes)):
        ax = axes[i]
        dim = ax["dimension"]
        direction = ax["direction"].lower()

        if orientation[dim].lower() == direction:
            continue

        for idx, d in orientation.items():
            # Same direction: swap only
            if d.lower() == direction:
                swaps.append((dim, idx))
            # Reversed direction: swap and flip
            elif d.lower() == "_".join(direction.split("_")[::-1]):
                swaps.append((dim, idx))
                flips.append(idx)

    return swaps, flips


def adjust_array(arr: np.ndarray, swaps: List[Tuple[int, int]], flips: List[int]) -> np.ndarray:
    """
    Reorder and flip array axes.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    swaps : List[Tuple[int, int]]
        Axis pairs to swap.
    flips : List[int]
        Axes to flip.

    Returns
    -------
    np.ndarray
        Adjusted array.
    """
    if swaps:
        in_axis, out_axis = zip(*swaps)
        arr = np.moveaxis(arr, in_axis, out_axis)
    if flips:
        arr = np.flip(arr, axis=flips)
    return arr


def reorder_scale(scale: List[float], swaps: List[Tuple[int, int]]) -> List[float]:
    """
    Apply the SAME axis permutation to a per-axis vector (voxel spacing) that
    adjust_array's np.moveaxis(swaps) applies to the image, so the spacing stays
    locked to the reoriented image axes. Swaps only -- flips do not change voxel
    size. Mirrors the reference capsule's move_columns fix.

    Parameters
    ----------
    scale : List[float]
        Per-axis voxel spacing, in the raw (pre-swap) array axis order.
    swaps : List[Tuple[int, int]]
        Axis pairs (from get_adjustments), same list adjust_array uses.

    Returns
    -------
    List[float]
        Spacing reordered to match the reoriented image axes.
    """
    swaps = [(i, o) for (i, o) in swaps if i != o]
    if not swaps:
        return list(scale)
    in_axis, out_axis = zip(*swaps)
    ndim = len(scale)
    order = [n for n in range(ndim) if n not in in_axis]   # numpy.moveaxis transpose order
    for dest, src in sorted(zip(out_axis, in_axis)):
        order.insert(dest, src)
    return [scale[o] for o in order]


def perc_normalization(
    ants_img: ants.ANTsImage, 
    lower_perc: float = 2, 
    upper_perc: float = 98
) -> ants.ANTsImage:
    """
    Perform percentile normalization on an ANTs image.

    Parameters
    ----------
    ants_img : ants.ANTsImage
        Input ANTs image to normalize
    lower_perc : float
        Lower percentile for normalization (default: 2)
    upper_perc : float
        Upper percentile for normalization (default: 98)

    Returns
    -------
    ants.ANTsImage
        Normalized ANTs image
    """
    percentiles = [lower_perc, upper_perc]
    percentile_values = np.percentile(ants_img.view(), percentiles)
    assert percentile_values[1] > percentile_values[0]
    
    ants_img = (ants_img - percentile_values[0]) / (
        percentile_values[1] - percentile_values[0]
    )

    return ants_img