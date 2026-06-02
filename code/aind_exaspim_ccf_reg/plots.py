"""
Plot functions for easy and fast visualiztaion of images and
regsitration results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
import ants

from aind_exaspim_ccf_reg.configs import PathLike

    
def visualize_reg(
    moving: ants.ANTsImage,
    fixed: ants.ANTsImage, 
    moved: ants.ANTsImage,
    task_name: str, 
    outprefix: PathLike
) -> None:
    """
    Plot registration results.
    """
    moved_figpath = f"{outprefix}/{task_name}"
    plot_antsimgs(moved, moved_figpath, title=task_name)
    for loc in [0, 1, 2]:
        reg_figpath = f"{outprefix}/reg_{task_name}_{loc}"
        plot_reg(moving, 
                 fixed,
                 moved, 
                 reg_figpath, 
                 task_name, 
                 loc=loc, 
                 vmin=0, 
                 vmax=1.5
                )

def plot_antsimgs(
    ants_img: ants.ANTsImage, 
    figpath: PathLike, 
    title: str = "", 
    vmin: float = 0, 
    vmax: float = 1.5
) -> None:
    """
    Plot ANTs image in three orthogonal views.

    Parameters
    ----------
    ants_img : ants.ANTsImage
        ANTs image to plot
    figpath : PathLike
        Path where the plot is going to be saved
    title : str
        Figure title
    vmin : float
        Minimum value for color scaling
    vmax : float
        Maximum value for color scaling
    """
    if figpath:
        ants_img = ants_img.numpy()
        half_size = np.array(ants_img.shape) // 2
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        
        ax[0].imshow(
            ants_img[half_size[0], :, :], cmap="gray", vmin=vmin, vmax=vmax
        )
        ax[1].imshow(
            ants_img[:, half_size[1], :], cmap="gray", vmin=vmin, vmax=vmax
        )
        im = ax[2].imshow(
            ants_img[:, :, half_size[2]],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        
        fig.suptitle(title, y=0.9)
        plt.colorbar(
            im, ax=ax.ravel().tolist(), fraction=0.1, pad=0.025, shrink=0.7
        )
        plt.savefig(figpath, bbox_inches="tight", pad_inches=0.1)
        plt.close()


def plot_mask_overlay(
    mask_a: ants.ANTsImage,
    mask_b: ants.ANTsImage,
    figpath: PathLike,
    title: str = "",
    label_a: str = "A",
    label_b: str = "B",
) -> Optional[float]:
    """Overlay two binary masks (assumed on the same grid) in three orthogonal
    mid-slices and save a PNG for QC.

    mask_a is drawn red, mask_b green, so their overlap shows as yellow. A Dice
    coefficient (2|A n B| / (|A| + |B|)) is computed and shown in the title as a
    quick quantitative measure of agreement. Returns the Dice score (or None if
    nothing was written).

    Parameters
    ----------
    mask_a, mask_b : ants.ANTsImage
        Binary (or thresholded) masks on the same voxel grid.
    figpath : PathLike
        Where to save the PNG.
    title : str
        Figure title.
    label_a, label_b : str
        Legend labels for the red/green masks.
    """
    if not figpath:
        return None

    a = (mask_a.numpy() > 0).astype(np.uint8)
    b = (mask_b.numpy() > 0).astype(np.uint8)
    if a.shape != b.shape:
        # Different grids -> can't overlay meaningfully; skip rather than guess.
        return None

    inter = float(np.logical_and(a, b).sum())
    dice = (2.0 * inter) / (float(a.sum()) + float(b.sum()) + 1e-8)

    half = np.array(a.shape) // 2
    slabs = [
        (a[half[0], :, :], b[half[0], :, :]),
        (a[:, half[1], :], b[:, half[1], :]),
        (a[:, :, half[2]], b[:, :, half[2]]),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    for k, (sa, sb) in enumerate(slabs):
        rgb = np.zeros(sa.shape + (3,), dtype=float)
        rgb[..., 0] = sa  # red   = mask_a
        rgb[..., 1] = sb  # green = mask_b  (overlap -> yellow)
        ax[k].imshow(rgb)
        ax[k].set_axis_off()

    fig.suptitle(
        f"{title}\nDice = {dice:.3f}   "
        f"(red = {label_a}, green = {label_b}, yellow = overlap)",
        y=0.98,
    )
    plt.savefig(figpath, bbox_inches="tight", pad_inches=0.1, dpi=120)
    plt.close()
    return dice


def plot_reg(
    moving: ants.ANTsImage,
    fixed: ants.ANTsImage, 
    warped: ants.ANTsImage,
    figpath: PathLike,
    title: str = "",
    loc: int = 0,
    vmin: float = 0,
    vmax: float = 1.5
) -> None:
    """
    Plot registration results: moving, fixed, deformed, overlay and difference images.

    Parameters
    ----------
    moving : ants.ANTsImage
        Moving image
    fixed : ants.ANTsImage
        Fixed image
    warped : ants.ANTsImage
        Deformed image
    figpath : PathLike
        Path where the plot is going to be saved
    title : str
        Figure title
    loc : int
        Visualization direction (0: sagittal, 1: coronal, 2: axial)
    vmin : float
        Minimum value for color scaling
    vmax : float
        Maximum value for color scaling
        
    Raises
    ------
    ValueError
        If loc is not in [0, 1, 2] or exceeds image dimensions
    """
    if loc >= len(moving.shape):
        raise ValueError(
            f"loc {loc} is not allowed, should be less than {len(moving.shape)}"
        )

    half_size_moving = np.array(moving.shape) // 2
    half_size_fixed = np.array(fixed.shape) // 2
    half_size_warped = np.array(warped.shape) // 2

    if loc == 0:
        moving = moving.view()[half_size_moving[0], :, :]
        fixed = fixed.view()[half_size_fixed[0], :, :]
        warped = warped.view()[half_size_warped[0], :, :]
        y = 0.75
    elif loc == 1:
        moving = moving.view()[:, half_size_moving[1], :]
        fixed = fixed.view()[:, half_size_fixed[1], :]
        warped = warped.view()[:, half_size_warped[1], :]
        y = 0.82
    elif loc == 2:
        moving = np.rot90(np.fliplr(moving.view()[:, :, half_size_moving[2]]))
        fixed = np.rot90(np.fliplr(fixed.view()[:, :, half_size_fixed[2]]))
        warped = np.rot90(np.fliplr(warped.view()[:, :, half_size_warped[2]]))
        y = 0.82
    else:
        raise ValueError(
            f"loc {loc} is not allowed. Allowed values are: 0, 1, 2"
        )

    # Combine deformed and fixed images to an RGB image
    overlay = np.stack((warped, fixed, warped), axis=2)
    diff = fixed - warped

    fontsize = 14

    fig, ax = plt.subplots(1, 5, figsize=(16, 6))
    ax[0].imshow(moving, cmap="gray", vmin=vmin, vmax=vmax)
    ax[1].imshow(fixed, cmap="gray", vmin=vmin, vmax=vmax)
    ax[2].imshow(warped, cmap="gray", vmin=vmin, vmax=vmax)
    ax[3].imshow(overlay)
    ax[4].imshow(diff, cmap="gray", vmin=-(vmax), vmax=vmax)

    ax[0].set_title("Moving", fontsize=fontsize)
    ax[1].set_title("Fixed", fontsize=fontsize)
    ax[2].set_title("Deformed", fontsize=fontsize)
    ax[3].set_title("Deformed Overlay Fixed", fontsize=fontsize)
    ax[4].set_title("Fixed - Deformed", fontsize=fontsize)

    fig.suptitle(title, size=18, y=y)

    if figpath:
        plt.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
        plt.close()
    else:
        fig.show()  