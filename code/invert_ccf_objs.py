#!/usr/bin/env python3
"""Reverse-transform CCF region surface meshes (.obj) into sample space.

Mesh analog of register_ccf_annotation.py. Instead of resampling the CCF label
IMAGE into sample space, this maps CCF region-mesh VERTICES (points) through the
SAME two-stage inverse chain (CCF -> exaSPIM template -> sample), using the
capsule's existing ANTs (no new dependency). Vertices are transformed; faces,
normals and all other .obj lines are preserved verbatim.

Transforms (already produced by this capsule -- the same files the annotation
inversion uses):
  CCF -> template   : reg_exaspim_template_to_ccf {0GenericAffine.mat, 1InverseWarp.nii.gz}
  template -> sample: {id}_to_exaSPIM_SyN          {0GenericAffine.mat, 1InverseWarp.nii.gz}

=============================  ORIENTATION (the real fix)  ===================
The registration reorients the raw zarr BEFORE registering (check_orientation ->
adjust_array(swaps, flips)), so the SyN transforms -- and therefore the warped
points -- live in that REORIENTED frame. The annotation inversion returns to the
native zarr frame by calling adjust_array_reverse(inv_swaps, inv_flips) on the
label array. This script reproduces that exact remap for POINTS (see
_reverse_orient_indices, verified element-for-element against adjust_array_reverse).
Two earlier bugs caused the "90-degrees-off / wrong-axis / in-front-of-the-volume"
symptom: (1) physical->index used the native 1mm-spacing ccf_anno_in_sample_space
instead of the actual reoriented registration grid, and (2) the swaps/flips were
never undone. Both are fixed: --reoriented-reference supplies the registration
grid affine, --acquisition supplies the swaps/flips to invert.

Still confirm on first run via the --reference-image QC overlay:
  * TRANSFORM_DIRECTION (transformlist order + whichtoinvert), and
  * MESH_UNITS_UM (mesh unit convention).
If the meshes still don't track the annotation regions, flip the
transformlist/whichtoinvert; the orientation itself is now handled analytically.
==============================================================================
"""
import argparse
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import ants

try:
    import pandas as pd
except Exception:  # pandas ships with the capsule's stack; guard just in case
    pd = None

# get_adjustments lives in the capsule package; we derive the SAME swaps/flips the
# registration used, so we can UNDO them on the points (see orientation note below).
from aind_exaspim_ccf_reg.preprocess import get_adjustments


# --- tiny .obj I/O: transform only "v" (vertex) lines, keep everything else ---
def read_obj(path):
    """Return (lines, vert_idx, verts) where verts is (N,3) and vert_idx maps
    each vertex row back to its line in `lines` (so we can rewrite in place)."""
    lines = open(path, "r").read().splitlines()
    verts, vert_idx = [], []
    for i, ln in enumerate(lines):
        if ln.startswith("v "):
            parts = ln.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            vert_idx.append(i)
    return lines, vert_idx, np.asarray(verts, dtype=float)


def write_obj(path, lines, vert_idx, new_verts):
    for row, i in enumerate(vert_idx):
        x, y, z = new_verts[row]
        lines[i] = f"v {x:.6f} {y:.6f} {z:.6f}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")


# ---- geometry: ANTs array-index <-> physical, applying the image's affine -----
# This is the step the old `verts/1000` omitted. The mesh is in CCF *array-order
# micrometers* (it overlays the CCF voxel grid directly), but ANTs point
# transforms work in *physical* space, and the CCF volume's affine PERMUTES +
# FLIPS axes. So we must go array -> physical (via the CCF image) before the warp
# and physical -> array (via the sample image) after, or the meshes come out
# rotated/mirrored.
def _index_to_physical(idx, img):
    sp = np.asarray(img.spacing, float)
    org = np.asarray(img.origin, float)
    d = np.asarray(img.direction, float).reshape(3, 3)
    return org + (d @ (idx * sp).T).T          # (N,3) physical (ANTs LPS) mm


def _physical_to_index(phys, img):
    sp = np.asarray(img.spacing, float)
    org = np.asarray(img.origin, float)
    d = np.asarray(img.direction, float).reshape(3, 3)
    return (np.linalg.inv(d) @ (np.asarray(phys, float) - org).T).T / sp


# ---- the orientation reversal the annotation does, expressed for POINTS --------
# Before registration, check_orientation() reorients the raw zarr with
# adjust_array(swaps, flips), so the SyN transforms (and therefore the warped
# POINTS) live in that *reoriented* frame. The annotation inversion gets back to
# the native zarr frame by calling adjust_array_reverse(inv_swaps, inv_flips) on
# the resampled label ARRAY. A point/vertex can't be "flipped/moved" as an array,
# so we reproduce the exact same index remap analytically:
#   * a flip of axis a:  index_a -> (Sr[a]-1) - index_a   (Sr = reoriented shape)
#   * the axis move:     n[i] = flipped[order[i]]         (order = np.moveaxis order)
# This map was verified element-for-element against adjust_array_reverse for every
# swap/flip configuration (incl. the observed [(2,0),(1,2),(0,1)]).
def _acquisition_swaps_flips(acquisition_path):
    """Return (swaps, flips) exactly as the registration computed them, by
    reading the acquisition metadata and the beta/alpha-scope direction map."""
    with open(acquisition_path, "r") as f:
        metadata = json.load(f)
    first_tile = metadata["tiles"][0]["file_name"]
    if "tile_000000_ch_" in first_tile:  # beta scope
        ccf_directions = {0: "Anterior_to_posterior", 1: "Superior_to_inferior", 2: "Left_to_right"}
    else:                                # alpha scope
        ccf_directions = {0: "Posterior_to_anterior", 1: "Inferior_to_superior", 2: "Left_to_right"}
    return get_adjustments(metadata["axes"], ccf_directions)


def _moveaxis_order(ndim, source, destination):
    """The transpose order np.moveaxis(a, source, destination) uses internally."""
    order = [n for n in range(ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return order


def _reverse_orient_indices(idx_reor, reor_shape, inv_swaps, inv_flips):
    """Map reoriented-frame array indices -> native zarr-frame array indices,
    reproducing adjust_array_reverse(arr, inv_swaps, inv_flips). Also returns the
    axis `order` so the caller can permute the voxel size the same way."""
    r = np.asarray(idx_reor, float).copy()
    for a in inv_flips:
        r[:, a] = (reor_shape[a] - 1) - r[:, a]
    if inv_swaps:
        in_ax, out_ax = zip(*inv_swaps)
        order = _moveaxis_order(3, in_ax, out_ax)
    else:
        order = [0, 1, 2]
    return r[:, order], order


def transform_ccf_to_sample(verts, transformlist, whichtoinvert, mesh_units_um,
                            ccf_img, reoriented_img, inv_swaps, inv_flips):
    """Map CCF mesh vertices -> NATIVE sample space.

    verts: (N,3) CCF array-order micrometers (x mesh_units_um -> um).
    Returns (warped_verts_um (N,3), native_voxel_um (3,)) where warped_verts_um is
    in the native zarr array orientation (same frame as ccf_anno_in_sample_space),
    scaled by the native voxel size, so it overlays the sample volume the way the
    input overlaid the CCF volume.
    """
    if pd is None:
        raise RuntimeError("pandas required for ants.apply_transforms_to_points")
    # 1) mesh um -> CCF array index -> CCF physical (applies the CCF affine)
    ccf_vox_um = np.asarray(ccf_img.spacing, float) * 1000.0
    idx_ccf = (verts * mesh_units_um) / ccf_vox_um
    phys_ccf = _index_to_physical(idx_ccf, ccf_img)
    # 2) warp CCF physical -> REORIENTED sample physical (the frame the SyN
    #    transforms were computed in)
    df = pd.DataFrame(phys_ccf, columns=["x", "y", "z"])
    out = ants.apply_transforms_to_points(3, df, transformlist, whichtoinvert=whichtoinvert)
    phys_sample = out[["x", "y", "z"]].to_numpy()
    # 3) reoriented physical -> reoriented array index, using the ACTUAL registration
    #    grid affine (loaded zarr image: same oriented direction/origin the SyN used).
    idx_reor = _physical_to_index(phys_sample, reoriented_img)
    # 4) undo the registration's orientation (swaps/flips) -> native zarr indices
    reor_shape = np.asarray(reoriented_img.shape, int)
    idx_native, order = _reverse_orient_indices(idx_reor, reor_shape, inv_swaps, inv_flips)
    # 5) native index -> native um (voxel size permutes the same way as the axes)
    reor_vox_um = np.asarray(reoriented_img.spacing, float) * 1000.0
    native_vox_um = reor_vox_um[order]
    return idx_native * native_vox_um, native_vox_um


def qc_overlay(reference_nii, all_verts_um, out_png, native_vox_um):
    """Scatter transformed vertices over mid-slices of the inverted annotation
    so a human can confirm the meshes land on the annotation regions. The output
    meshes are in native um; ccf_anno_in_sample_space is in native array indices,
    so vertices map to its voxels by dividing by the native voxel size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ref = ants.image_read(reference_nii)
        arr = ref.numpy()
        # native um -> native voxel index (same frame/shape as the reference array)
        vox = all_verts_um / np.asarray(native_vox_um, float)
        # cap plotted points so the scatter stays fast for dense meshes
        CAP = 200000
        if vox.shape[0] > CAP:
            vox = vox[:: max(1, vox.shape[0] // CAP)]
        half = np.array(arr.shape) // 2
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        for k in range(3):
            sl = np.take(arr, half[k], axis=k)
            ax[k].imshow(sl.T if k == 0 else sl, cmap="gray")
            a = [j for j in range(3) if j != k]
            sel = (np.abs(vox[:, k] - half[k]) < 3)
            if sel.any():
                ax[k].scatter(vox[sel, a[0]], vox[sel, a[1]], s=0.2, c="r", alpha=0.4)
            ax[k].set_axis_off()
        fig.suptitle("CCF objs (red) warped to sample vs inverted annotation\n"
                     "VALIDATE: red should track the annotation region boundaries")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"wrote QC overlay {out_png}")
    except Exception as exc:
        print(f"QC overlay skipped (non-fatal): {exc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj-dir", required=True, help="folder of CCF .obj meshes (ccf_2017_obj)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ccf-to-template-transforms", nargs="+", required=True)
    p.add_argument("--template-to-sample-transforms", nargs="+", required=True)
    p.add_argument("--ccf-template", required=True,
                   help="CCF average_template image (defines the CCF physical frame the "
                        "mesh um are converted into; same space the transforms were computed in)")
    p.add_argument("--reoriented-reference", required=True,
                   help="the registration's loaded/sample zarr image "
                        "(<id>_10um_loaded_zarr_img.nii.gz) -- carries the REORIENTED grid "
                        "affine (direction/origin) the SyN transforms map into; used for "
                        "physical->index before the orientation is undone")
    p.add_argument("--acquisition", required=True,
                   help="acquisition_<id>.json -- source of the swaps/flips the registration "
                        "applied (check_orientation); we invert them to return to native space")
    p.add_argument("--reference-image", required=True,
                   help="inverted annotation in sample space (ccf_anno_in_sample_space.nii.gz); "
                        "native-frame QC target the warped meshes must overlay")
    p.add_argument("--mesh-units-um", type=float, default=1.0,
                   help="micrometers per mesh unit (Allen CCF meshes are usually um=1.0)")
    p.add_argument("--start-iso", default=None)
    args = p.parse_args()

    start = args.start_iso or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Affines: CCF (input) frame, and the REORIENTED registration grid the points
    # land in. The native output frame is recovered by undoing the swaps/flips.
    ccf_img = ants.image_read(args.ccf_template)
    reoriented_img = ants.image_read(args.reoriented_reference)

    # Same swaps/flips the registration applied, inverted exactly as the annotation
    # inversion does (register_ccf_annotation.py): flips unchanged, swaps reversed.
    swaps, flips = _acquisition_swaps_flips(args.acquisition)
    inv_swaps = [(b, a) for (a, b) in reversed(swaps)]
    inv_flips = flips
    print(f"orientation swaps={swaps} flips={flips} -> inv_swaps={inv_swaps} inv_flips={inv_flips}")

    # ----- VALIDATE ON FIRST RUN: order + whichtoinvert for POINTS -----
    # Mirror of the annotation image chain (CCF->template->sample, each affine
    # inverted + inverse-warp), expressed for points. Confirm via QC; flip if needed.
    transformlist = list(args.ccf_to_template_transforms) + list(args.template_to_sample_transforms)
    whichtoinvert = [True, False, True, False]
    print(f"transformlist={transformlist}\nwhichtoinvert={whichtoinvert}")

    objs = sorted(glob.glob(os.path.join(args.obj_dir, "**", "*.obj"), recursive=True))
    print(f"found {len(objs)} .obj meshes under {args.obj_dir}")
    out_root = os.path.join(args.output_dir, "ccf_obj_to_sample")
    os.makedirs(out_root, exist_ok=True)

    # Read every mesh once, concatenate all vertices, and transform them in a
    # SINGLE apply_transforms_to_points call. This is the key cost control: each
    # call re-reads the warp displacement fields + spawns the ANTs point binary,
    # so a per-mesh loop over hundreds-thousands of CCF meshes would reload the
    # warps that many times (minutes -> hours). Batched, it is one warp load.
    meshes, chunks = [], []   # meshes: (rel_path, lines, vert_idx); chunks: (Ni,3) verts
    for obj in objs:
        lines, vidx, verts = read_obj(obj)
        if verts.size == 0:
            continue
        meshes.append((os.path.relpath(obj, args.obj_dir), lines, vidx))
        chunks.append(verts)
    if not chunks:
        print("no vertices found in any .obj; nothing to do")
        return

    counts = [c.shape[0] for c in chunks]
    all_in = np.vstack(chunks)
    print(f"transforming {all_in.shape[0]} vertices from {len(chunks)} meshes in ONE call")
    all_out, native_vox_um = transform_ccf_to_sample(
        all_in, transformlist, whichtoinvert, args.mesh_units_um,
        ccf_img, reoriented_img, inv_swaps, inv_flips)

    offset = 0
    for (rel, lines, vidx), n in zip(meshes, counts):
        write_obj(os.path.join(out_root, rel), lines, vidx, all_out[offset:offset + n])
        offset += n
    print(f"wrote {len(meshes)} warped meshes to {out_root}")

    # all_out is in native um; ccf_anno_in_sample_space is in native array indices,
    # so QC divides by the native voxel size to land vertices on the right voxels.
    qc_overlay(args.reference_image, all_out,
               os.path.join(out_root, "qc", "ccf_objs_vs_annotation.png"), native_vox_um)

    # metadata record (stdlib helper, vendored in this capsule)
    try:
        from aind_process_record import make_data_process, write_data_process
        dp = make_data_process(
            process_type="Image atlas alignment",
            name="CCF objects to sample space",
            start=start,
            end=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            code_url="https://github.com/AllenNeuralDynamics/aind-exaspim-ccf-registration.git",
            code_name="aind-exaspim-ccf-registration",
            code_version="0.0.1",
            run_script="/code/run_invert_ccf_objs.sh",
            experimenters=["Peter Grotz"],
            parameters={"n_meshes": len(objs), "mesh_units_um": args.mesh_units_um,
                        "transformlist": transformlist, "whichtoinvert": whichtoinvert,
                        "inv_swaps": inv_swaps, "inv_flips": inv_flips},
            output_path="ccf_alignment/ccf_obj_to_sample/",
            notes="Reverse-transforms CCF region surface meshes into sample space "
                  "(point transform of vertices through the CCF->template->sample chain).",
        )
        write_data_process(dp, args.output_dir)
        print("wrote CCF-objs data_process.json")
    except Exception as exc:
        print(f"metadata emit failed (non-fatal): {exc}")


if __name__ == "__main__":
    main()
