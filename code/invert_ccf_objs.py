#!/usr/bin/env python3
"""Reverse-transform CCF region surface meshes (.obj) into sample space.

Mesh analog of register_ccf_annotation.py: instead of resampling the CCF label
IMAGE into sample space, this maps CCF region-mesh VERTICES (points) through the
registration, reusing the capsule's existing ANTs (no new dependency). Vertices
are transformed; faces, normals and all other .obj lines are preserved verbatim.

The full chain (each step verified -- see the per-step notes below):
  1. mesh CCF array-order um  -> CCF physical   (via the CCF average_template ASL affine)
  2. CCF physical -> sample physical            (the registration warp, below)
  3. sample physical -> reoriented array index  (via the loaded zarr image affine)
  4. undo check_orientation's swaps/flips        -> native zarr array index
  5. native index -> native um                   (overlays ccf_anno_in_sample_space / fused.zarr)

THE WARP DIRECTION (step 2) -- the part that took the longest to pin down:
The two registrations are reg(fixed=CCF, moving=template) and
SyN(fixed=template, moving=sample). For POINTS, moving fixed->moving uses each
registration's FORWARD transforms [1Warp, 0GenericAffine] (NO inversion) -- the
OPPOSITE of the annotation IMAGE path (which uses [0GenericAffine(inv), 1InverseWarp]),
because points travel opposite to image content. So C->T->S is two sequential
forward point transforms: reg (C->T) then SyN (T->S).
Confirmed three ways: (a) derived from ANTs semantics, (b) an affine-only test
landed CCF region 362 within 11 voxels of its true location in
ccf_anno_in_sample_space (vs 440-580 vox for every alternative), (c) the warped
mesh overlays the actual specimen brain (fused.zarr) at the correct region.

ORIENTATION (step 4): check_orientation reorients the raw zarr BEFORE registering
(adjust_array(swaps, flips)), so the warped points land in that reoriented frame;
the annotation inversion returns to native via adjust_array_reverse. We reproduce
that exact remap for points in _reverse_orient_indices (verified element-for-element
against adjust_array_reverse). --reoriented-reference supplies the registration
grid affine; --acquisition supplies the swaps/flips to invert.

MESH_UNITS_UM (=1.0 for Allen CCF) is the only hand-set convention. The QC overlay
(ccf_mesh_to_sample/qc/ccf_mesh_vs_annotation.png) is the per-run visual check.
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


def transform_ccf_to_sample(verts, ccf_to_template_fwd, template_to_sample_fwd, mesh_units_um,
                            ccf_img, reoriented_img, inv_swaps, inv_flips):
    """Map CCF mesh vertices -> NATIVE sample space.

    DIRECTION (derived from ANTs semantics + confirmed by an affine-only test that
    landed CCF region 362 within 11 voxels of its true location in
    ccf_anno_in_sample_space, vs 440-580 vox for every alternative):

    The two registrations are reg(fixed=CCF, moving=template) and
    SyN(fixed=template, moving=sample). For POINTS, moving fixed->moving uses each
    registration's FORWARD transforms [1Warp, 0GenericAffine] (NO inversion) --
    the opposite of the annotation IMAGE path, because points travel opposite to
    image content. The path C->T->S is therefore two sequential point transforms:
      1) reg  FORWARD  (C -> T)
      2) SyN  FORWARD  (T -> S)
    applied as two calls (unambiguous composition), each whichtoinvert all-False.

    verts: (N,3) CCF array-order micrometers (x mesh_units_um -> um).
    Returns (warped_verts_um (N,3), native_voxel_um (3,), native_shape (3,)) in
    native zarr array orientation (same frame as ccf_anno_in_sample_space), scaled
    by native voxel um. native_shape is the native (z,y,x) grid size -- used to
    auto-detect the fused pyramid level for the physical-micron (Horta) export.
    """
    if pd is None:
        raise RuntimeError("pandas required for ants.apply_transforms_to_points")
    # 1) mesh um -> CCF array index -> CCF physical (applies the CCF ASL affine)
    ccf_vox_um = np.asarray(ccf_img.spacing, float) * 1000.0
    idx_ccf = (verts * mesh_units_um) / ccf_vox_um
    phys = _index_to_physical(idx_ccf, ccf_img)
    # 2) two-stage FORWARD point transform: reg (C->T) then SyN (T->S)
    df = pd.DataFrame(phys, columns=["x", "y", "z"])
    df = ants.apply_transforms_to_points(3, df, list(ccf_to_template_fwd),
                                         whichtoinvert=[False] * len(ccf_to_template_fwd))
    df = ants.apply_transforms_to_points(3, df, list(template_to_sample_fwd),
                                         whichtoinvert=[False] * len(template_to_sample_fwd))
    phys_sample = df[["x", "y", "z"]].to_numpy()
    # 3) reoriented(sample) physical -> reoriented array index (loaded zarr affine)
    idx_reor = _physical_to_index(phys_sample, reoriented_img)
    # 4) undo the registration's orientation (swaps/flips) -> native zarr indices
    reor_shape = np.asarray(reoriented_img.shape, int)
    idx_native, order = _reverse_orient_indices(idx_reor, reor_shape, inv_swaps, inv_flips)
    # 5) native index -> native um (voxel size permutes the same way as the axes)
    reor_vox_um = np.asarray(reoriented_img.spacing, float) * 1000.0
    native_vox_um = reor_vox_um[order]
    native_shape = np.asarray(reoriented_img.shape, int)[order]   # native (z,y,x) grid
    return idx_native * native_vox_um, native_vox_um, native_shape


def _plot_overlay(arr, vox, out_png, title):
    """Scatter native-voxel-index points (red) over mid-slices of `arr` (the
    inverted annotation). imshow plots sl[row=a0 -> y, col=a1 -> x]; the scatter
    MUST use the SAME mapping (x=a1, y=a0) or the points look squished/offset on
    any anisotropic slice."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        CAP = 200000
        if vox.shape[0] > CAP:
            vox = vox[:: max(1, vox.shape[0] // CAP)]
        half = np.array(arr.shape) // 2
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        for k in range(3):
            a0, a1 = [j for j in range(3) if j != k]   # in-plane axes, increasing
            sl = np.take(arr, half[k], axis=k)          # shape (len a0, len a1)
            ax[k].imshow(sl, cmap="gray")
            sel = (np.abs(vox[:, k] - half[k]) < 3)
            if sel.any():
                ax[k].scatter(vox[sel, a1], vox[sel, a0], s=0.2, c="r", alpha=0.4)
            ax[k].set_title(f"axis {k} @ slice {half[k]}", fontsize=8)
            ax[k].set_axis_off()
        fig.suptitle(title)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"wrote QC overlay {out_png}")
    except Exception as exc:
        print(f"QC overlay skipped (non-fatal): {exc}")


def qc_overlay(reference_nii, all_verts_um, out_png, native_vox_um):
    """Top-level mesh QC: warped vertices (native um) over the inverted annotation
    (native array indices) -> divide by the native voxel size to get voxels."""
    arr = ants.image_read(reference_nii).numpy()
    vox = all_verts_um / np.asarray(native_vox_um, float)
    _plot_overlay(arr, vox, out_png,
                  "CCF objs (red) warped to sample vs inverted annotation\n"
                  "VALIDATE: red should track the annotation region boundaries")


# ---- Horta-ready export: warped meshes in the fused image's TRUE physical um ----
# The native step above stores native_um = voxel_index * NOMINAL spacing (the
# [10.125,13.5,10.125] the registration set_spacing'd onto the loaded grid). That
# nominal value is NOT the fused image's real resolution: the registration input
# (fused_ccf_ch.zarr) is a pre-downsampled pyramid, so the loaded grid is physically
# fused.zarr level 5 (~23.936 um xy), not 10 um. A viewer like HortaCloud places the
# fused volume by its OWN OME-Zarr scale, so to land the mesh on the image we recover
# the integer voxel INDEX (divide out the nominal spacing) and re-express it in the
# fused image's true micrometers: physical = index * level_scale + level_translation,
# then reorder columns (z,y,x)->(x,y,z) since Horta reads .obj columns as (x,y,z).
# The pyramid level is AUTO-DETECTED by matching the native grid shape (not hardcoded),
# so this stays correct if a run registers at a different level.
def _fused_levels(fused_zarr_path):
    """Read an OME-Zarr group's multiscales[0] -> list of
    (level_path, shape_zyx, scale_zyx, translation_zyx). Used to map native voxel
    indices into the fused image's true physical micrometers."""
    import zarr
    g = zarr.open(fused_zarr_path, mode="r")
    ms = g.attrs["multiscales"][0]
    axes = [a["name"] for a in ms["axes"]]
    zyx = [axes.index(k) for k in ("z", "y", "x")]
    levels = []
    for ds in ms["datasets"]:
        cts = ds["coordinateTransformations"]
        scale = next(t["scale"] for t in cts if t["type"] == "scale")
        trans = next((t["translation"] for t in cts if t["type"] == "translation"),
                     [0.0] * len(scale))
        shp = g[ds["path"]].shape
        levels.append((ds["path"],
                       tuple(int(shp[i]) for i in zyx),
                       tuple(float(scale[i]) for i in zyx),
                       tuple(float(trans[i]) for i in zyx)))
    return levels


def to_fused_microns(verts_native_um, native_vox_um, native_shape, fused_levels):
    """native (z,y,x) um -> fused-image physical um in Horta (x,y,z) order.
    Picks the fused pyramid level whose (z,y,x) grid matches the native grid, then
    physical_um = (native_um / nominal_voxel) * level_scale + level_translation.
    Returns (horta_xyz (N,3), chosen_level tuple)."""
    target = np.asarray(native_shape, float)
    lvl = min(fused_levels, key=lambda L: float(np.abs(np.asarray(L[1]) - target).sum()))
    _, _, scale, trans = lvl
    idx = np.asarray(verts_native_um, float) / np.asarray(native_vox_um, float)   # -> voxel index
    phys = idx * np.asarray(scale, float) + np.asarray(trans, float)              # (z,y,x) um
    return phys[:, [2, 1, 0]], lvl                                                # -> (x,y,z)


def write_obj_xyz(path, lines, vert_idx, new_xyz):
    """Write an .obj with vertices replaced by new_xyz and any vertex-normal (vn)
    components reordered (z,y,x)->(x,y,z) to match the column reorder. Faces are kept
    verbatim -- the validated Horta export reorders columns only, it does not reverse
    winding (HortaCloud renders the imported surface fine without it)."""
    lines = list(lines)
    for row, i in enumerate(vert_idx):
        x, y, z = new_xyz[row]
        lines[i] = f"v {x:.6f} {y:.6f} {z:.6f}"
    for i, ln in enumerate(lines):
        if ln.startswith("vn "):
            q = ln.split()
            lines[i] = f"vn {q[3]} {q[2]} {q[1]}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")


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
                        "native-frame QC target the warped meshes overlay")
    p.add_argument("--mesh-units-um", type=float, default=1.0,
                   help="micrometers per mesh unit (Allen CCF meshes are usually um=1.0)")
    p.add_argument("--fused-zarr", default=None,
                   help="OME-Zarr the meshes should be expressed against for viewers "
                        "(the fused.zarr HortaCloud loads). If given, also writes a "
                        "physical-micron, (x,y,z)-ordered copy to ccf_obj_to_sample_micron/ "
                        "(RESULTS-ONLY -- not whitelisted for S3). Non-fatal if unreadable.")
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

    # ----- CONFIRMED point-transform direction (derived + affine-only-validated) --
    # reg(fixed=CCF, moving=template) and SyN(fixed=template, moving=sample). For
    # POINTS, fixed->moving uses each registration's FORWARD transforms
    # [1Warp, 0GenericAffine] (no inversion). Path C->T->S = reg FORWARD then SyN
    # FORWARD, two sequential calls. Confirmed: affine-only landed CCF region 362
    # within 11 voxels of its true location in ccf_anno_in_sample_space (vs
    # 440-580 vox for every other order/inversion). The run script passes the
    # FORWARD warps in [1Warp, 0GenericAffine] order.
    ccf_to_template_fwd = list(args.ccf_to_template_transforms)
    template_to_sample_fwd = list(args.template_to_sample_transforms)
    print(f"reg (C->T) FWD: {ccf_to_template_fwd}\nSyN (T->S) FWD: {template_to_sample_fwd}")

    objs = sorted(glob.glob(os.path.join(args.obj_dir, "**", "*.obj"), recursive=True))
    print(f"found {len(objs)} .obj meshes under {args.obj_dir}")
    out_root = os.path.join(args.output_dir, "ccf_mesh_to_sample")
    os.makedirs(out_root, exist_ok=True)

    # Read every mesh once and concatenate ALL vertices, so the two
    # apply_transforms_to_points calls (reg, then SyN) load the displacement fields
    # ONCE for the whole batch. A per-mesh loop over hundreds-thousands of CCF
    # meshes would reload the warps that many times (minutes -> hours).
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
    print(f"transforming {all_in.shape[0]} vertices from {len(chunks)} meshes")
    all_out, native_vox_um, native_shape = transform_ccf_to_sample(
        all_in, ccf_to_template_fwd, template_to_sample_fwd, args.mesh_units_um,
        ccf_img, reoriented_img, inv_swaps, inv_flips)

    offset = 0
    for (rel, lines, vidx), n in zip(meshes, counts):
        write_obj(os.path.join(out_root, rel), lines, vidx, all_out[offset:offset + n])
        offset += n
    print(f"wrote {len(meshes)} warped meshes to {out_root}")

    # ---- Horta-ready copies in the fused image's TRUE physical micrometers --------
    # RESULTS-ONLY: ccf_obj_to_sample_micron/ is NOT in the upload capsule's
    # PUBLISH_WHITELIST, so it stays in /results and is never pushed to S3. Skips
    # silently (non-fatal) if --fused-zarr is absent or its metadata can't be read.
    if args.fused_zarr:
        try:
            levels = _fused_levels(args.fused_zarr)
            horta, lvl = to_fused_microns(all_out, native_vox_um, native_shape, levels)
            out_micron = os.path.join(args.output_dir, "ccf_obj_to_sample_micron")
            os.makedirs(out_micron, exist_ok=True)
            offset = 0
            for (rel, lines, vidx), n in zip(meshes, counts):
                write_obj_xyz(os.path.join(out_micron, rel), lines, vidx, horta[offset:offset + n])
                offset += n
            print(f"wrote {len(meshes)} Horta-micron meshes to {out_micron} "
                  f"(native grid {tuple(int(s) for s in native_shape)} -> fused level "
                  f"'{lvl[0]}' shape{lvl[1]} scale{lvl[2]} trans{lvl[3]} um; NOT uploaded to S3)")
        except Exception as exc:
            print(f"Horta-micron export skipped (non-fatal): {exc}")

    # all_out is in native um; ccf_anno_in_sample_space is in native array indices,
    # so QC divides by the native voxel size to land vertices on the right voxels.
    qc_overlay(args.reference_image, all_out,
               os.path.join(out_root, "qc", "ccf_mesh_vs_annotation.png"), native_vox_um)

    # metadata record (stdlib helper, vendored in this capsule)
    try:
        from aind_process_record import make_data_process, write_data_process
        dp = make_data_process(
            process_type="Image atlas alignment",
            name="CCF meshes to sample space",
            start=start,
            end=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            code_url="https://codeocean.allenneuraldynamics.org/capsule/6898460/tree",
            code_name="aind-exaspim-ccf-registration",
            code_version="0.0.1",
            run_script="/code/run_invert_ccf_objs.sh",
            experimenters=["Peter Grotz"],
            parameters={"n_meshes": len(objs), "mesh_units_um": args.mesh_units_um,
                        "ccf_to_template_fwd": ccf_to_template_fwd,
                        "template_to_sample_fwd": template_to_sample_fwd,
                        "direction": "reg-FWD(C->T) then SyN-FWD(T->S), two-stage points",
                        "inv_swaps": inv_swaps, "inv_flips": inv_flips},
            output_path="ccf_alignment/ccf_mesh_to_sample/",
            notes="Reverse-transforms CCF region surface meshes into sample space "
                  "(point transform of vertices through the CCF->template->sample chain).",
        )
        write_data_process(dp, args.output_dir)
        print("wrote CCF-objs data_process.json")
    except Exception as exc:
        print(f"metadata emit failed (non-fatal): {exc}")


if __name__ == "__main__":
    main()
