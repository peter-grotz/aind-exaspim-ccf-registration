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

================================  VALIDATE ON FIRST RUN  ======================
ANTs transforms points in the OPPOSITE direction from images, so the two knobs
below MUST be confirmed against the already-inverted annotation
(ccf_anno_in_sample_space) via the --reference-image QC overlay this writes:
  * TRANSFORM_DIRECTION (the transformlist order + whichtoinvert), and
  * MESH_UNITS_UM (the mesh<->physical unit/orientation convention).
If the QC overlay shows the meshes NOT tracking the annotation regions, flip the
transformlist/whichtoinvert and/or fix the unit convention, then re-run. Do not
trust/publish the output until the overlay confirms alignment.
==============================================================================
"""
import argparse
import glob
import os
from datetime import datetime, timezone

import numpy as np
import ants

try:
    import pandas as pd
except Exception:  # pandas ships with the capsule's stack; guard just in case
    pd = None


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


# ----------------------------- the point transform ----------------------------
def transform_ccf_to_sample(verts, transformlist, whichtoinvert, mesh_units_um):
    """Map CCF mesh vertices -> sample space using ANTs point transforms.

    verts: (N,3) in the mesh's native units. mesh_units_um converts to mm for
    ANTs physical space (Allen CCF meshes are typically micrometers -> /1000).
    """
    if pd is None:
        raise RuntimeError("pandas required for ants.apply_transforms_to_points")
    pts_mm = verts * (mesh_units_um / 1000.0)
    df = pd.DataFrame(pts_mm, columns=["x", "y", "z"])
    out = ants.apply_transforms_to_points(3, df, transformlist, whichtoinvert=whichtoinvert)
    out_mm = out[["x", "y", "z"]].to_numpy()
    return out_mm / (mesh_units_um / 1000.0)  # back to the mesh's native units (sample space)


def qc_overlay(reference_nii, all_verts_um, out_png, mesh_units_um):
    """Scatter transformed vertices over mid-slices of the inverted annotation
    so a human can confirm the meshes land on the annotation regions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ref = ants.image_read(reference_nii)
        arr = ref.numpy()
        sp = np.array(ref.spacing) * 1000.0  # mm -> um
        # vertices (sample-space, mesh units) -> reference voxel indices
        vox = (all_verts_um * (mesh_units_um) ) / sp  # approx; QC only
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
    p.add_argument("--reference-image", default=None,
                   help="inverted annotation in sample space (for the QC overlay)")
    p.add_argument("--mesh-units-um", type=float, default=1.0,
                   help="micrometers per mesh unit (Allen CCF meshes are usually um=1.0)")
    p.add_argument("--start-iso", default=None)
    args = p.parse_args()

    start = args.start_iso or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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

    all_verts = []
    for obj in objs:
        lines, vidx, verts = read_obj(obj)
        if verts.size == 0:
            continue
        new_verts = transform_ccf_to_sample(verts, transformlist, whichtoinvert, args.mesh_units_um)
        all_verts.append(new_verts)
        rel = os.path.relpath(obj, args.obj_dir)
        write_obj(os.path.join(out_root, rel), lines, vidx, new_verts)
    print(f"wrote {len(objs)} warped meshes to {out_root}")

    if args.reference_image and all_verts:
        qc_overlay(args.reference_image, np.vstack(all_verts),
                   os.path.join(out_root, "qc", "ccf_objs_vs_annotation.png"), args.mesh_units_um)

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
                        "transformlist": transformlist, "whichtoinvert": whichtoinvert},
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
