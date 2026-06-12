#!/usr/bin/env python3
"""Local single-mesh direction test for the CCF obj -> sample inversion.

Runs ONE .obj through ALL candidate point-transform directions and writes a QC
overlay + label-agreement score for each -- WITHOUT re-running the registration.
Reuses the exact logic in invert_ccf_objs.py so the result is faithful.

It needs the same transforms/refs as the real step, EXCEPT the reoriented
reference image: only its GEOMETRY (spacing/origin/direction/shape) is used, so
you can pass a tiny JSON of those 4 values instead of the multi-GB .nii.gz.
Print them once, in the capsule, with:

    python -c "import ants,json; im=ants.image_read('/results/ccf_alignment/registration_metadata/786400_10um_loaded_zarr_img.nii.gz'); \
print(json.dumps({'spacing':list(im.spacing),'origin':list(im.origin),'direction':[list(r) for r in im.direction],'shape':list(im.shape)}))" > reoriented_geom.json

Example (all paths local after you download the assets):

    python test_one_mesh.py \
      --mesh '362.obj' \
      --ccf-to-template-transforms reg_0GenericAffine.mat reg_1InverseWarp.nii.gz \
      --template-to-sample-transforms 786400_to_exaSPIM_SyN_0GenericAffine.mat 786400_to_exaSPIM_SyN_1InverseWarp.nii.gz \
      --ccf-template average_template_25.nii.gz \
      --acquisition acquisition.json \
      --reference-image ccf_anno_in_sample_space.nii.gz \
      --reoriented-geom reoriented_geom.json \
      --ccf-annotation annotation_10.nii.gz \
      --out-dir ./mesh_test

The forward 1Warp files must sit NEXT TO the 1InverseWarp ones (same name with
"InverseWarp"->"Warp"); the candidate builder auto-discovers them.
"""
import argparse
import json
import os

import numpy as np
import ants

# reuse the production logic verbatim
import invert_ccf_objs as ic


def _reoriented_ref(args):
    """Return an ANTsImage carrying ONLY the reoriented grid's geometry (no pixels
    needed: transform_ccf_to_sample uses spacing/origin/direction/shape only)."""
    if args.reoriented_reference:
        return ants.image_read(args.reoriented_reference)
    g = json.load(open(args.reoriented_geom))
    ref = ants.from_numpy(np.zeros(tuple(int(s) for s in g["shape"]), dtype=np.float32))
    ref.set_spacing(tuple(float(s) for s in g["spacing"]))
    ref.set_origin(tuple(float(o) for o in g["origin"]))
    ref.set_direction(np.asarray(g["direction"], dtype=float))
    return ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True, help="a single CCF .obj")
    p.add_argument("--ccf-to-template-transforms", nargs="+", required=True)
    p.add_argument("--template-to-sample-transforms", nargs="+", required=True)
    p.add_argument("--ccf-template", required=True)
    p.add_argument("--reference-image", required=True,
                   help="ccf_anno_in_sample_space.nii.gz (QC + agreement ground truth)")
    p.add_argument("--acquisition", required=True)
    p.add_argument("--ccf-annotation", default=None,
                   help="annotation_10.nii.gz; if given, also prints label-agreement per candidate")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--reoriented-reference", help="the loaded zarr nii.gz (if you have it)")
    grp.add_argument("--reoriented-geom", help="JSON {spacing,origin,direction,shape} of the loaded image")
    p.add_argument("--mesh-units-um", type=float, default=1.0)
    p.add_argument("--out-dir", default="./mesh_test")
    args = p.parse_args()

    ccf_img = ants.image_read(args.ccf_template)
    reoriented_img = _reoriented_ref(args)
    sample_arr = ants.image_read(args.reference_image).numpy()

    swaps, flips = ic._acquisition_swaps_flips(args.acquisition)
    inv_swaps = [(b, a) for (a, b) in reversed(swaps)]
    inv_flips = flips
    print(f"orientation swaps={swaps} flips={flips} -> inv_swaps={inv_swaps} inv_flips={inv_flips}")

    # optional ground-truth annotation voxels for the agreement score
    anno = lab = labels = verts_anno_um = None
    if args.ccf_annotation and os.path.exists(args.ccf_annotation):
        anno = ants.image_read(args.ccf_annotation)
        lab = anno.numpy()
        nz = np.argwhere(lab > 0)
        step = max(1, nz.shape[0] // 8000)
        vidx = nz[::step][:8000]
        labels = lab[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
        verts_anno_um = vidx.astype(float) * (np.asarray(anno.spacing, float) * 1000.0)

    lines, vidx_m, verts = ic.read_obj(args.mesh)
    print(f"{args.mesh}: {verts.shape[0]} vertices")

    candidates = ic._build_direction_candidates(
        list(args.ccf_to_template_transforms), list(args.template_to_sample_transforms))
    print(f"testing {len(candidates)} candidate directions ...")
    os.makedirs(args.out_dir, exist_ok=True)

    for label, files, inv in candidates:
        try:
            # warp the single mesh
            out_um, vox_um = ic.transform_ccf_to_sample(
                verts, files, inv, args.mesh_units_um,
                ccf_img, reoriented_img, inv_swaps, inv_flips)
            ic.write_obj(os.path.join(args.out_dir, f"{label}.obj"), lines, vidx_m, out_um)
            # agreement (uses the annotation voxel sample, the same metric as the pipeline)
            score = ""
            if labels is not None:
                a_um, a_vox = ic.transform_ccf_to_sample(
                    verts_anno_um, files, inv, args.mesh_units_um,
                    ccf_img, reoriented_img, inv_swaps, inv_flips)
                frac, n_eval = ic._neighborhood_agreement(
                    sample_arr, np.rint(a_um / a_vox).astype(int), labels)
                score = f"  agreement={frac*100:5.1f}% (n={n_eval})"
            # overlay of THIS mesh
            ic._plot_overlay(sample_arr, out_um / vox_um,
                             os.path.join(args.out_dir, f"{label}.png"),
                             f"candidate '{label}'{score}\n(single mesh red over ccf_anno_in_sample_space)")
            print(f"  {label:16s}{score}")
        except Exception as exc:
            print(f"  {label:16s}  ERROR {exc}")

    print(f"\nwrote per-candidate .obj + .png to {args.out_dir} -- open the PNGs, "
          "pick the one whose red fills the regions.")


if __name__ == "__main__":
    main()
