#!/usr/bin/env python3
"""Validate the CCF-mesh -> sample point transform against the proven ORACLE.

Run this IN the Code Ocean registration capsule (cloud workstation), after a
normal run has populated /results, so the real transforms + the annotation
oracle are present and the warp can actually load.

It answers the open unknowns by MEASUREMENT (not assumption):

  Test A  headers/geometry of every file the registration/annotation uses
          -> resolves the CCF-frame contradiction + whether the 10um and 25um
             CCF templates are stored in different orientations.

  Test B  LABEL-AGREEMENT against the oracle (the decisive test):
          sample labeled CCF voxels from annotation_10, warp each as a POINT
          through each candidate transform config, look up the label in
          ccf_anno_in_sample_space at the landing voxel, score the % that match.
          The annotation image path is known-correct, so the config that
          reproduces it IS the correct convention. Unconfounded (no bbox/surface
          artifacts). -> resolves CCF frame, per-leg direction, output frame.

  Test C  one combined apply_transforms_to_points call vs two sequential
          (per-leg) calls -> resolves whether the 25um/10um template junction
          needs explicit handling.

Outputs a ranked table; the top row is the recipe to hardcode into
invert_ccf_objs.py. Nothing is written to S3.

Defaults assume the standard capsule layout; override with flags if needed.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import ants

try:
    import pandas as pd
except Exception:
    pd = None

import invert_ccf_objs as ic   # reuse the proven helpers


def _find(*cands):
    for c in cands:
        hits = sorted(glob.glob(c))
        if hits:
            return hits[0]
    return None


def discover(args):
    """Locate the real files in the capsule (or use overrides)."""
    syn_aff = args.syn_affine or _find("/results/ccf_alignment/*_to_exaSPIM_SyN_0GenericAffine.mat")
    sid = None
    if syn_aff:
        base = os.path.basename(syn_aff)
        sid = base.split("_to_exaSPIM")[0]
    p = {
        "syn_aff": syn_aff,
        "syn_iwarp": args.syn_iwarp or _find("/results/ccf_alignment/*_to_exaSPIM_SyN_1InverseWarp.nii.gz"),
        "syn_warp":  _find("/results/ccf_alignment/*_to_exaSPIM_SyN_1Warp.nii.gz"),
        "reg_aff":   args.reg_affine or _find("/data/reg_exaspim_template_to_ccf_25um*/0GenericAffine.mat"),
        "reg_iwarp": args.reg_iwarp or _find("/data/reg_exaspim_template_to_ccf_25um*/1InverseWarp.nii.gz"),
        "reg_warp":  _find("/data/reg_exaspim_template_to_ccf_25um*/1Warp.nii.gz"),
        "tmpl25": _find("/data/allen_mouse_ccf/average_template/average_template_25.nii.gz"),
        "tmpl10": _find("/data/allen_mouse_ccf/average_template/average_template_10.nii.gz"),
        "anno10": args.annotation or _find("/data/allen_mouse_ccf/annotation/ccf_2017/annotation_10.nii.gz"),
        "loaded": args.loaded or _find(f"/results/ccf_alignment/registration_metadata/*_10um_loaded_zarr_img.nii.gz"),
        "acq":    args.acquisition or _find("/results/ccf_alignment/registration_metadata/acquisition_*.json"),
        "oracle": args.oracle or _find("/results/ccf_alignment/ccf_anno_to_sample/ccf_anno_in_sample_space.nii.gz"),
        "sid": sid,
    }
    return p


def test_A(paths):
    print("\n================ TEST A: file geometry ================")
    for key in ["tmpl25", "tmpl10", "anno10", "loaded", "oracle"]:
        f = paths.get(key)
        if not f or not os.path.exists(f):
            print(f"  {key:8s}: MISSING ({f})")
            continue
        h = ants.image_header_info(f)
        d = np.asarray(h["direction"]).reshape(3, 3)
        print(f"  {key:8s}: shape={tuple(int(x) for x in h['dimensions'])} "
              f"spacing={[round(float(x),5) for x in h['spacing']]} "
              f"origin={[round(float(x),3) for x in h['origin']]}")
        print(f"           direction={d.tolist()}")
    print("  NOTE: if tmpl10 and tmpl25 directions differ, that is the local-vs-capsule")
    print("        CCF-orientation discrepancy. The mesh must use the frame the warp expects.")


def _agreement(oracle, idx, labels, tol=1):
    S = np.array(oracle.shape)
    inb = np.all((idx >= 0) & (idx < S), axis=1)
    if inb.sum() == 0:
        return 0.0, 0.0, 0
    base = np.clip(idx, 0, S - 1)
    exact = (oracle[base[:, 0], base[:, 1], base[:, 2]] == labels)
    near = np.zeros(idx.shape[0], bool)
    rng = range(-tol, tol + 1)
    for di in rng:
        for dj in rng:
            for dk in rng:
                j = np.clip(base + np.array([di, dj, dk]), 0, S - 1)
                near |= (oracle[j[:, 0], j[:, 1], j[:, 2]] == labels)
    m = inb
    return float(exact[m].mean()), float(near[m].mean()), int(m.sum())


def candidates(reg_aff, reg_iwarp, reg_warp, syn_aff, syn_iwarp, syn_warp):
    """Direction configs: inverse-warp (image convention) + forward-warp variants."""
    c = [
        ("inv:reg,syn",  [reg_aff, reg_iwarp, syn_aff, syn_iwarp], [True, False, True, False]),
        ("inv:syn,reg",  [syn_aff, syn_iwarp, reg_aff, reg_iwarp], [True, False, True, False]),
    ]
    if reg_warp and syn_warp:
        c += [
            ("fwd-wa:reg,syn", [reg_warp, reg_aff, syn_warp, syn_aff], [False, False, False, False]),
            ("fwd-wa:syn,reg", [syn_warp, syn_aff, reg_warp, reg_aff], [False, False, False, False]),
            ("fwd-aw:reg,syn", [reg_aff, reg_warp, syn_aff, syn_warp], [False, False, False, False]),
            ("fwd-aw:syn,reg", [syn_aff, syn_warp, reg_aff, reg_warp], [False, False, False, False]),
        ]
    return c


def test_BC(paths, n=6000):
    print("\n================ TEST B/C: label-agreement vs oracle ================")
    if pd is None:
        print("  pandas unavailable; cannot run apply_transforms_to_points."); return
    for k in ["anno10", "oracle", "loaded", "acq", "reg_aff", "reg_iwarp", "syn_aff", "syn_iwarp"]:
        if not paths.get(k) or not os.path.exists(paths[k]):
            print(f"  MISSING required input '{k}' ({paths.get(k)}); aborting Test B."); return

    # CCF source = annotation_10 (its OWN affine defines the CCF physical frame the
    # meshes also live in: mesh_um overlays the 10um CCF grid).
    anno = ants.image_read(paths["anno10"])
    lab = anno.numpy()
    nz = np.argwhere(lab > 0)
    step = max(1, nz.shape[0] // n)
    vidx = nz[::step][:n]
    labels = lab[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
    phys_ccf = ic._index_to_physical(vidx.astype(float), anno)   # CCF voxel idx -> CCF physical

    reor = ants.image_read(paths["loaded"])                       # output (sample) grid geometry
    oracle = ants.image_read(paths["oracle"]).numpy()
    swaps, flips = ic._acquisition_swaps_flips(paths["acq"])
    inv_swaps = [(b, a) for (a, b) in reversed(swaps)]
    inv_flips = flips
    print(f"  source: {vidx.shape[0]} labeled CCF voxels | oracle shape {oracle.shape} | "
          f"inv_swaps={inv_swaps} inv_flips={inv_flips}")

    cands = candidates(paths["reg_aff"], paths["reg_iwarp"], paths["reg_warp"],
                       paths["syn_aff"], paths["syn_iwarp"], paths["syn_warp"])
    df = pd.DataFrame(phys_ccf, columns=["x", "y", "z"])
    rows = []
    for label, files, inv in cands:
        try:
            out = ants.apply_transforms_to_points(3, df, files, whichtoinvert=inv)
            phys_s = out[["x", "y", "z"]].to_numpy()
            idx_reor = ic._physical_to_index(phys_s, reor)
            idx_nat, _ = ic._reverse_orient_indices(idx_reor, np.array(reor.shape), inv_swaps, inv_flips)
            ex, nr, nval = _agreement(oracle, np.rint(idx_nat).astype(int), labels)
            rows.append((nr, ex, label, files, inv, nval))
            print(f"  [B] {label:16s} exact={ex*100:5.1f}%  within-1vox={nr*100:5.1f}%  (n={nval})")
        except Exception as exc:
            rows.append((-1, -1, label, files, inv, 0))
            print(f"  [B] {label:16s} ERROR {exc}")

    rows.sort(reverse=True)
    best = rows[0]
    print(f"\n  >>> BEST direction: '{best[2]}'  within-1vox={best[0]*100:.1f}%  exact={best[1]*100:.1f}%")
    print(f"      transformlist={best[3]}\n      whichtoinvert={best[4]}")

    # Test C: two sequential calls (per-leg) for the best direction's files, to check
    # whether the template junction needs explicit per-leg handling.
    if best[0] > 0:
        files, inv = best[3], best[4]
        try:
            o1 = ants.apply_transforms_to_points(3, df, files[:2], whichtoinvert=inv[:2])
            o2 = ants.apply_transforms_to_points(3, o1[["x", "y", "z"]], files[2:], whichtoinvert=inv[2:])
            phys_s = o2[["x", "y", "z"]].to_numpy()
            idx_reor = ic._physical_to_index(phys_s, reor)
            idx_nat, _ = ic._reverse_orient_indices(idx_reor, np.array(reor.shape), inv_swaps, inv_flips)
            ex, nr, nval = _agreement(oracle, np.rint(idx_nat).astype(int), labels)
            print(f"\n  [C] two-call (per-leg) on best files: exact={ex*100:.1f}%  within-1vox={nr*100:.1f}%")
            print("      (compare to the one-call [B] number above; large difference => junction matters)")
        except Exception as exc:
            print(f"  [C] two-call ERROR {exc}")

    print("\n  INTERPRETATION: a high within-1vox % means that config reproduces the proven")
    print("  annotation result -> that IS the correct recipe to hardcode. If even the best is")
    print("  low, the CCF frame (Test A) is the lever: try --annotation/--ccf-template at the")
    print("  matching resolution/orientation.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation");   p.add_argument("--oracle")
    p.add_argument("--loaded");       p.add_argument("--acquisition")
    p.add_argument("--reg-affine");   p.add_argument("--reg-iwarp")
    p.add_argument("--syn-affine");   p.add_argument("--syn-iwarp")
    p.add_argument("--n", type=int, default=6000)
    args = p.parse_args()
    paths = discover(args)
    print("Discovered files (subject id =", paths.get("sid"), "):")
    for k, v in paths.items():
        if k != "sid":
            print(f"  {k:9s} {v}")
    test_A(paths)
    test_BC(paths, n=args.n)


if __name__ == "__main__":
    main()
