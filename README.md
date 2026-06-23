# aind-exaspim-ccf-registration

Registers the fused exaSPIM sample to the Allen **CCF** (sample → exaSPIM
template → CCF) with **ANTs**, restricting registration with the fused
flat-field brain mask, and maps the CCF annotation back into sample space. Runs
on Python 3.9 + ANTsPy. Part of the exaSPIM CCF-registration + soma-reg pipeline
(Code Ocean pipeline `9578158`).

## Run it standalone

```bash
cd code && ./run
```

`run` executes, in order:
1. `python -u main.py` — load fused image, mask-restrict, register at 25 µm + 10 µm, emit per-step metadata
2. `sh run_register_ccf_annotation_10um.sh` — warp the CCF annotation into sample space
3. `python -u finalize_registration_metadata.py` — emit the annotation metadata record
4. `sh run_invert_ccf_objs.sh` — reverse-transform the CCF region meshes (.obj) into sample space (non-fatal)

## Inputs
- `../data/exaspim_manifest1.json` — `zarr_multiscale.input_uri` → the fused
  `fused_ccf_ch.zarr` (the sibling `fused_mask_ch.zarr` is read for masking).
  The fused image + mask are read from S3.
- Reference data assets in `../data/`:
  `exaSPIM_template_25um/`, `exaSPIM_template_mask_25um_otsu.nii.gz`,
  `reg_exaspim_template_to_ccf_25um_v1.x/`, `allen_mouse_ccf/`,
  `exaspim_template_7subjects_nomask_10um_round6_template_only/`.
- `../data/ccf_2017_obj/` — CCF region surface meshes (`.obj`) for the obj→sample
  inversion (step 4). Optional: if absent, that step skips.

## Parameters / environment variables
- **`USE_FUSED_MASK`** (App-Panel parameter, default **`true`**) — whether to
  restrict registration with the fused flat-field brain mask (`fused_mask_ch.zarr`).
  The capsule's **first** App-Panel parameter is passed to `run` as `$1`, which
  sets this env var; accepts **`true`/`false`** (also `yes/no`, `1/0`, `on/off`).
  Anything that isn't an explicit `false`/`no`/`0`/`off` keeps the mask ON. With it
  off, registration runs unmasked (no fused mask is loaded or written). Standalone:
  `./run false` (mask off) or `./run` / `./run true` (mask on). The choice is
  recorded in the `Image atlas alignment - 25 um` `DataProcess`
  (`parameters.use_fused_mask`).
- `OUTPUT_PREFIX` — if set, the fused image + mask are read from
  `<OUTPUT_PREFIX>/<asset>/fusion/` (a scratch test dir) instead of the input
  asset; acquisition metadata is still read from the input asset.

## Outputs (to `/results/ccf_alignment/`)
- `*_to_exaSPIM_SyN_{0GenericAffine.mat,1Warp.nii.gz,1InverseWarp.nii.gz}` — transforms
- `ccf_aligned.zarr/`, `ccf_anno_to_sample/{ccf_anno_in_sample_space.nii.gz,.zarr}`
- `*_data_process.json` (per step) + `registration_metadata/{id}_fused_mask.nii.gz`
- `mask_qc/{id}_fused_mask_vs_template_mask.png` — QC overlay (results-only)
- `ccf_mesh_to_sample/` — CCF region meshes warped into sample space. The `.obj`
  files are **published to S3** under `ccf_alignment/ccf_mesh_to_sample/` by the
  upload capsule (whitelisted as `ccf_mesh_to_sample/**/*.obj`). The QC overlay
  `ccf_mesh_to_sample/qc/ccf_mesh_vs_annotation.png` is **results-only** (the
  whitelist matches only `.obj`, so the `qc/` folder is never uploaded).
- `ccf_obj_to_sample_micron/` — the same warped meshes re-expressed in the fused
  image's **true physical micrometers**, `(x, y, z)` column order, ready to import
  into viewers such as **HortaCloud**. **Results-only — NOT uploaded to S3** (it is
  intentionally absent from the upload capsule's `PUBLISH_WHITELIST`).

## CCF region meshes → sample space (step 4)

`invert_ccf_objs.py` is the **mesh analog** of the annotation inversion: instead
of resampling the CCF label *image*, it maps CCF region-mesh **vertices** through
the registration (CCF → exaSPIM template → sample) using the capsule's existing
ANTs (`ants.apply_transforms_to_points`) plus a tiny `.obj` reader/writer — **no
extra dependency**. It reuses the transforms produced earlier in the run
(`reg_exaspim_template_to_ccf_25um/*` and `{id}_to_exaSPIM_SyN_*`), so no new
registration compute. Faces/normals are preserved; only vertices are transformed.

**Direction (settled):** the two registrations are `reg(fixed=CCF, moving=template)`
and `SyN(fixed=template, moving=sample)`. For *points*, fixed→moving uses each
registration's **FORWARD** transforms `[1Warp, 0GenericAffine]` (no inversion) —
the opposite of the annotation *image* path (which uses the inverse warps),
because points travel opposite to image content. So CCF→sample is two sequential
forward point transforms: reg (C→T) then SyN (T→S). Verified by an affine-only
test (CCF region 362 → within 11 voxels of truth, vs 440–580 for every
alternative) and by overlaying the warped mesh on the actual specimen brain.

The step writes a QC overlay — `ccf_mesh_to_sample/qc/ccf_mesh_vs_annotation.png`
(warped vertices in red over `ccf_anno_in_sample_space`) — as the per-run visual
check; this overlay stays **results-only**. The warped meshes (`.obj`) are
**published to S3** at `ccf_alignment/ccf_mesh_to_sample/` (the upload capsule
whitelists `ccf_mesh_to_sample/**/*.obj` — meshes only, not the `qc/` folder);
the step's `DataProcess` record (`name="CCF meshes to sample space"`) is
aggregated into `processing.json`.

> Coordinate note for downstream viewers. The `ccf_mesh_to_sample/` `.obj` vertices
> are in the registration's native frame — numpy/OME-Zarr **(z, y, x)** array order,
> scaled by the sample's *nominal* voxel size — which overlays
> `ccf_anno_in_sample_space` directly. Viewers that read `.obj` columns as physical
> **(x, y, z)** (e.g. **HortaCloud**) need a different convention, so the step also
> emits **`ccf_obj_to_sample_micron/`** (results-only): the same meshes recovered to
> their integer voxel index and re-expressed in the fused image's **true physical
> micrometers** via the matching `fused.zarr` pyramid level
> (`index × level_scale + level_translation`), with columns reordered to `(x, y, z)`.
> The level is auto-detected from the native grid shape, so it stays correct if a
> run registers at a different level. Note the registration input
> `fused_ccf_ch.zarr` is pre-downsampled (its level 0 = `fused.zarr` level 3), so the
> loaded 10 µm grid is physically `fused.zarr` **level 5** (~23.936 µm xy), not 10 µm —
> hence the rescale from the nominal `[10.125, 13.5, 10.125]` to the true level scale.
