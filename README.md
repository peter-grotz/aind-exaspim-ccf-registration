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

## Inputs
- `../data/exaspim_manifest1.json` — `zarr_multiscale.input_uri` → the fused
  `fused_ccf_ch.zarr` (the sibling `fused_mask_ch.zarr` is read for masking).
  The fused image + mask are read from S3.
- Reference data assets in `../data/`:
  `exaSPIM_template_25um/`, `exaSPIM_template_mask_25um_otsu.nii.gz`,
  `reg_exaspim_template_to_ccf_25um_v1.x/`, `allen_mouse_ccf/`,
  `exaspim_template_7subjects_nomask_10um_round6_template_only/`.

## Environment variables (optional)
- `OUTPUT_PREFIX` — if set, the fused image + mask are read from
  `<OUTPUT_PREFIX>/<asset>/fusion/` (a scratch test dir) instead of the input
  asset; acquisition metadata is still read from the input asset.

## Outputs (to `/results/ccf_alignment/`)
- `*_to_exaSPIM_SyN_{0GenericAffine.mat,1Warp.nii.gz,1InverseWarp.nii.gz}` — transforms
- `ccf_aligned.zarr/`, `ccf_anno_to_sample/{ccf_anno_in_sample_space.nii.gz,.zarr}`
- `*_data_process.json` (per step) + `registration_metadata/{id}_fused_mask.nii.gz`
- `mask_qc/{id}_fused_mask_vs_template_mask.png` — QC overlay (results-only)
