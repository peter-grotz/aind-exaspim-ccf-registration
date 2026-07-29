#!/bin/bash
# Reverse-transform CCF region meshes (.obj) into sample space, reusing the
# transforms from main.py and the annotation inversion.
set -e

echo "Starting CCF obj-mesh -> sample-space inversion..."

DATA_FOLDER="../data/"
processing_manifest_file=$(find "$DATA_FOLDER" -maxdepth 1 -name "exaspim_manifest*.json" | head -n 1)
DATASET_PATH=$(awk -F'"' '/"zarr_multiscale"/{b=1} b&&/"input_uri"/{print $4;exit} b&&/}/{b=0}' "$processing_manifest_file")
# Subject id, matching the names main.py used for the transforms read below.
# Recovered from the acquisition_<id>.json main.py wrote; falls back to the asset
# name, where the platform prefix is optional (dropped in aind-data-schema v2).
ACQ_JSON=$(ls /results/ccf_alignment/registration_metadata/acquisition_*.json 2>/dev/null | head -n 1)
if [ -n "$ACQ_JSON" ]; then
  SUBJECTID=$(basename "$ACQ_JSON" .json)
  SUBJECTID="${SUBJECTID#acquisition_}"
else
  SUBJECTID=$(echo "${DATASET_PATH}" \
    | grep -oP '(?<![0-9])[0-9]{6}(?=_\d{4}-\d{2}-\d{2})' | head -n 1 || true)
fi
# Viewer target for the micron mesh export: rewrite fused_ccf_ch.zarr to the
# sibling fused.zarr that HortaCloud loads. sed leaves DATASET_PATH unchanged
# if it doesn't match.
FUSED_ZARR=$(printf '%s' "$DATASET_PATH" | sed 's#fused_ccf_ch\.zarr#fused.zarr#')
echo "DATASET_PATH=${DATASET_PATH}  SUBJECTID=${SUBJECTID}  FUSED_ZARR=${FUSED_ZARR}"

# CCF region meshes (data asset connected to this capsule).
OBJ_DIR="../data/ccf_2017_obj"

# Forward transforms in [1Warp, 0GenericAffine] order, no inversion. Points use
# the forward transforms (opposite of the annotation image path, which uses the
# inverse warps).
CCF_TO_TEMPLATE_1="/data/reg_exaspim_template_to_ccf_25um_v1.4/1Warp.nii.gz"
CCF_TO_TEMPLATE_2="/data/reg_exaspim_template_to_ccf_25um_v1.4/0GenericAffine.mat"
TEMPLATE_TO_SAMPLE_1="/results/ccf_alignment/${SUBJECTID}_to_exaSPIM_SyN_1Warp.nii.gz"
TEMPLATE_TO_SAMPLE_2="/results/ccf_alignment/${SUBJECTID}_to_exaSPIM_SyN_0GenericAffine.mat"

# CCF average_template defining the CCF physical frame the mesh microns map into.
# 10 and 25 um share the same physical frame.
CCF_TEMPLATE="../data/allen_mouse_ccf/average_template/average_template_25.nii.gz"

# Loaded/sample zarr image carrying the reoriented grid affine the SyN transforms
# map into. Warped points are indexed on this grid, then swaps/flips are undone to
# reach native sample space.
REORIENTED_REFERENCE="/results/ccf_alignment/registration_metadata/${SUBJECTID}_10um_loaded_zarr_img.nii.gz"

# Acquisition metadata; source of the swaps/flips applied.
ACQUISITION="/results/ccf_alignment/registration_metadata/acquisition_${SUBJECTID}.json"

# Inverted annotation in sample space; QC target the warped meshes overlay.
REFERENCE_IMAGE="/results/ccf_alignment/ccf_anno_to_sample/ccf_anno_in_sample_space.nii.gz"

OUTPUT_DIR="/results/ccf_alignment"

if [ ! -d "$OBJ_DIR" ]; then
    echo "No $OBJ_DIR present; skipping CCF obj inversion."
    exit 0
fi

echo "Checking transform files..."
for f in "$CCF_TO_TEMPLATE_1" "$CCF_TO_TEMPLATE_2" "$TEMPLATE_TO_SAMPLE_1" "$TEMPLATE_TO_SAMPLE_2"; do
    [ -f "$f" ] && echo "Found: $f" || echo "Warning: transform not found: $f"
done

python invert_ccf_objs.py \
    --obj-dir "$OBJ_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --ccf-to-template-transforms "$CCF_TO_TEMPLATE_1" "$CCF_TO_TEMPLATE_2" \
    --template-to-sample-transforms "$TEMPLATE_TO_SAMPLE_1" "$TEMPLATE_TO_SAMPLE_2" \
    --ccf-template "$CCF_TEMPLATE" \
    --reoriented-reference "$REORIENTED_REFERENCE" \
    --acquisition "$ACQUISITION" \
    --reference-image "$REFERENCE_IMAGE" \
    --fused-zarr "$FUSED_ZARR" \
    --mesh-units-um 1.0

echo "CCF obj inversion completed!"
