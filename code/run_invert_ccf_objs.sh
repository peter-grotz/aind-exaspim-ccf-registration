#!/bin/bash
# Reverse-transform CCF region meshes (.obj) into sample space. Runs AFTER
# main.py + the annotation inversion, reusing the transforms they produced.
set -e

echo "Starting CCF obj-mesh -> sample-space inversion..."

DATA_FOLDER="../data/"
processing_manifest_file=$(find "$DATA_FOLDER" -maxdepth 1 -name "exaspim_manifest*.json" | head -n 1)
DATASET_PATH=$(awk -F'"' '/"zarr_multiscale"/{b=1} b&&/"input_uri"/{print $4;exit} b&&/}/{b=0}' "$processing_manifest_file")
SUBJECTID=$(echo "${DATASET_PATH}" | grep -oP 'exaSPIM_\K[0-9]+')
echo "DATASET_PATH=${DATASET_PATH}  SUBJECTID=${SUBJECTID}"

# CCF region meshes (data asset connected to this capsule).
OBJ_DIR="../data/ccf_2017_obj"

# Same transforms the annotation inversion uses.
CCF_TO_TEMPLATE_1="/data/reg_exaspim_template_to_ccf_25um_v1.4/0GenericAffine.mat"
CCF_TO_TEMPLATE_2="/data/reg_exaspim_template_to_ccf_25um_v1.4/1InverseWarp.nii.gz"
TEMPLATE_TO_SAMPLE_1="/results/ccf_alignment/${SUBJECTID}_to_exaSPIM_SyN_0GenericAffine.mat"
TEMPLATE_TO_SAMPLE_2="/results/ccf_alignment/${SUBJECTID}_to_exaSPIM_SyN_1InverseWarp.nii.gz"

# CCF average_template -> defines the CCF physical frame the mesh um are mapped
# into (same space the transforms were computed in). 10 vs 25 um share the same
# physical frame, so either works.
CCF_TEMPLATE="../data/allen_mouse_ccf/average_template/average_template_25.nii.gz"

# The registration's loaded/sample zarr image: carries the REORIENTED grid affine
# the SyN transforms map into (this is the same file the annotation inversion uses
# as its --sample_image_path). Warped points are converted to indices on this grid,
# then the registration's swaps/flips are undone to reach native sample space.
REORIENTED_REFERENCE="/results/ccf_alignment/registration_metadata/${SUBJECTID}_10um_loaded_zarr_img.nii.gz"

# Acquisition metadata -> source of the swaps/flips check_orientation applied.
ACQUISITION="/results/ccf_alignment/registration_metadata/acquisition_${SUBJECTID}.json"

# Already-inverted annotation in sample space -> native-frame QC target AND the
# ground truth the point-transform direction is validated against.
REFERENCE_IMAGE="/results/ccf_alignment/ccf_anno_to_sample/ccf_anno_in_sample_space.nii.gz"

# Source CCF annotation (same volume the annotation inversion used) -> lets the
# script auto-validate the transform direction by label agreement.
CCF_ANNOTATION="../data/allen_mouse_ccf/annotation/ccf_2017/annotation_10.nii.gz"

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
    --ccf-annotation "$CCF_ANNOTATION" \
    --mesh-units-um 1.0

echo "CCF obj inversion completed!"
