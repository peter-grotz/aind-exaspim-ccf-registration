#!/bin/bash

# Script to run CCF annotation registration
# This script calls register_ccf_annotation.py with all necessary parameters

set -e  # Exit on any error

echo "Starting CCF annotation-to-sample registration..."

# # Dataset information
# SUBJECTID="720165"
# DATASET="exaSPIM_720165_2024-07-20_20-55-58_flatfield-correction_2025-03-27_18-48-01_fusion_2025-03-28_01-53-40"

# SUBJECTID="718168"
# DATASET="exaSPIM_718168_2025-03-18_15-31-16_flatfield-correction_2025-05-09_13-40-01_fusion_2025-07-10_16-54-42"
# DATASET_PATH="s3://aind-open-data/${DATASET}/fused.zarr"

DATA_FOLDER="../data/"
# Find the first JSON file in DATA_FOLDER
processing_manifest_file=$(find "$DATA_FOLDER" -maxdepth 1 -name "exaspim_manifest*.json" | head -n 1)
echo "processing_manifest_file: ${processing_manifest_file}"


DATASET_PATH=$(
  awk -F'"' '
    /"zarr_multiscale"/ {inblk=1}
    inblk && /"input_uri"/ {print $4; exit}
    inblk && /}/ {inblk=0}
  ' "$processing_manifest_file"
)

echo "DATASET_PATH: ${DATASET_PATH}"


SUBJECTID=$(echo "${DATASET_PATH}" \
  | grep -oP 'exaSPIM_\K[0-9]+')
echo "SUBJECTID: ${SUBJECTID}"


# SUBJECTID=$(find /results/ccf_alignment/ -type f -name "*_to_exaSPIM_SyN_0GenericAffine.mat" \
#      -exec basename {} \; | cut -d'_' -f1)
# echo "$SUBJECTID"



# Output settings
SEG_PATH="/results/ccf_alignment/ccf_anno_to_sample/"
BUCKET_PATH="aind-scratch-data/di.wang"
LEVEL=2
NEW_DATASET_NAME="${DATASET}/ccf_anno_in_sample_space.zarr"


# Input paths
CCF_ANNOTATION_PATH="../data/allen_mouse_ccf/annotation/ccf_2017/annotation_10.nii.gz"
CCF_TEMPLATE_PATH="../data/allen_mouse_ccf/average_template/average_template_10.nii.gz"
# EXASPIM_TEMPLATE_PATH="../data/exaSPIM_template_25um/exaspim_template_7sujects_nomask_25um_round6.nii.gz"
EXASPIM_TEMPLATE_PATH="../data/exaspim_template_7subjects_nomask_10um_round6_template_only/fixed_median.nii.gz"
RESAMPLED_IMAGE_PATH="../results/ccf_alignment/registration_metadata/${SUBJECTID}_10um_resampled_zarr_img.nii.gz"
SAMPLE_IMAGE_PATH="../results/ccf_alignment/registration_metadata/${SUBJECTID}_10um_loaded_zarr_img.nii.gz"

# Acquisition metadata
ACQUISITION_PATH="/results/ccf_alignment/registration_metadata/acquisition_${SUBJECTID}.json"

# Transform paths
CCF_TO_TEMPLATE_TRANSFORM_1="/data/reg_exaspim_template_to_ccf_25um_v1.4/0GenericAffine.mat"
CCF_TO_TEMPLATE_TRANSFORM_2="/data/reg_exaspim_template_to_ccf_25um_v1.4/1InverseWarp.nii.gz"

TEMPLATE_TO_SAMPLE_TRANSFORM_1="/results/ccf_alignment/${SUBJECTID}_to_exaSPIM_SyN_0GenericAffine.mat"
TEMPLATE_TO_SAMPLE_TRANSFORM_2="/results/ccf_alignment/${SUBJECTID}_to_exaSPIM_SyN_1InverseWarp.nii.gz"

# Check if required files exist
echo "Checking input files..."
for file in "$CCF_ANNOTATION_PATH" "$CCF_TEMPLATE_PATH" "$EXASPIM_TEMPLATE_PATH" "$RESAMPLED_IMAGE_PATH" "$SAMPLE_IMAGE_PATH" "$ACQUISITION_PATH"; do
    if [ ! -f "$file" ]; then
        echo "Warning: File not found: $file"
    else
        echo "Found: $file"
    fi
done

# Check transform files
echo "Checking transform files..."
for transform in "$CCF_TO_TEMPLATE_TRANSFORM_1" "$CCF_TO_TEMPLATE_TRANSFORM_2" "$TEMPLATE_TO_SAMPLE_TRANSFORM_1" "$TEMPLATE_TO_SAMPLE_TRANSFORM_2"; do
    if [ ! -f "$transform" ]; then
        echo "Warning: Transform file not found: $transform"
    else
        echo "Found: $transform"
    fi
done

# Create output directory if it doesn't exist
mkdir -p "$SEG_PATH"

echo "Running CCF annotation registration..."
python register_ccf_annotation.py \
    --ccf_annotation_path "$CCF_ANNOTATION_PATH" \
    --ccf_template_path "$CCF_TEMPLATE_PATH" \
    --exaspim_template_path "$EXASPIM_TEMPLATE_PATH" \
    --resampled_image_path "$RESAMPLED_IMAGE_PATH" \
    --sample_image_path "$SAMPLE_IMAGE_PATH" \
    --ccf_to_template_transforms "$CCF_TO_TEMPLATE_TRANSFORM_1" "$CCF_TO_TEMPLATE_TRANSFORM_2" \
    --template_to_sample_transforms "$TEMPLATE_TO_SAMPLE_TRANSFORM_1" "$TEMPLATE_TO_SAMPLE_TRANSFORM_2" \
    --acquisition_path "$ACQUISITION_PATH" \
    --dataset_path "$DATASET_PATH" \
    --level "$LEVEL" \
    --seg_path "$SEG_PATH" \
    --bucket_path "$BUCKET_PATH" \
    --new_dataset_name "$NEW_DATASET_NAME" \
    --show_visualizations

echo "CCF annotation registration completed!" 