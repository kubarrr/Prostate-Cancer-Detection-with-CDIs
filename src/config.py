"""Paths and CDIs parameters."""
from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DICOMs (DWI per b-value lives here).
DWI_DICOM_DIR = PROJECT_ROOT / "data" / "images" / "prostatex"

# Cuocolo PROSTATEx_masks: T2/ADC NIfTI, prostate + zonal + lesion masks.
MASKS_ROOT     = PROJECT_ROOT / "PROSTATEx_masks" / "Files"
T2_IMAGE_DIR   = MASKS_ROOT / "prostate" / "Images"
ADC_IMAGE_DIR  = MASKS_ROOT / "lesions"  / "Images" / "ADC"
PROSTATE_MASK_DIR = MASKS_ROOT / "prostate" / "mask_prostate"
PZ_MASK_DIR    = MASKS_ROOT / "prostate" / "mask_pz"
TZ_MASK_DIR    = MASKS_ROOT / "prostate" / "mask_tz"
LESION_T2_DIR  = MASKS_ROOT / "lesions"  / "Masks" / "T2"
LESION_ADC_DIR = MASKS_ROOT / "lesions"  / "Masks" / "ADC"
CLASSES_CSV    = MASKS_ROOT / "lesions"  / "PROSTATEx_Classes.csv"

OUT_DIR   = Path(__file__).resolve().parent / "outputs"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# CDIs (Wong et al. 2022, Eq. 1-5).
B_NATIVE = (50, 400, 800)
B_REF    = 50
B_SYNTH  = (1000, 2000, 3000, 4000, 5000, 6000, 7000)
B_MIX    = (B_REF,) + B_SYNTH

# Signal-mixing kernel: Σ = diag(σ², σ², 0) mm², σ = 2 mm in plane.
SIGMA_INPLANE_MM = 2.0
SIGMA_THROUGH_MM = 0.0

# Healthy voxel subsampling per patient (Wong et al. 2022: 30/patient).
HEALTHY_VOXELS_PER_PATIENT = 30
RNG_SEED = 42

# DICOM series matching for DWI.
DWI_DESCS = (
    "ep2d_diff_tra_dyndist",
    "ep2d_diff_tra_dyndist_mix",
    "diffusie-3scan-4bval_fs",
)

TISSUE_COLORS = {
    "healthy": "#1f77b4",
    "insPCa":  "#ff7f0e",
    "csPCa":   "#d62728",
}
MODALITY_COLORS = {
    "CDIs":   "#d62728",
    "CDIs_t": "#8c1d40",
    "T2w":    "#7f7f7f",
    "ADC":    "#1f77b4",
}
